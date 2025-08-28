#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# ── 메모리 파편화 방지를 위한 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import os.path
import sys
import time
import random
import argparse
import codecs
import copy
import lmdb
from tqdm import trange

import torch
import torch.onnx        # ONNX export
from torch.onnx import register_custom_op_symbolic
import torch.backends.cudnn as cudnn

import torch.optim as optim
import torch.utils.data
import numpy as np
from torch.optim import AdamW
from torchvision import transforms as T
from PIL import Image

# dataset 모듈 전체를 import 해두면 patch 가능
import dataset
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from utils import (
    CTCLabelConverter,
    CTCLabelConverterForBaiduWarpctc,
    AttnLabelConverter,
    Averager
)
from model import Model
from test import validation  # validation(model, criterion, converter, loader, opt)

# 전역 변수로 현재 이터레이션과 valInterval 저장
CURRENT_ITERATION = 0
VAL_INTERVAL = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ────────────────────────────────────────────────────────────────────────────────
# ONNX 모델 저장 함수 (완전 캡슐화)
# ────────────────────────────────────────────────────────────────────────────────
def save_onnx_model(model, opt, onnx_path):
    net = model.module if isinstance(model, torch.nn.DataParallel) else model
    was_training = net.training
    orig_device  = next(net.parameters()).device

    net.eval()

    # adaptive_avg_pool2d 패치
    import torch.nn.functional as F
    _orig_pool = F.adaptive_avg_pool2d
    def _patched_pool(input, output_size):
        if isinstance(output_size, (list, tuple)):
            osize = []
            for idx, o in enumerate(output_size):
                osize.append(input.size(2 + idx) if not isinstance(o, int) else o)
            output_size = tuple(osize)
        elif not isinstance(output_size, int):
            output_size = (input.size(2), input.size(3))
        return _orig_pool(input, output_size)
    F.adaptive_avg_pool2d = _patched_pool

    # custom ONNX symbolic 등록
    def adaptive_pool_symbolic(g, input, output_size):
        return g.op("ReduceMean", input, axes_i=[2,3], keepdims_i=1)
    register_custom_op_symbolic("aten::adaptive_avg_pool2d", adaptive_pool_symbolic, 16)

    dummy_img  = torch.randn(1, opt.input_channel, opt.imgH, opt.imgW, device=orig_device)
    dummy_text = torch.zeros(1, opt.batch_max_length + 1, dtype=torch.long, device=orig_device)

    orig_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False

    try:
        with torch.no_grad():
            torch.onnx.export(
                net,
                (dummy_img, dummy_text),
                onnx_path,
                export_params=True,
                opset_version=16,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                input_names=['image', 'text'],
                output_names=['output'],
                dynamic_axes={
                    'image': {0: 'batch_size'},
                    'text':  {0: 'batch_size', 1: 'seq_len'},
                }
            )
        print(f"[INFO] ONNX model saved: {onnx_path}")
    except Exception as e:
        print(f"[WARN] ONNX 저장 실패: {e}")
    finally:
        torch.backends.cudnn.enabled = orig_cudnn
        if was_training:
            net.train()
        F.adaptive_avg_pool2d = _orig_pool

# ────────────────────────────────────────────────────────────────────────────────
# Levenshtein 거리
# ────────────────────────────────────────────────────────────────────────────────
def levenshtein(s1, s2):
    len1, len2 = len(s1), len(s2)
    dp = [list(range(len2 + 1))] + [[i] + [0] * len2 for i in range(1, len1 + 1)]
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[len1][len2]

# ────────────────────────────────────────────────────────────────────────────────
# 이미지 증강 유틸 & AlignCollateAugment
# ────────────────────────────────────────────────────────────────────────────────
def conditional_grayscale(img):
    if random.random() < 0.5:
        return T.functional.to_grayscale(img, num_output_channels=3)
    return img

def to_3channel(img):
    if img.mode == 'L':
        return img.convert('RGB')
    return img

# 마지막 노이즈 람다 제거
TRANSFORMS_LIST = [
    T.RandomRotation(15),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.8, 1.2)),
    T.RandomPerspective(distortion_scale=0.2, p=0.3),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.RandomInvert(p=0.2),
    T.RandomApply([T.Lambda(conditional_grayscale)], p=0.1),
    T.Lambda(to_3channel),
    T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
]

class AlignCollateAugment(AlignCollate):
    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        super().__init__(imgH, imgW, keep_ratio_with_pad)
        self.transforms_list = TRANSFORMS_LIST

    def __call__(self, batch):
        global CURRENT_ITERATION, VAL_INTERVAL
        batch = list(filter(lambda x: x is not None, batch))
        images, labels = zip(*batch)

        applied = []
        augmented_images = []
        for img in images:
            selected = random.sample(self.transforms_list, 3)
            for aug in selected:
                img = aug(img)
                applied.append(aug)
            augmented_images.append(img)

        if VAL_INTERVAL and (CURRENT_ITERATION % VAL_INTERVAL == 0 or CURRENT_ITERATION == 1):
            names = sorted({type(aug).__name__ for aug in applied})
            print(f"[Iter {CURRENT_ITERATION}] 적용된 증강: {names}")

        if self.keep_ratio_with_pad:
            resized_max_w = self.imgW
            input_channel = 3 if isinstance(augmented_images[0], Image.Image) and augmented_images[0].mode == 'RGB' else 1
            transform_norm = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in augmented_images:
                w, h = image.size
                ratio = w / float(h)
                resized_w = self.imgW if np.ceil(self.imgH * ratio) > self.imgW else int(np.ceil(self.imgH * ratio))
                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform_norm(resized_image))
            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
        else:
            transform_resize = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = torch.cat([transform_resize(image).unsqueeze(0) for image in augmented_images], 0)

        # 텐서 단계에서 가벼운 노이즈 추가
        noise = torch.randn_like(image_tensors) * 0.01
        image_tensors = image_tensors + noise

        return image_tensors, labels

def extract_unique_chars_from_lmdb(lmdb_paths):
    charset = set()
    for lmdb_root in lmdb_paths:
        env = lmdb.open(
            lmdb_root,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        with env.begin(write=False) as txn:
            raw_count = txn.get("num-samples".encode())
            if raw_count is not None:
                n_samples = int(raw_count)
                for idx in range(1, n_samples + 1):
                    label_key = f"label-{idx:09d}".encode()
                    label_bytes = txn.get(label_key)
                    if label_bytes:
                        label = label_bytes.decode("utf-8")
                        charset.update(label)
        env.close()
    return "".join(sorted(charset))

def train(opt):
    global CURRENT_ITERATION, VAL_INTERVAL
    VAL_INTERVAL = opt.valInterval

    if not opt.data_filtering_off:
        print('Filtering images with invalid chars or length > batch_max_length')

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')

    best_valid_loss = float('inf')
    no_improve_count = 0
    patience = opt.early_stop_patience

    # 학습 데이터셋 with augmentation 적용
    def make_train_dataset():
        original_align = AlignCollate
        try:
            globals()['AlignCollate'] = AlignCollateAugment
            ds = Batch_Balanced_Dataset(opt)
        finally:
            globals()['AlignCollate'] = original_align
        return ds

    train_dataset = make_train_dataset()
    os.makedirs(f'{opt.save_path}/{opt.exp_name}', exist_ok=True)

    # 검증 데이터셋 (no augmentation)
    log = open(f'{opt.save_path}/{opt.exp_name}/log_dataset.txt', 'a')
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid,
        pin_memory=True
    )
    log.write(valid_dataset_log)
    log.write('-' * 80 + '\n')
    log.close()

    # Model + Converter
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverterForBaiduWarpctc(opt.character) if opt.baiduCTC else CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    model = Model(opt)

    model = torch.nn.DataParallel(model).to(device)
    model.train()

    if opt.saved_model:
        state = torch.load(opt.saved_model)
        model.load_state_dict(state, strict=not opt.FT)

    # Loss, Optimizer, Scheduler
    if 'CTC' in opt.Prediction:
        criterion = CTCLoss() if opt.baiduCTC else torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1).to(device)
    loss_avg = Averager()

    filtered_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(filtered_parameters, lr=opt.lr, weight_decay=5e-4)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #    optimizer, mode='min', factor=0.5, patience=2, verbose=True
    #)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.num_iter) #

    start_iter = 0
    if opt.saved_model:
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    total_iters = opt.num_iter

    for iteration in trange(start_iter + 1, total_iters + 1, desc='Training'):
        CURRENT_ITERATION = iteration

        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)

        if 'CTC' in opt.Prediction:
            preds = model(image, text)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if opt.baiduCTC:
                preds = preds.permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length) / batch_size
            else:
                preds = preds.log_softmax(2).permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length)
        else:
            preds = model(image, text[:, :-1])
            target = text[:, 1:]
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        model.zero_grad()
        cost.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        loss_avg.add(cost)
        scheduler.step()

        if iteration % opt.valInterval == 0 or iteration == 1:
            elapsed_time = time.time() - start_time
            with open(f'{opt.save_path}/{opt.exp_name}/log_train.txt', 'a') as log_file:
                model.eval()
                valid_loss, current_accuracy, current_norm_ED, preds_out, conf, raw, infer_t, ldata = validation(
                    model, criterion, converter, valid_loader, opt
                )
                model.train()
                # 두 번째 코드에 맞춰 scheduler.step은 제거되었습니다

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count >= patience:
                    print(f"No improvement for {patience} validations. Early stopping at iter {iteration}.")
                    torch.save(model.state_dict(), f'{opt.save_path}/{opt.exp_name}/early_stopped_final.pth')
                    sys.exit()

                current_lr = optimizer.param_groups[0]['lr']
                loss_log = f'[{iteration}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Acc: {current_accuracy:0.3f}, Elapsed: {elapsed_time:0.5f}, LR: {current_lr: 0.9f}'
                loss_avg.reset()

                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'{opt.save_path}/{opt.exp_name}/best_accuracy.pth')
                    save_onnx_model(model, opt, f'{opt.save_path}/{opt.exp_name}/best_accuracy.onnx')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'{opt.save_path}/{opt.exp_name}/best_norm_ED.pth')
                    save_onnx_model(model, opt, f'{opt.save_path}/{opt.exp_name}/best_norm_ED.onnx')

                log_file.write(loss_log + '\n')
                print(loss_log)

        if iteration % 100000 == 0:
            torch.save(model.state_dict(), f'{opt.save_path}/{opt.exp_name}/iter_{iteration}.pth')
            save_onnx_model(model, opt, f'{opt.save_path}/{opt.exp_name}/iter_{iteration}.onnx')

        if iteration == total_iters:
            print('End of training')
            torch.save(model.state_dict(), f'{opt.save_path}/{opt.exp_name}/final.pth')
            save_onnx_model(model, opt, f'{opt.save_path}/{opt.exp_name}/final.onnx')
            sys.exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',      help='실험 이름 (logs/models)')
    parser.add_argument('--save_path',     default='./saved_models', help='모델/로그 저장 디렉토리')
    parser.add_argument('--train_data',    required=True, help='train LMDB root')
    parser.add_argument('--valid_data',    required=True, help='valid LMDB root')
    parser.add_argument('--manualSeed',    type=int, default=1111, help='random seed')
    parser.add_argument('--workers',       type=int, default=4, help='DataLoader workers')
    parser.add_argument('--batch_size',    type=int, default=128, help='batch size per GPU')
    parser.add_argument('--num_iter',      type=int, default=300000, help='total iterations')
    parser.add_argument('--valInterval',   type=int, default=2000, help='validation interval')
    parser.add_argument('--saved_model',   default='', help='continue training from')
    parser.add_argument('--FT',            action='store_true', help='fine-tuning mode')
    parser.add_argument('--adam',          action='store_true', help='AdamW or Adadelta 선택')
    parser.add_argument('--lr',            type=float, default=1e-4, help='learning rate')
    parser.add_argument('--beta1',         type=float, default=0.9, help='beta1 for Adam')
    parser.add_argument('--rho',           type=float, default=0.95, help='rho for Adadelta')
    parser.add_argument('--eps',           type=float, default=1e-8, help='eps for Adadelta')
    parser.add_argument('--grad_clip',     type=float, default=2, help='gradient clipping value')
    parser.add_argument('--baiduCTC',      action='store_true', help='BaiduWarpCTC 사용')
    parser.add_argument('--select_data',   type=str, default='custom', help='데이터 선택 (e.g. MJ-ST)')
    parser.add_argument('--batch_ratio',   type=str, default='0.5-0.5', help='batch ratio 설정')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0', help='전체 데이터 사용 비율')
    parser.add_argument('--batch_max_length',       type=int, default=20, help='최대 라벨 길이')
    parser.add_argument('--imgH',          type=int, default=32, help='입력 이미지 높이')
    parser.add_argument('--imgW',          type=int, default=100, help='입력 이미지 너비')
    parser.add_argument('--rgb',           action='store_true', help='RGB 입력 사용')
    parser.add_argument('--character',     type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='문자셋 또는 파일 경로')
    parser.add_argument('--sensitive',     action='store_true', help='대소문자 구분 모드')
    parser.add_argument('--PAD',           action='store_true', help='ratio 유지 후 패딩')
    parser.add_argument('--data_filtering_off', action='store_true', help='데이터 필터링 비활성화')
    parser.add_argument('--Transformation',    type=str, required=True, help='TPS|None')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling',  type=str, required=True, help='None|BiLSTM')
    parser.add_argument('--Prediction',        type=str, required=True, help='CTC|Attn')
    parser.add_argument('--num_fiducial',      type=int, default=20, help='TPS fiducial 수')
    parser.add_argument('--input_channel',     type=int, default=1, help='입력 채널 수')
    parser.add_argument('--output_channel',    type=int, default=512, help='출력 채널 수')
    parser.add_argument('--hidden_size',       type=int, default=256, help='LSTM hidden size')
    parser.add_argument('--pretrained', action='store_true', help='starting from pretrained model')
    parser.add_argument('--early_stop_patience', type=int, default=1000, help='early stopping patience')
    opt = parser.parse_args()

    # character 옵션이 파일 경로일 경우 파일 읽기
    if os.path.isfile(opt.character):
        chars = []
        with codecs.open(opt.character, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chars.append(line)
        opt.character = "".join(chars)
        print(f">>> Loaded character set ({len(chars)} chars) from file.")

    # LMDB에서 실제 사용되는 문자 보강
    lmdb_train_root = None
    lmdb_valid_root = None
    for dirpath, dirnames, filenames in os.walk(opt.train_data + "/"):
        if not dirnames and any(sd in dirpath for sd in opt.select_data):
            lmdb_train_root = dirpath
            break
    for dirpath, dirnames, filenames in os.walk(opt.valid_data + "/"):
        if not dirnames:
            lmdb_valid_root = dirpath
            break

    if lmdb_train_root is None:
        print("[WARN] 학습용 LMDB 말단 경로를 찾지 못했습니다.")
    if lmdb_valid_root is None:
        print("[WARN] 검증용 LMDB 말단 경로를 찾지 못했습니다.")

    all_chars_train = set(opt.character)
    if lmdb_train_root:
        all_chars_train |= set(extract_unique_chars_from_lmdb([lmdb_train_root]))
    all_chars_valid = set()
    if lmdb_valid_root:
        all_chars_valid = set(extract_unique_chars_from_lmdb([lmdb_valid_root]))

    combined_chars = "".join(sorted(all_chars_train | all_chars_valid))
    opt.character = combined_chars
    print(f">>> Final character set ({len(opt.character)} chars)")

    # 시드 고정
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)
    cudnn.benchmark = True
    cudnn.deterministic = True

    opt.num_gpu = torch.cuda.device_count()
    if opt.num_gpu > 1:
        opt.workers    *= opt.num_gpu
        opt.batch_size *= opt.num_gpu

    train(opt)
