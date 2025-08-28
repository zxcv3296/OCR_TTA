#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import re
import csv
from collections import Counter

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance
from tqdm import tqdm

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ──────────────────────────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────────────────────────
def read_character_file(char_path):
    chars = []
    with open(char_path, 'r', encoding='utf-8') as f:
        for line in f:
            c = line.strip()
            if c:
                chars.append(c)
    return ''.join(chars)


def read_gt_file(gt_file):
    image_paths, labels = [], []
    with open(gt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, label = line.split('\t', 1)
            image_paths.append(path)
            labels.append(label)
    return image_paths, labels


def strip_module_prefix(state_dict):
    """키 앞의 'module.' 접두사를 제거"""
    return { (k.replace('module.', '', 1) if k.startswith('module.') else k): v
             for k, v in state_dict.items() }


def try_load_weights(model: nn.Module, ckpt_path: str, *, verbose: bool = True):
    """
    다양한 체크포인트 포맷을 자동 처리하여 model에 가중치 로드.
    1) dict에 'state_dict' 키가 있으면 꺼냄
    2) 'module.' 접두사 자동 처리
    3) strict 로드 실패 시 non-strict 로드로 재시도(차이점 출력)
    """
    raw = torch.load(ckpt_path, map_location='cpu')
    sd = raw.get('state_dict', raw) if isinstance(raw, dict) else raw

    # 1차: 접두사 제거 후 strict 로드
    sd_stripped = strip_module_prefix(sd)
    try:
        missing, unexpected = model.load_state_dict(sd_stripped, strict=True)
        if verbose:
            print(f"[LOAD] strict OK (keys={len(sd_stripped)})")
        return True
    except Exception as e1:
        if verbose:
            print(f"[LOAD] strict 실패 -> {e1}")

    # 2차: 접두사 그대로 strict 로드 시도(혹시 모를 맞춤형 모듈명 대응)
    try:
        missing, unexpected = model.load_state_dict(sd, strict=True)
        if verbose:
            print(f"[LOAD] strict OK with original keys (keys={len(sd)})")
        return True
    except Exception as e2:
        if verbose:
            print(f"[LOAD] strict(원본키) 실패 -> {e2}")

    # 3차: non-strict 로드(가능한 레이어만 매칭), 차이점 요약 출력
    print("[LOAD] non-strict 로 재시도합니다(가능한 가중치만 로드).")
    missing, unexpected = model.load_state_dict(sd_stripped, strict=False)
    if isinstance(missing, list) and isinstance(unexpected, list):
        print(f"  - Missing keys   : {len(missing)}개 -> 예: {missing[:10]}")
        print(f"  - Unexpected keys: {len(unexpected)}개 -> 예: {unexpected[:10]}")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 평가 루프
# ──────────────────────────────────────────────────────────────────────────────
def validation_and_save_csv(model, criterion, converter, data_loader, opt, csv_path, image_paths):
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    total_digit_count = 0
    correct_digit_count = 0
    path_idx = 0

    first_debug_printed = False

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as cf:
        writer = csv.writer(cf)
        writer.writerow(['image_path', 'ground_truth', 'prediction', 'confidence', 'correct', 'digit_correct'])

        for i, (image_tensors, raw_labels_batch) in enumerate(tqdm(data_loader, desc='Evaluating', ncols=80)):
            batch_size = image_tensors.size(0)
            length_of_data += batch_size
            image = image_tensors.to(device)
            last_raw_labels = list(raw_labels_batch)

            # Prepare for prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
            text_for_loss, length_for_loss = converter.encode(raw_labels_batch, batch_max_length=opt.batch_max_length)

            # Inference
            start_time = time.time()
            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)  # (N, T, C)
                forward_time = time.time() - start_time

                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                if opt.baiduCTC:
                    cost = criterion(preds.permute(1, 0, 2), text_for_loss, preds_size, length_for_loss) / batch_size
                else:
                    cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

                _, preds_index = preds.max(2)
                if opt.baiduCTC:
                    preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index.data, preds_size.data)

            else:
                preds = model(image, text_for_pred, is_train=False)  # (N, T, C)
                forward_time = time.time() - start_time

                preds = preds[:, : text_for_loss.shape[1] - 1, :]
                target = text_for_loss[:, 1:]
                cost = criterion(preds.contiguous().view(-1, preds.shape[-1]),
                                 target.contiguous().view(-1))

                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            infer_time += forward_time
            valid_loss_avg.add(cost)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            # 배치별 경로 슬라이스
            batch_paths = image_paths[path_idx:path_idx + batch_size]
            path_idx += batch_size

            # 샘플 단위 평가
            for j, (img_path, gt_raw, pred, pred_max_prob) in enumerate(
                    zip(batch_paths, last_raw_labels, preds_str, preds_max_prob)):

                gt = gt_raw
                # Attn 예외문자 제거
                if 'Attn' in opt.Prediction:
                    if '[s]' in pred:
                        pred = pred.split('[s]', 1)[0]
                    if '[s]' in gt:
                        gt = gt.split('[s]', 1)[0]

                # 민감도/필터링 옵션
                if opt.sensitive and opt.data_filtering_off:
                    gt_low = gt.lower()
                    pred_low = pred.lower()
                    alpha_num = '0123456789abcdefghijklmnopqrstuvwxyz'
                    gt_low = re.sub(f'[^{alpha_num}]', '', gt_low)
                    pred_low = re.sub(f'[^{alpha_num}]', '', pred_low)
                    gt, pred = gt_low, pred_low

                # 첫 샘플 디버그
                if not first_debug_printed:
                    print(f"[DEBUG] First sample ▶ GT: '{gt}' | PTH pred: '{pred}'")
                    first_debug_printed = True

                # 일반 정확도
                correct = (pred == gt)
                if correct:
                    n_correct += 1

                # 정규화 편집 거리
                if len(gt) == 0 or len(pred) == 0:
                    norm_ED += 0
                elif len(gt) > len(pred):
                    norm_ED += 1 - edit_distance(pred, gt) / len(gt)
                else:
                    norm_ED += 1 - edit_distance(pred, gt) / len(pred)

                # 숫자 정확도(샘플 내 숫자 시퀀스 완전일치 시 그 샘플의 숫자 수만큼 정답 가산)
                gt_digits = [c for c in gt if c.isdigit()]
                pred_digits = [c for c in pred if c.isdigit()]
                digit_flag = (len(gt_digits) > 0 and gt_digits == pred_digits)
                sample_digit_cnt = len(gt_digits)
                total_digit_count += sample_digit_cnt
                if digit_flag:
                    correct_digit_count += sample_digit_cnt

                # confidence 계산
                try:
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1].item()
                except Exception:
                    confidence_score = 0.0

                writer.writerow([
                    img_path,
                    gt,
                    pred,
                    f'{confidence_score:.6f}',
                    str(correct),
                    str(digit_flag)
                ])

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)
    digit_accuracy = (correct_digit_count / total_digit_count * 100) if total_digit_count > 0 else 0.0

    return accuracy, norm_ED, infer_time, length_of_data, digit_accuracy


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_single_dataset(opt):
    image_paths, _ = read_gt_file(opt.gt_file)
    char_list = read_character_file(opt.character)
    converter = CTCLabelConverter(char_list) if 'CTC' in opt.Prediction else AttnLabelConverter(char_list)
    opt.num_class = len(converter.character)
    opt.input_channel = 3 if opt.rgb else 1

    # 모델 생성(단일-GPU 기준으로 먼저 로드 → 이후 멀티GPU면 감싸기)
    model = Model(opt).to(device)

    print(f'>>> loading checkpoint from {opt.saved_model}')
    try_load_weights(model, opt.saved_model, verbose=True)

    # 멀티 GPU면 나중에 DataParallel로 감쌈(로드 이후라 접두사 문제 없음)
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    criterion = torch.nn.CTCLoss(zero_infinity=True).to(device) if 'CTC' in opt.Prediction \
        else torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

    model.eval()

    AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    eval_dataset, _ = hierarchical_dataset(root=opt.eval_data, opt=opt)
    evaluation_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_evaluation,
        pin_memory=True
    )

    csv_path = opt.csv_path
    with torch.no_grad():
        accuracy, norm_ED, infer_time, length_of_data, digit_accuracy = validation_and_save_csv(
            model, criterion, converter, evaluation_loader, opt, csv_path, image_paths
        )

    avg_time_ms = infer_time / length_of_data * 1000
    params_num = sum([np.prod(p.size()) for p in model.parameters()]) / 1e6

    print('\n' + '-' * 60)
    print(f'Dataset: {os.path.basename(opt.eval_data)}')
    print(f'  * Sequence Accuracy   : {accuracy:0.3f}%')
    print(f'  * Digit  Accuracy     : {digit_accuracy:0.3f}%')
    print(f'  * Norm. Edit Distance : {norm_ED:0.3f}')
    print(f'  * Avg. Inference Time : {avg_time_ms:0.3f} ms')
    print(f'  * # Parameters        : {params_num:0.3f} M')
    print('-' * 60 + '\n')
    print(f"CSV saved to: {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OCR Recognition Evaluation (PTH)")

    # 필수
    parser.add_argument('--eval_data',    required=True, help='LMDB root (evaluation)')
    parser.add_argument('--saved_model',  required=True, help='.pth checkpoint')
    parser.add_argument('--character',    required=True, help='characters .txt')
    parser.add_argument('--gt_file',      required=True, help='GT txt (path<TAB>label)')

    # 네트워크 옵션
    parser.add_argument('--batch_max_length', type=int, default=25)
    parser.add_argument('--imgH', type=int, default=32)
    parser.add_argument('--imgW', type=int, default=100)
    parser.add_argument('--rgb', action='store_true')
    parser.add_argument('--PAD', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=192)

    parser.add_argument('--Transformation', type=str, required=True, choices=['None', 'TPS'])
    parser.add_argument('--FeatureExtraction', type=str, required=True, choices=['VGG', 'RCNN', 'ResNet'])
    parser.add_argument('--SequenceModeling', type=str, required=True, choices=['None', 'BiLSTM'])
    parser.add_argument('--Prediction', type=str, required=True, choices=['CTC', 'Attn'])
    parser.add_argument('--num_fiducial', type=int, default=20)
    parser.add_argument('--output_channel', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=256)

    # 부가 옵션
    parser.add_argument('--sensitive', action='store_true')
    parser.add_argument('--data_filtering_off', action='store_true')
    parser.add_argument('--baiduCTC', action='store_true')

    parser.add_argument('--csv_path', type=str, default='eval_results_pth.csv')

    opt = parser.parse_args()

    # 재현성/성능 설정
    cudnn.deterministic = True
    cudnn.benchmark = False

    evaluate_single_dataset(opt)
