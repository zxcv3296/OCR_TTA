import os
import time
import string
import argparse
import re

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def benchmark_all_eval(model, criterion, converter, opt, calculate_infer_time=False):
    """ evaluation with 10 benchmark evaluation datasets """
    eval_data_list = [
        'IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
        'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80'
    ]

    if calculate_infer_time:
        evaluation_batch_size = 1
    else:
        evaluation_batch_size = opt.batch_size

    list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
    total_correct_number = 0
    os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    log = open(f'./result/{opt.exp_name}/log_all_evaluation.txt', 'a')
    dashed_line = '-' * 80
    print(dashed_line)
    log.write(dashed_line + '\n')

    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_evaluation = AlignCollate(
            imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD
        )
        eval_dataset, eval_dataset_log = hierarchical_dataset(
            root=eval_data_path, opt=opt
        )
        evaluation_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation,
            pin_memory=True
        )

        _, accuracy_by_best_model, norm_ED_by_best_model, _, _, _, infer_time, length_of_data = validation(
            model, criterion, converter, evaluation_loader, opt
        )
        list_accuracy.append(f'{accuracy_by_best_model:0.3f}')
        total_forward_time += infer_time
        total_evaluation_data_number += length_of_data
        total_correct_number += accuracy_by_best_model * length_of_data / 100.0
        log.write(eval_dataset_log)
        print(f'Acc {accuracy_by_best_model:0.3f}\t normalized_ED {norm_ED_by_best_model:0.3f}')
        log.write(f'Acc {accuracy_by_best_model:0.3f}\t normalized_ED {norm_ED_by_best_model:0.3f}\n')
        print(dashed_line)
        log.write(dashed_line + '\n')

    averaged_forward_time = total_forward_time / total_evaluation_data_number * 1000
    total_accuracy = total_correct_number / total_evaluation_data_number * 100.0
    params_num = sum([np.prod(p.size()) for p in model.parameters()])

    evaluation_log = 'accuracy: '
    for name, accuracy in zip(eval_data_list, list_accuracy):
        evaluation_log += f'{name}: {accuracy}\t'
    evaluation_log += f'total_accuracy: {total_accuracy:0.3f}\t'
    evaluation_log += f'averaged_infer_time: {averaged_forward_time:0.3f}\t# parameters: {params_num/1e6:0.3f}'
    print(evaluation_log)
    log.write(evaluation_log + '\n')
    log.close()


def validation(model, criterion, converter, evaluation_loader, opt):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    last_raw_labels = []

    for i, (image_tensors, raw_labels_batch) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data += batch_size
        image = image_tensors.to(device)

        # raw_labels_batch는 DataLoader가 내보낸 원본 GT 튜플이므로 리스트로 변환 후 복사
        raw_labels_batch = list(raw_labels_batch)
        last_raw_labels = raw_labels_batch.copy()

        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        # CTC/Attention용으로 인코더에 넘길 텍스트 인덱스를 생성
        text_for_loss, length_for_loss = converter.encode(
            raw_labels_batch, batch_max_length=opt.batch_max_length
        )

        start_time = time.time()
        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred)
            forward_time = time.time() - start_time

            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if opt.baiduCTC:
                cost = criterion(
                    preds.permute(1, 0, 2),
                    text_for_loss,
                    preds_size,
                    length_for_loss
                ) / batch_size
            else:
                cost = criterion(
                    preds.log_softmax(2).permute(1, 0, 2),
                    text_for_loss,
                    preds_size,
                    length_for_loss
                )

            if opt.baiduCTC:
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
            else:
                _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index.data, preds_size.data)

        else:
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = criterion(
                preds.contiguous().view(-1, preds.shape[-1]),
                target.contiguous().view(-1)
            )

            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # preds_prob 계산
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []

        for gt_raw, pred, pred_max_prob in zip(last_raw_labels, preds_str, preds_max_prob):
            gt = gt_raw
            if 'Attn' in opt.Prediction:
                if '[s]' in pred:
                    pred = pred.split('[s]', 1)[0]
                if '[s]' in gt:
                    gt = gt.split('[s]', 1)[0]

            if opt.sensitive and opt.data_filtering_off:
                gt = gt.lower()
                pred = pred.lower()
                alpha_num = '0123456789abcdefghijklmnopqrstuvwxyz'
                out_of_alpha = f'[^{alpha_num}]'
                gt = re.sub(out_of_alpha, '', gt)
                pred = re.sub(out_of_alpha, '', pred)

            if pred == gt:
                n_correct += 1

            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0
            confidence_score_list.append(confidence_score)

        labels_for_return = last_raw_labels.copy()
        preds_for_return = preds_str.copy()

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)

    # 마지막 배치의 raw_labels와 preds, confidence를 함께 반환
    return (
        valid_loss_avg.val(),
        accuracy,
        norm_ED,
        preds_for_return,
        confidence_score_list,
        labels_for_return,
        infer_time,
        length_of_data
    )


def test(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print(
        'model input parameters',
        opt.imgH,
        opt.imgW,
        opt.num_fiducial,
        opt.input_channel,
        opt.output_channel,
        opt.hidden_size,
        opt.num_class,
        opt.batch_max_length,
        opt.Transformation,
        opt.FeatureExtraction,
        opt.SequenceModeling,
        opt.Prediction
    )
    model = torch.nn.DataParallel(model).to(device)

    print(f'loading pretrained model from {opt.saved_model}')
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])

    os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')

    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

    model.eval()
    with torch.no_grad():
        if opt.benchmark_all_eval:
            benchmark_all_eval(model, criterion, converter, opt)
        else:
            log = open(f'./result/{opt.exp_name}/log_evaluation.txt', 'a')
            AlignCollate_evaluation = AlignCollate(
                imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD
            )
            eval_dataset, eval_dataset_log = hierarchical_dataset(
                root=opt.eval_data, opt=opt
            )
            evaluation_loader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=int(opt.workers),
                collate_fn=AlignCollate_evaluation,
                pin_memory=True
            )
            _, accuracy_by_best_model, _, _, _, _, _, _ = validation(
                model, criterion, converter, evaluation_loader, opt
            )
            log.write(eval_dataset_log)
            print(f'{accuracy_by_best_model:0.3f}')
            log.write(f'{accuracy_by_best_model:0.3f}\n')
            log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', required=True, help='path to evaluation dataset')
    parser.add_argument('--benchmark_all_eval', action='store_true',
                        help='evaluate 10 benchmark evaluation datasets')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True,
                        help="path to saved_model to evaluation")
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--Transformation', type=str, required=True,
                        help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True,
                        help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True,
                        help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20,
                        help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    if opt.sensitive:
        opt.character = string.printable[:-6]

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    test(opt)
