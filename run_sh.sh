set -uo pipefail
export CUDA_VISIBLE_DEVICES=3,4
# Python 경고 메시지(DeprecationWarning, UserWarning 등) 무시
export PYTHONWARNINGS="ignore"


python3 -W ignore train9_onnx.py \
  --exp_name        . \
  --train_data      /home/kgh/ocr-tta/data_lmdb_03/train/all_generate_hub/custom \
  --valid_data      /home/kgh/ocr-tta/data_lmdb_03/val/all_change2 \
  --select_data     custom \
  --batch_ratio     1.5 \
  --Transformation  TPS \
  --FeatureExtraction ResNet \
  --SequenceModeling BiLSTM \
  --Prediction      Attn \
  --character       /home/kgh/ocr-tta/final_charlist_1770.txt \
  --num_iter        12000 \
  --valInterval     200 \
  --data_filtering_off \
  --save_path       ./saved_models_set_onnx/saved_models_9_all_generate_hub_br1.5

# 2) batch_ratio=2.5
python3 -W ignore train9_onnx.py \
  --exp_name        . \
  --train_data      /home/kgh/ocr-tta/data_lmdb_03/train/all_generate_hub/custom \
  --valid_data      /home/kgh/ocr-tta/data_lmdb_03/val/all_change2 \
  --select_data     custom \
  --batch_ratio     2.5 \
  --Transformation  TPS \
  --FeatureExtraction ResNet \
  --SequenceModeling BiLSTM \
  --Prediction      Attn \
  --character       /home/kgh/ocr-tta/final_charlist_1770.txt \
  --num_iter        12000 \
  --valInterval     200 \
  --data_filtering_off \
  --save_path       ./saved_models_set_onnx/saved_models_9_all_generate_hub_br2.5

   python3 -W ignore train9_onnx.py \
  --exp_name        . \
  --train_data      /home/kgh/ocr-tta_copy/data_lmdb_08/train/custom \
  --valid_data      /home/kgh/ocr-tta_copy/data_lmdb_08/val \
  --select_data     custom \
  --batch_ratio     2.5 \
  --Transformation  TPS \
  --FeatureExtraction ResNet \
  --SequenceModeling BiLSTM \
  --Prediction      Attn \
  --character       /home/kgh/ocr-tta/final_charlist_1770.txt \
  --num_iter        12000 \
  --valInterval     200 \
  --data_filtering_off \
  --save_path       ./saved_models_set_onnx/saved_models_9_all_generate_hub_0816