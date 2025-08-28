set -uo pipefail
export CUDA_VISIBLE_DEVICES=3
# Python 경고 메시지(DeprecationWarning, UserWarning 등) 무시
export PYTHONWARNINGS="ignore"


python3 eval_model.py \
  --eval_data         /home/kgh/ocr-tta_copy/data_lmdb_08/od_res/custom \
  --saved_model       /home/kgh/ocr-tta/saved_models_set_onnx/saved_models_9_all_generate_hub_0816_2_br_25/best_accuracy.pth \
  --gt_file           /home/kgh/ocr-tta_copy/od_res.txt \
  --character         /home/kgh/ocr-tta/final_charlist_1770.txt \
  --batch_max_length  20 \
  --imgH              32 \
  --imgW              100 \
  --Transformation    TPS \
  --FeatureExtraction ResNet \
  --SequenceModeling  BiLSTM \
  --Prediction        Attn \
  --workers           4 \
  --batch_size        196 \
  --data_filtering_off 

python3 eval_model.py \
  --eval_data         /home/kgh/ocr-tta_copy/data_lmdb_08/od_res/custom \
  --saved_model       /home/kgh/ocr-tta/saved_models_set_onnx/saved_models_9_all_generate_hub_0816_lr_br_25_ft/best_accuracy.pth \
  --gt_file           /home/kgh/ocr-tta_copy/od_res.txt \
  --character         /home/kgh/ocr-tta/final_charlist_1770.txt \
  --batch_max_length  20 \
  --imgH              32 \
  --imgW              100 \
  --Transformation    TPS \
  --FeatureExtraction ResNet \
  --SequenceModeling  BiLSTM \
  --Prediction        Attn \
  --workers           4 \
  --batch_size        196 \
  --data_filtering_off

  python3 eval_model.py \
  --eval_data         /home/kgh/ocr-tta_copy/data_lmdb_08/od_res/custom \
  --saved_model       /home/kgh/ocr-tta/saved_models_set_onnx/saved_models_9_all_generate_hub_0816_lr_br_25_ft_2/best_accuracy.pth \
  --gt_file           /home/kgh/ocr-tta_copy/od_res.txt \
  --character         /home/kgh/ocr-tta/final_charlist_1770.txt \
  --batch_max_length  20 \
  --imgH              32 \
  --imgW              100 \
  --Transformation    TPS \
  --FeatureExtraction ResNet \
  --SequenceModeling  BiLSTM \
  --Prediction        Attn \
  --workers           4 \
  --batch_size        196 \
  --data_filtering_off

python3 eval_model.py \
  --eval_data         /home/kgh/ocr-tta/data_lmdb_03/val/all_change2/custom \
  --saved_model       /home/kgh/ocr-tta/saved_models_set_onnx/saved_models_9_all_generate_hub_0816_2_br_25/best_accuracy.pth \
  --gt_file           /home/kgh/ocr-tta/data_lmdb_03/val/all_change2_gt.txt \
  --character         /home/kgh/ocr-tta/final_charlist_1770.txt \
  --batch_max_length  20 \
  --imgH              32 \
  --imgW              100 \
  --Transformation    TPS \
  --FeatureExtraction ResNet \
  --SequenceModeling  BiLSTM \
  --Prediction        Attn \
  --workers           4 \
  --batch_size        196 \
  --data_filtering_off 

  python3 eval_model.py \
  --eval_data         /home/kgh/ocr-tta/data_lmdb_03/val/all_change2/custom \
  --saved_model       /home/kgh/ocr-tta/saved_models_set_onnx/saved_models_9_all_generate_hub_0816_lr_br_25_ft/best_accuracy.pth \
  --gt_file           /home/kgh/ocr-tta/data_lmdb_03/val/all_change2_gt.txt \
  --character         /home/kgh/ocr-tta/final_charlist_1770.txt \
  --batch_max_length  20 \
  --imgH              32 \
  --imgW              100 \
  --Transformation    TPS \
  --FeatureExtraction ResNet \
  --SequenceModeling  BiLSTM \
  --Prediction        Attn \
  --workers           4 \
  --batch_size        196 \
  --data_filtering_off

  python3 eval_model.py \
  --eval_data         /home/kgh/ocr-tta/data_lmdb_03/val/all_change2/custom \
  --saved_model       /home/kgh/ocr-tta/saved_models_set_onnx/saved_models_9_all_generate_hub_0816_lr_br_25_ft_2/best_accuracy.pth \
  --gt_file           /home/kgh/ocr-tta/data_lmdb_03/val/all_change2_gt.txt \
  --character         /home/kgh/ocr-tta/final_charlist_1770.txt \
  --batch_max_length  20 \
  --imgH              32 \
  --imgW              100 \
  --Transformation    TPS \
  --FeatureExtraction ResNet \
  --SequenceModeling  BiLSTM \
  --Prediction        Attn \
  --workers           4 \
  --batch_size        196 \
  --data_filtering_off