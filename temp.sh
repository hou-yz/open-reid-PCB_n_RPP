# Train Model using gt
CUDA_VISIBLE_DEVICES=3 python IDE.py --epochs 60  --train -d ai_gt --logs-dir logs/ide_new/256/ai_city/train/1_fps/basis --height 384 -s 1 --features 256 --output_feature fc --mygt_fps 1 --re 0.5

# Save the features
CUDA_VISIBLE_DEVICES=2 python save_cnn_feature.py  -a ide --resume logs/ide_new/256/ai_city/train/1_fps/basis/model_best.pth.tar --features 256 --output_feature fc --l0_name ide_basis_train_1fps -s 1 --height 384 -d detections --det_time val