
# unet training with noisy images
# unet training with non noisy images
#python unet_train.py --cfg configs/train_config.yaml
#
#python unet_inference.py --cfg configs/test_config.yaml

python src/pc/unet_train.py --cfg src/pc/config/train_config.yaml --batch_size 8 --num_epochs 80

