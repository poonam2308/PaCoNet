# training unet

#python src/pc/unet_train.py --cfg src/pc/config/train_config.yaml --batch_size 8 --num_epochs 80

#python src/pc/unet_train.py --cfg src/pc/config/train_config.yaml --batch_size 8 --num_epochs 80 --task cluster_unet

# generating denoised images using trained unet

python src/pc/unet_inference.py --cfg src/pc/config/train_config.yaml --batch_size 8

#for test dataset - generating the denoised images from the test dataset for color based unet
#python src/pc/unet_inference.py --cfg src/pc/config/test_config.yaml --batch_size 8


# for the opaque pc images

#python src/pc/unet_train.py --cfg src/pc/config/train_config_op.yaml --batch_size 8 --num_epochs 80 --task color_unet

#python src/pc/unet_inference.py --cfg src/pc/config/train_config_op.yaml --batch_size 8

#python src/pc/unet_train.py --cfg src/pc/config/train_config_op.yaml --batch_size 8 --num_epochs 80 --task cluster_unet

#python src/pc/unet_inference.py --cfg src/pc/config/test_config_op.yaml --batch_size 8