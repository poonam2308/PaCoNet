# step 1 Create raw csv files 5000
# step 2 generated synthetic multi category plots for training and create crops and those are saved in a cropping images directory  ( it is performed with run method)
# step 3 after cropping perform category separation ( it is performed with method separate_by_colors),
# step 4 after separating the category based on the colors,  rescale the lines  224*224 because unet will denoise the images as per this size ( method: rescale_lines)
# step 5 after rescaling the data lines split them to train and valid lines coordinates from rescaled_all_data json (method split_data)

# training
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --num_files 5000 --task run_dis
python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --num_files 5000 --task run
python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task separate_by_color
python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task rescale_lines
python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task split_data

# step 1-4 for testing data
# testing
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/test_config.yaml --num_files 1000 --seed 0 --task run_dist
python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/test_config.yaml --num_files 1000 --seed 0 --task run
python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/test_config.yaml --seed 0 --task separate_by_color
python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/test_config.yaml --seed 0 --task rescale_lines


# unet training with noisy images
# unet training with non noisy images
#python unet_train.py --cfg configs/train_config.yaml
#
#python unet_inference.py --cfg configs/test_config.yaml

python src/pc/unet_train.py --cfg src/pc/config/train_config.yaml --batch_size 8 --num_epochs 80





######--------------------------------------------------
# category separation where the background is white  not required anymore
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task separate_by_color_wbg
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task split_data_wbg
