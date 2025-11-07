# step 1 Create raw csv files 5000 : i have removed the command for this .. i do not generate the raw data again.

# step 2 generated synthetic multi category plots for training and create crops and those are saved in a cropping images directory  ( it is performed with run method)
# step 3 after cropping perform category separation ( it is performed with method separate_by_colors),
# step 4 after separating the category based on the colors,  rescale the lines  224*224 because unet will denoise the images as per this size ( method: rescale_lines)
# step 5 after rescaling the data lines split them to train and valid lines coordinates from rescaled_all_data json (method split_data)

# training -one time
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config_op.yaml --num_files 5000 --task run
#
## separating the categories based on color peaks histogram
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config_op.yaml --task separate_by_color
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config_op.yaml --task rescale_lines
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config_op.yaml --task split_data

# separating the categories based on cluster
python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config_op.yaml --task separate_by_cluster
python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config_op.yaml --task rescale_lines_cluster
python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config_op.yaml --task split_data_cluster


# step 1-4 for testing data
# testing - one time

#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/test_config_op.yaml --num_files 1000 --seed 0 --task run
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/test_config_op.yaml --seed 0 --task separate_by_color
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/test_config_op.yaml --seed 0 --task rescale_lines

python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/test_config_op.yaml --seed 0 --task separate_by_cluster
python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/test_config_op.yaml --seed 0 --task rescale_lines_cluster


