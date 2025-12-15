# step 1 Create raw csv files 5000
# step 2 generated synthetic multi category plots for training and create crops and those are saved in a cropping images directory  ( it is performed with run method)
# step 3 after cropping perform category separation ( it is performed with method separate_by_colors),
# step 4 after separating the category based on the colors,  rescale the lines  224*224 because unet will denoise the images as per this size ( method: rescale_lines)
# step 5 after rescaling the data lines split them to train and valid lines coordinates from rescaled_all_data json (method split_data)

# training -one time
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --num_files 5000 --task run_dist
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --num_files 5000 --task run

#separating the categories based on color
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task separate_by_color
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task rescale_lines
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task split_data

#separating the categories based on cluster dbscan reduced size

#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task separate_by_cluster
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task rescale_lines_cluster
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task split_data_cluster


#separating the categories based on cluster elbo bayesian gmm  reduced size

#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task separate_by_elbo
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task rescale_lines_elbo
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task split_data_elbo


#separating the categories based on cluster elbo bayesian gmm  full size

#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task separate_by_elbo_fres
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task rescale_lines_elbo_fres
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task split_data_elbo_fres

#separating the categories based on cluster using lab color space

#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task separate_by_lab_cluster
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task rescale_lines_lab_cluster
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task split_data_lab_cluster


# separating the categories based on peak filtering and then dbscan clustering

#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task separate_by_peakcluster
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task rescale_lines_peakcluster
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task split_data_peakcluster

#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task separate_by_hdbscan
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task rescale_lines_hdbscan
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task split_data_hdbscan

# gt name categories with white background and the lines data changed to train and valid split
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task gt_rename
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task split_data_gt

# crops with white background and the cat lines are grouped under the crops
# python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task crop_whitebg_lines
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/test_config.yaml --task white_bg

#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task white_bg

# color_space evalaution RGb, Lab, HSV
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/test_config.yaml --task color_space_evaluation

#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/test_config.yaml --task dino_features_evaluation

# category evaluation
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task cat_eval_color
#
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task cat_eval_cluster

# resize the images to 224 224 which are not denoised using unet
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task resize_noisy_images


#-----------------------------------------------------------------------------------------------------------
# step 1-4 for testing data
# testing - one time
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/test_config.yaml --num_files 1000 --seed 0 --task run_dist
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/test_config.yaml --num_files 1000 --seed 0 --task run

# color histogram clustering
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/test_config.yaml --seed 0 --task separate_by_color
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/test_config.yaml --seed 0 --task rescale_lines
#
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/test_config.yaml --seed 0 --task separate_by_cluster
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/test_config.yaml --seed 0 --task rescale_lines_cluster

# lab color space clustering
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/test_config.yaml --seed 0 --task separate_by_lab_cluster
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/test_config.yaml --seed 0 --task rescale_lines_lab_cluster

# resize the images to 224 224 which are not denoised using unet
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/test_config.yaml --seed 0 --task resize_noisy_images

######--------------------------------------------------
# category separation where the background is white  not required anymore
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task separate_by_color_wbg
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task split_data_wbg

# crops  resize
#python src/pc/plot_gen/plots_processing.py --cfg ./src/pc/config/test_config.yaml --seed 0  --task test_crops
