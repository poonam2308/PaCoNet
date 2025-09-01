# Create raw csv files and synthetic multi category plots for training and testing set ( 5000 and 1000)
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --num_files 5000 --task run_dis
# python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --num_files 5000 --task run

#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/test_config.yaml --num_files 1000 --task run_dist
#python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/test_config.yaml --num_files 1000 --task run


# create crops and those are saved in a cropping images directory  ( it is performed with run method)

# after cropping perform category separation
python src/pc/plot_gen/plots_processing.py --cfg src/pc/config/train_config.yaml --task separate_by_color