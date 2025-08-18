#python plots_processing.py --cfg configs/train_config.yaml

#python plots_processing.py --cfg configs/train_config.yaml --task generate_data

python pc/plot_gen/plots_processing.py --cfg pc/config/test_config.yaml --num_files 1000 --task run_dist
python pc/plot_gen/plots_processing.py --cfg pc/config/test_config.yaml --num_files 1000 --task run
