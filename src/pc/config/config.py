from datetime import datetime
import argparse

import yaml

now = datetime.now()
timestamp = now.strftime("%d%m-%y_%H%M-%S")
model_log_dir= f"outputs/chkpt/{timestamp}/"
plot_log_dir= f"outputs/plots/{timestamp}/"
save_dir = f"outputs/unet_vis/{timestamp}/"

def get_args():
    parse = argparse.ArgumentParser(description='PaCoNet')
    parse.add_argument('--cfg', type=str, required=True, help="Path to YAML config file")

    # csv count
    parse.add_argument('--num_files', type=int, default=10)
    parse.add_argument('--seed', type=int, default=42)

    #plots generated tasks
    parse.add_argument('--task', type=str, default='run', help="Which method to run",
                       choices=['run_dist', 'run', 'run_single', 'generate_data', 'generate_plots',
                                'crop_plots', 'separate_by_color','rescale_lines', 'split_data',
                                'separate_by_cluster','rescale_lines_cluster', 'split_data_cluster',
                                'separate_by_elbo', 'rescale_lines_elbo', 'split_data_elbo',
                                'separate_by_elbo_fres', 'rescale_lines_elbo_fres', 'split_data_elbo_fres',
                                'separate_by_lab_cluster', 'rescale_lines_lab_cluster', 'split_data_lab_cluster',
                                'separate_by_peakcluster', 'rescale_lines_peakcluster', 'split_data_peakcluster',
                                'separate_by_hdbscan', 'rescale_lines_hdbscan', 'split_data_hdbscan',
                                'cat_eval_color', 'cat_eval_cluster',
                                'color_space_evaluation', 'dino_features_evaluation',
                                'separate_by_color_wbg', 'split_data_wbg',
                                'gt_rename', 'split_data_gt',
                                'crop_whitebg_lines',
                                'color_unet', 'cluster_unet'])

    # Network Training Parameters
    parse.add_argument('--reg_model', type =str, default='lenet6', choices=['lenet4', 'lenet6', 'lenet', 'resnet18'])
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--learning_rate', type=float, default=0.001)
    parse.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'])

    parse.add_argument('--num_epochs', type=int, default=1)
    parse.add_argument('--num_workers', type=int, default=4)

    parse.add_argument('--alpha', type=float, default=0.3)
    parse.add_argument('--new_width', type=int, default=224)
    parse.add_argument('--new_height', type=int, default=224)
    parse.add_argument('--mean', type=float, default=0.5)
    parse.add_argument('--std', type=float, default=0.5)

    parse.add_argument('--out_features', type=int, default=1)
    parse.add_argument('--out_channels', type=int, default=3)
    parse.add_argument('--in_channels', type=int, default=3)
    parse.add_argument('--channel_mode', type=str, default="RGB")
    parse.add_argument('--swin_out_features', type=int, default=3)

    parse.add_argument('--max_lines', type=int, default=100)

    # wandb
    parse.add_argument('--project', type=str, default='PaCoNet')
    parse.add_argument('--sweep', type=str, default='false', choices=['false', 'true'])
    parse.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline'])

    # dht_model
    parse.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'resnet101'])
    parse.add_argument('--numangle', type=int, default=100)
    parse.add_argument('--numrho', type=int, default=100)
    parse.add_argument('--threshold', type=float, default=0.01)
    parse.add_argument('--gamma', type=float, default=0.1)
    parse.add_argument('--steps', default=[], nargs='*', type=int)
    parse.add_argument('--edge-align', action='store_true')

    return parse

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)
