# imports
import sys
import os

import numpy as np
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))  # 2 levels up from this file
sys.path.insert(0, project_root)


from src.pc.config.config import get_args, load_config

import src.pc.data_gen.data_generator as dgen
from src.pc.data_gen.real_dist_info import extract_distributions_from_excel, extract_dist_plots_from_excel
from src.pc.plot_gen.multi_cat import MultiCatPCPGenerator
from src.pc.plot_gen.axes_crop import CroppingProcessor
from src.pc.plot_gen.category_separation import CategorySeparator
from src.pc.plot_gen.plot_utils import split_data, update_lines


class PlotsPipeline:
    def __init__(self):
        parser = get_args()
        self.args = parser.parse_args()
        self.cfg = load_config(self.args.cfg)
        self.paths = self.cfg['paths']
        self._set_seed(self.args.seed)

    def _set_seed(self, seed):
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def generate_data(self):
        dgen.generate_synthetic_datasets(self.paths['input_dir'],
                                         self.args.num_files,  seed=self.args.seed)

    def generate_data_from_excel_distribution(self):
        excel_path = self.paths['real_dist_file']
        axes_dist, cat_dist, rows_dist = extract_distributions_from_excel(excel_path)
        dgen.generate_synthetic_datasets_from_distributions(
            directory_path=self.paths['input_dir'],
            k=self.args.num_files,
            axes_distribution=axes_dist,
            categories_distribution=cat_dist,
            rows_distribution=rows_dist,
            seed=self.args.seed
        )


    def generate_plots(self):
        print("Generating SVG plots...")
        input_dir = self.paths['input_dir']
        plot_dir = self.paths['m_plots']
        svg_dir = self.paths['m_plots_svg']
        gt_plot_dir = self.paths['m_gt_plots']
        gt_plot_dir_cat = self.paths['m_gt_plots_cat']
        gt_plot_dir_cat_ntl = self.paths['m_gt_plots_cat_ntl']
        excel_path = self.paths['real_dist_file']

        bg_dist, grid_dist, ticks_dist = extract_dist_plots_from_excel(excel_path)
        mcat = MultiCatPCPGenerator()
        mcat.generate_batch(
            input_dir=input_dir,
            output_dir=plot_dir,
            svg_dir = svg_dir,
            num_files=self.args.num_files,
            background_distribution=bg_dist,
            grid_distribution=grid_dist,
            ticks_labels_distribution=ticks_dist,
            no_ticks_output_dir=gt_plot_dir,
            per_cat_dir=gt_plot_dir_cat,
            per_cat_ntl_dir=gt_plot_dir_cat_ntl

        )
        print("Generated SVG plots...")


    def crop_plots(self):
        print(" Cropping SVGs...")
        cropper = CroppingProcessor()
        cropper.create_crops(self.paths['m_plots'], self.paths['m_crops'])
        cropper.create_crops(self.paths['m_gt_plots'], self.paths['m_gt_crops'])
        cropper.create_crops(self.paths['m_gt_plots_cat'], self.paths['m_gt_plots_cat_crops'])
        cropper.create_crops(self.paths['m_gt_plots_cat_ntl'], self.paths['m_gt_plots_cat_ntl_crops'])

        print("cropped SVG plot and saved as pngs ...")

    def separate_by_color(self):
        print(f"🎨 Separating by color using {method} method...")
        sep = CategorySeparator()
        sep.process_batch(
            input_dir=self.paths['m_crops'],
            json_dir=self.paths['m_plots'],
            output_dir=self.paths['m_color_sep_plots']
        )

    def rescale_lines(self):
        if 'm_color_all_json' in self.paths:
            update_lines(
                json_file=self.paths['m_color_all_json'],
                output_file=self.paths['m_color_rescaled_all_json']
            )

    def split_data(self):
        if 'm_color_rescaled_all_json' in self.paths:
            split_data(
                input_file=self.paths['m_color_rescaled_all_json'],
                train_file=self.paths['m_color_train_json'],
                valid_file=self.paths['m_color_valid_json']
            )

    def split_data_wbg(self):
        # Optionally split into train/val
        if 'm_color_wbg_all_json' in self.paths:
            split_data(
                input_file=self.paths['m_color_wbg_all_json'],
                train_file=self.paths['m_color_wbg_train_json'],
                valid_file=self.paths['m_color_wbg_valid_json']
            )

    def run_dist(self):
        self.generate_data_from_excel_distribution()

    def run(self):
        self.generate_plots()
        self.crop_plots()


if __name__ == "__main__":
    pipeline = PlotsPipeline()
    task = pipeline.args.task

    # Dispatch the task
    if task == 'run_dist':
        pipeline.run_dist()
    elif task == 'run':
        pipeline.run()
    elif task == 'run_single':
        pipeline.run_single()
    elif hasattr(pipeline, task):
        method = getattr(pipeline, task)
        if callable(method):
            method()
        else:
            print(f"Attribute {task} is not callable.")
    else:
        print(f"Unknown task: {task}")



