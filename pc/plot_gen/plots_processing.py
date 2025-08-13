# imports
import sys
import os

import numpy as np
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))  # 2 levels up from this file
sys.path.insert(0, project_root)

from pc.config.config import get_args, load_config
from pc.plot_gen.single_cat import SingleCatPCPGenerator

import pc.data_gen.data_generator as dgen
from pc.plot_gen.multi_cat import MultiCatPCPGenerator
from pc.plot_gen.axes_crop import CroppingProcessor
from pc.plot_gen.line_data import LineCoordinateExtractor
from pc.plot_gen.cat_sep import CategorySeparator
from pc.plot_gen.plot_utils import split_json_data
from pc.data_gen.real_dist_info import extract_distributions_from_excel, extract_dist_plots_from_excel


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
        gt_plot_dir = self.path['m_gt_plots']
        excel_path = self.paths['real_dist_file']

        bg_dist, grid_dist, ticks_dist = extract_dist_plots_from_excel(excel_path)
        mcat = MultiCatPCPGenerator()
        mcat.generate_batch(
            input_dir=input_dir,
            output_dir=plot_dir,
            num_files=self.args.num_files,
            background_distribution=bg_dist,
            grid_distribution=grid_dist,
            ticks_labels_distribution=ticks_dist,
            no_ticks_output_dir= gt_plot_dir
        )
        print("Generated SVG plots...")


    def crop_plots(self):
        print(" Cropping SVGs...")
        cropper = CroppingProcessor()
        cropper.create_crops(self.paths['m_plots'], self.paths['m_crops'])
        cropper.create_crops(self.paths['m_gt_plots'], self.paths['m_gt_crops'])

        print("cropped SVG plot and saved as pngs ...")

    def extract_lines(self):
        print("Extracting line coordinates data ...")
        extractor = LineCoordinateExtractor(main_dir=self.paths['m_plots'], output_file=self.paths['m_all_json'])
        extractor.extract_all()

    def separate_by_color(self, method='dbscan'):
        print(f"🎨 Separating by color using {method} method...")
        sep = CategorySeparator(input_dir=self.paths['m_crops'], line_coords_json=self.paths['m_all_json'])

        if method == 'hist':
            sep.separate_by_hist_peaks(
                output_dir=self.paths['m_color_sep_plots'],
                output_json=self.paths['m_color_all_json'],
                color_json=self.paths['m_color_line_color']
            )
        elif method == 'dbscan':
            sep.separate_by_dbscan(
                output_dir=self.paths['m_cluster_sep_plots'],
                output_json=self.paths['m_cluster_all_json'],
                color_json=self.paths['m_cluster_line_color']
            )

        # Optionally split into train/val
        if 'm_cluster_train_json' in self.paths:
            split_json_data(
                input_json=self.paths['m_cluster_all_json'],
                train_json=self.paths['m_cluster_train_json'],
                valid_json=self.paths['m_cluster_valid_json']
            )

    def generate_plots_single(self):
        print("Generating SVG plots...")
        input_dir = self.paths['input_dir']
        plot_dir = self.paths['s_plots']
        scat = SingleCatPCPGenerator(show_labels=False)
        scat.generate_batch(input_dir, plot_dir, self.args.num_files)
        print("Generated SVG plots...")

    def crop_plots_single(self):
        print(" Cropping SVGs...")
        cropper = CroppingProcessor()
        cropper.create_crops(self.paths['s_plots'], self.paths['s_crops'])
        print("cropped SVG plot and saved as pngs ...")

    def extract_lines_single(self):
        print("Extracting line coordinates data ...")
        extractor = LineCoordinateExtractor(main_dir=self.paths['s_plots'], output_file=self.paths['s_all_json'])
        extractor.extract_all()

    def run_single(self):
        self.generate_plots_single()
        self.crop_plots_single()
        self.extract_lines_single()

    def run(self):
        self.generate_data_from_excel_distribution()
        self.generate_plots()
        self.crop_plots()
        # self.extract_lines()
        # self.separate_by_color(method='dbscan')  # or method='hist'


if __name__ == "__main__":
    pipeline = PlotsPipeline()
    task = pipeline.args.task

    # Dispatch the task
    if task == 'run':
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



