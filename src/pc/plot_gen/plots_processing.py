# imports
import sys
import os
import numpy as np
import torch



project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))  # 2 levels up from this file
sys.path.insert(0, project_root)
from pc.plot_gen.category_separation_downsample import DownsampledHistogramBatchSeparator
from src.pc.plot_gen.process_gt_images import run_rename, whiten_backgrounds_in_dir, group_crops_to_new_json
from src.pc.plot_gen.color_space_evaluator import RGBKMeansEvaluator, LabKMeansEvaluator, HSVFullKMeansEvaluator, \
    HSVHueKMeansEvaluator, DinoKMeansEvaluator
from src.pc.plot_gen.hdbscan_category_separation import HDBSCANCategorySeparator
from src.pc.plot_gen.peak_clustering_category_separator import PeakClusteringCategorySeparator
from src.pc.plot_gen.lab_clustering_category_separation import LabClusteringCategorySeparator
from src.pc.plot_gen.elbo_category_separator import ELBOCategorySeparator
from src.pc.plot_gen.elbo_fullres_category_separator import ELBOFullResCategorySeparator
from src.pc.config.config import get_args, load_config

import src.pc.data_gen.data_generator as dgen
from src.pc.data_gen.real_dist_info import extract_distributions_from_excel, extract_dist_plots_from_excel
from src.pc.plot_gen.multi_cat import MultiCatPCPGenerator
from src.pc.plot_gen.axes_crop import CroppingProcessor
from src.pc.plot_gen.category_separation import CategorySeparator
from src.pc.plot_gen.clustering_category_separation import ClusteringCategorySeparator
from src.pc.plot_gen.plot_utils import split_data, update_lines, resize_images_to_224
from src.pc.plot_gen.cat_sep_evaluation import evaluate_catsep_vs_gt


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
        sep.save_cluster_vs_gt_only(
            input_dir=self.paths['m_color_sep_plots'],
            json_dir=self.paths['m_plots'],
        )

    def separate_peak_downsample(self):
        print(f"🎨 Separating by color using {method} method...")
        sep = DownsampledHistogramBatchSeparator()
        sep.process_batch(
            input_dir=self.paths['m_crops'],
            json_dir=self.paths['m_plots'],
            output_dir=self.paths['m_color_down_sep_plots'],
            resize_factor_large=0.30
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

    def separate_by_cluster(self):
        print(f"🎨 Separating by cluster using {method} method...")
        sep = ClusteringCategorySeparator()
        sep.process_batch(
            input_dir=self.paths['m_crops'],
            json_dir=self.paths['m_plots'],
            output_dir=self.paths['m_cluster_sep_plots']
        )

    def rescale_lines_cluster(self):
        if 'm_cluster_all_json' in self.paths:
            update_lines(
                json_file=self.paths['m_cluster_all_json'],
                output_file=self.paths['m_cluster_rescaled_all_json']
            )

    def split_data_cluster(self):
        if 'm_cluster_rescaled_all_json' in self.paths:
            split_data(
                input_file=self.paths['m_cluster_rescaled_all_json'],
                train_file=self.paths['m_cluster_train_json'],
                valid_file=self.paths['m_cluster_valid_json']
            )
    #----Elbo ----#
    def separate_by_elbo(self):
        print(f"🎨 Separating by elbo using {method} method...")
        sep = ELBOCategorySeparator()
        sep.process_batch(
            input_dir=self.paths['m_crops'],
            json_dir=self.paths['m_plots'],
            output_dir=self.paths['m_elbo_sep_plots']
        )

    def rescale_lines_elbo(self):
        if 'm_elbo_all_json' in self.paths:
            update_lines(
                json_file=self.paths['m_elbo_all_json'],
                output_file=self.paths['m_elbo_rescaled_all_json']
            )

    def split_data_elbo(self):
        if 'm_elbo_rescaled_all_json' in self.paths:
            split_data(
                input_file=self.paths['m_elbo_rescaled_all_json'],
                train_file=self.paths['m_elbo_train_json'],
                valid_file=self.paths['m_elbo_valid_json']
            )

    # ----Elbo full resolution  ----#
    def separate_by_elbo_fres(self):
        print(f"🎨 Separating by elbo full resolution using {method} method...")
        sep = ELBOFullResCategorySeparator()
        sep.process_batch(
            input_dir=self.paths['m_crops'],
            json_dir=self.paths['m_plots'],
            output_dir=self.paths['m_elbofres_sep_plots']
        )

    def rescale_lines_elbo_fres(self):
        if 'm_elbofres_all_json' in self.paths:
            update_lines(
                json_file=self.paths['m_elbofres_all_json'],
                output_file=self.paths['m_elbofres_rescaled_all_json']
            )

    def split_data_elbo_fres(self):
        if 'm_elbofres_rescaled_all_json' in self.paths:
            split_data(
                input_file=self.paths['m_elbofres_rescaled_all_json'],
                train_file=self.paths['m_elbofres_train_json'],
                valid_file=self.paths['m_elbofres_valid_json']
            )


    def split_data_wbg(self):
        # Optionally split into train/val
        if 'm_color_wbg_all_json' in self.paths:
            split_data(
                input_file=self.paths['m_color_wbg_all_json'],
                train_file=self.paths['m_color_wbg_train_json'],
                valid_file=self.paths['m_color_wbg_valid_json']
            )

    #%------Lab color space for clustering-----

    def separate_by_lab_cluster(self):
        print(f"🎨 Separating by cluster using {method} method...")
        sep = LabClusteringCategorySeparator()
        sep.process_batch(
            input_dir=self.paths['m_crops'],
            json_dir=self.paths['m_plots'],
            output_dir=self.paths['m_lab_cluster_sep_plots']
        )

    def rescale_lines_lab_cluster(self):
        if 'm_lab_cluster_all_json' in self.paths:
            update_lines(
                json_file=self.paths['m_lab_cluster_all_json'],
                output_file=self.paths['m_lab_cluster_rescaled_all_json']
            )

    def split_data_lab_cluster(self):
        if 'm_lab_cluster_rescaled_all_json' in self.paths:
            split_data(
                input_file=self.paths['m_lab_cluster_rescaled_all_json'],
                train_file=self.paths['m_lab_cluster_train_json'],
                valid_file=self.paths['m_lab_cluster_valid_json']
            )

    #%-------------Hue peak filtering and then clustering -----------------
    def separate_by_peakcluster(self):
        print(f"🎨 Separating by cluster using {method} method...")
        sep = PeakClusteringCategorySeparator()
        sep.process_batch_with_peaks(
            input_dir=self.paths['m_crops'],
            json_dir=self.paths['m_plots'],
            output_dir=self.paths['m_peak_sep_plots']
        )

    def rescale_lines_peakcluster(self):
        if 'm_peak_all_json' in self.paths:
            update_lines(
                json_file=self.paths['m_peak_all_json'],
                output_file=self.paths['m_peak_rescaled_all_json']
            )

    def split_data_peakcluster(self):
        if 'm_peak_rescaled_all_json' in self.paths:
            split_data(
                input_file=self.paths['m_peak_rescaled_all_json'],
                train_file=self.paths['m_peak_train_json'],
                valid_file=self.paths['m_peak_valid_json']
            )

    #----------------- hdbscan clustering with downsampled images--------
    def separate_by_hdbscan(self):
        print(f"🎨 Separating by cluster using {method} method...")
        sep = HDBSCANCategorySeparator()
        sep.process_batch(
            input_dir=self.paths['m_crops'],
            json_dir=self.paths['m_plots'],
            output_dir=self.paths['m_hdbscan_sep_plots']
        )

    def rescale_lines_hdbscan(self):
        if 'm_hdbscan_all_json' in self.paths:
            update_lines(
                json_file=self.paths['m_hdbscan_all_json'],
                output_file=self.paths['m_hdbscan_rescaled_all_json']
            )

    def split_data_hdbscan(self):
        if 'm_hdbscan_rescaled_all_json' in self.paths:
            split_data(
                input_file=self.paths['m_hdbscan_rescaled_all_json'],
                train_file=self.paths['m_hdbscan_train_json'],
                valid_file=self.paths['m_hdbscan_valid_json']
            )

    ##--------------- Gt rename and all data merge and train and valid split --
    def gt_rename(self):
        run_rename(
            self.paths['m_gt_plots_cat_ntl_crops'],
            self.paths['m_plots'],
            self.paths['m_gt_cat_ntl_rename'],
            self.paths['m_gt_cat_all_data']

        )

    def split_data_gt(self):
        if 'm_gt_cat_all_data' in self.paths:
            split_data(
                input_file=self.paths['m_gt_cat_all_data'],
                train_file=self.paths['m_gt_train_json'],
                valid_file=self.paths['m_gt_valid_json']
            )


    def run_dist(self):
        self.generate_data_from_excel_distribution()

    def run(self):
        self.generate_plots()
        self.crop_plots()

    def cat_eval_color(self):
        evaluate_catsep_vs_gt(
            pred_dir=self.paths['m_color_sep_plots'],
            gt_dir=self.paths['m_gt_plots_cat_ntl_crops'],
            white_thresh=750,
            per_crop_csv=self.paths['m_color_cat_eval_crop_csv'],
            per_base_csv=self.paths['m_color_cat_eval_base_csv'],
            summary_json=self.paths['m_color_cat_eval_json'],
            verbose=True
        )
    def cat_eval_cluster(self):
        evaluate_catsep_vs_gt(
            pred_dir=self.paths['m_cluster_sep_plots'],
            gt_dir=self.paths['m_gt_plots_cat_ntl_crops'],
            white_thresh=750,
            per_crop_csv=self.paths['m_cluster_cat_eval_crop_csv'],
            per_base_csv=self.paths['m_cluster_cat_eval_base_csv'],
            summary_json=self.paths['m_cluster_cat_eval_json'],
            verbose=True
        )

    def color_space_evaluation(self):
        rgb_eval = RGBKMeansEvaluator(sample_size=10000)
        hsv_h_eval = HSVHueKMeansEvaluator(sample_size=10000)
        lab_eval = LabKMeansEvaluator(sample_size=10000)
        hsv_full_eval = HSVFullKMeansEvaluator(sample_size=10000)


        rgb_eval.evaluate_batch(input_dir=self.paths['m_crops'],
            json_dir=self.paths['m_plots'],
            output_dir=self.paths['m_color_space'])

        hsv_h_eval.evaluate_batch(input_dir=self.paths['m_crops'],
            json_dir=self.paths['m_plots'],
            output_dir=self.paths['m_color_space'])

        lab_eval.evaluate_batch(input_dir=self.paths['m_crops'],
            json_dir=self.paths['m_plots'],
            output_dir=self.paths['m_color_space'])

        hsv_full_eval.evaluate_batch(input_dir=self.paths['m_crops'],
            json_dir=self.paths['m_plots'],
            output_dir=self.paths['m_color_space'])


    def dino_features_evaluation(self):
        dino_eval = DinoKMeansEvaluator(
            model_name="vit_small_patch16_224.dino",
            sample_size=10000,  # max number of tokens to use for KMeans
        )
        dino_eval.evaluate_batch(input_dir=self.paths['m_crops'],
            json_dir=self.paths['m_plots'],
            output_dir=self.paths['m_color_space'])



    def resize_noisy_images(self):
        # resize_images_to_224(self.paths['m_color_sep_plots'], self.paths['m_color_sep_plots_224'])
        resize_images_to_224(self.paths['m_cluster_sep_plots'], self.paths['m_cluster_sep_plots_224'])


    def white_bg(self):
        # whiten_backgrounds_in_dir(self.paths['pcw_test'])
        # whiten_backgrounds_in_dir(self.paths['pcw_test_cls'])
        # whiten_backgrounds_in_dir(self.paths['pcw_ntest'])
        # whiten_backgrounds_in_dir(self.paths['pcw_ntest_cls'])
        whiten_backgrounds_in_dir(self.paths['m_cluster_sep_plots_224'])


    def crop_whitebg_lines(self):
        # whiten_backgrounds_in_dir(self.paths['m_crops_white'])
        # group_crops_to_new_json(self.paths['m_gt_train_json'], self.paths['m_crops_white_train_data'])
        group_crops_to_new_json(self.paths['m_gt_valid_json'], self.paths['m_crops_white_valid_data'])


    def test_crops(self):
        resize_images_to_224(self.paths['m_crops'], self.paths['m_crops_224'])
        whiten_backgrounds_in_dir(self.paths['m_crops_224'])
        group_crops_to_new_json(self.paths['m_crops_all_data'], self.paths['m_crops_combined_data'])
        if 'm_crops_combined_data' in self.paths:
            update_lines(
                json_file=self.paths['m_crops_combined_data'],
                output_file=self.paths['m_crops_rescaled_data']
            )

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



