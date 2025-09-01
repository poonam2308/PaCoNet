import os
import json
import numpy as np
import pandas as pd
import altair as alt
import random
from src.pc.plot_gen.plot_utils import normalize_column, create_ticks_labels, calculate_pixel_positions, safe_join


class SingleCatPCPGenerator:
    def __init__(self, width=600, height=300, seed=42, show_labels=True):
        self.width = width
        self.height = height
        self.seed = seed
        self.show_labels = show_labels
        np.random.seed(self.seed)
        random.seed(self.seed)

    def generate_plot(self, df, filename=None):
        """Generate a single parallel coordinates plot and optionally save it."""
        column_names = sorted(list(df.columns)[:-1])
        color_column = df.columns[-1]

        normalized_columns = [normalize_column(df, col) for col in column_names]

        unique_categories = sorted(df[color_column].unique())
        selected_category = np.random.choice(unique_categories)  # Randomly select one category
        df2 = df[df[color_column] == selected_category]

        base = alt.Chart(df2).transform_window(index="count()").transform_fold(
            normalized_columns
        ).transform_calculate(mid="(datum.value + datum.value) / 2").properties(
            width=self.width, height=self.height
        )

        lines = base.mark_line(opacity=0.3).encode(
            x='key:N',
            y=alt.Y('value:Q', axis=None),
            detail="index:N",
            tooltip=column_names
        )

        # rules = base.mark_rule(color="#ccc", tooltip=None).encode(x="key:N")
        rules = base.mark_rule(color="#ccc", tooltip=None).encode(
            x="key:N",
            detail="count():Q"
        )

        tick_dfs = [create_ticks_labels(df, norm_col, orig_col)
                    for norm_col, orig_col in zip(normalized_columns, column_names)]
        ticks_labels_df = pd.concat(tick_dfs)

        if self.show_labels:
            ticks = alt.Chart(ticks_labels_df).mark_tick(size=8, color="#ccc", orient="horizontal").encode(
                x='variable:N', y='value:Q'
            )

            labels = alt.Chart(ticks_labels_df).mark_text(
                align='center', baseline='middle', dx=-10
            ).encode(
                x='variable:N', y='value:Q', text='label:N'
            )
        else:
            ticks = alt.Chart(ticks_labels_df).mark_tick(size=8, color="transparent", orient="horizontal").encode(
                x='variable:N', y='value:Q'
            )

            labels = alt.Chart(ticks_labels_df).mark_text(
                align='center', baseline='middle', dx=-10, color="transparent"
            ).encode(
                x='variable:N', y='value:Q', text='label:N'
            )

        axis_config = {
            "domain": False,
            "labelAngle": 0,
            "title": None,
            "tickColor": "#ccc",
            "labelColor": "#000"
        }

        if not self.show_labels:
            axis_config["tickColor"] = "transparent"
            axis_config["labelColor"] = "transparent"

        chart = alt.layer(lines, rules, ticks, labels).configure_axisX(**axis_config).configure_view(stroke=None)

        if filename:
            chart.save(filename)

            # Calculate pixel positions for distribution analysis
        pixel_positions = calculate_pixel_positions(df2, normalized_columns, self.height)

        # Calculate mean and standard deviation for each axis
        mean_std = {}
        for col in pixel_positions:
            positions = pixel_positions[col]
            mean = np.mean(positions)
            std = np.std(positions)
            mean_std[col] = {
                'mean': mean,
                'std': std,
                'lower_deviation': mean - std,
                'upper_deviation': mean + std
            }
        return chart, selected_category, mean_std

    def generate_batch(self, input_dir, output_dir, num_files, annotation_file="dist_annotations.json"):
        """Generate plots and annotations for a directory of CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        annotation_file = safe_join(output_dir, annotation_file)

        annotations = []

        for i in range(1, num_files + 1):
            csv_path = os.path.join(input_dir, f'synthetic_data_{i}.csv')
            if not os.path.exists(csv_path):
                print(f"File not found: {csv_path}")
                continue

            df = pd.read_csv(csv_path)

            if df.empty:
                print(f"No valid categories in {csv_path}")
                continue

            output_file = os.path.join(output_dir, f'image_{i}.svg')
            chart, category, mean_std = self.generate_plot(df, filename=output_file)

            # Collect annotations
            annotation = {
                'image_name': f'image_{i}.svg',
                'category': category,
                'columns': []
            }
            for col, stats in mean_std.items():
                annotation['columns'].append({
                    'column': col,
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'lower_deviation': stats['lower_deviation'],
                    'upper_deviation': stats['upper_deviation']
                })
            annotations.append(annotation)

        with open(annotation_file, 'w') as f:
            json.dump(annotations, f, indent=4)

        print(f"Saved {len(annotations)} plots to {output_dir}")
        print(f"Annotations written to {annotation_file}")
