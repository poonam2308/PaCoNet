import os
import json
import numpy as np
import pandas as pd
import altair as alt
from pc.plot_gen.plot_utils import normalize_column, hsv_to_rgb, create_ticks_labels, calculate_pixel_positions, \
    safe_join
import random


class MultiCatPCPGenerator:
    def __init__(self, width=600, height=300, seed =42, show_labels=True):
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
        category_colors = {
            category: hsv_to_rgb(round(h / len(unique_categories), 2), 1, 1)
            for h, category in enumerate(unique_categories)
        }

        base = alt.Chart(df).transform_window(index="count()").transform_fold(
            normalized_columns
        ).transform_calculate(mid="(datum.value + datum.value) / 2").properties(
            width=self.width, height=self.height
        )

        lines = base.mark_line(opacity=0.3).encode(
            x='key:N',
            y=alt.Y('value:Q', axis=None),
            color=alt.Color(f"{color_column}:N", scale=alt.Scale(
                domain=list(category_colors.keys()),
                range=['rgb({},{},{})'.format(*color) for color in category_colors.values()]
            )),
            detail="index:N",
            tooltip=column_names
        )

        rules = base.mark_rule(color="#ccc", tooltip=None).encode(x="key:N")

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

        # chart = alt.layer(lines, rules, ticks, labels).configure_axisX(
        #     domain=False, labelAngle=0, tickColor="#ccc", title=None
        # ).configure_view(stroke=None)

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

        chart = alt.layer(lines, rules, ticks, labels).configure_axisX(
            **axis_config).configure_view(stroke=None)

        if filename:
            chart.save(filename)

        return chart, normalized_columns

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
            color_column = df.columns[-1]
            unique_categories = df[color_column].unique()
            selected_categories = unique_categories

            df = df[df[color_column].isin(selected_categories)]
            if df.empty:
                print(f"No valid categories in {csv_path}")
                continue

            output_file = os.path.join(output_dir, f'image_{i}.svg')
            _, normalized_columns = self.generate_plot(df, filename=output_file)

            image_annotation = {
                'image_name': os.path.basename(output_file),
                'categories': []
            }

            for category in unique_categories:
                df_cat = df[df[color_column] == category]
                pixel_positions = calculate_pixel_positions(df_cat, normalized_columns, self.height)

                if category not in selected_categories:
                    continue

                h_index = np.where(selected_categories == category)[0][0]
                color_hsv = {'h': round(h_index / len(unique_categories), 2), 's': 1, 'v': 1}

                category_stats = {
                    'category': category,
                    'color_hsv': color_hsv,
                    'columns': []
                }

                for col in pixel_positions:
                    values = pixel_positions[col]
                    mean = np.mean(values)
                    std = np.std(values)
                    category_stats['columns'].append({
                        'column': col.strip(),
                        'mean': mean,
                        'std': std,
                        'lower_deviation': mean - std,
                        'upper_deviation': mean + std
                    })

                image_annotation['categories'].append(category_stats)

            annotations.append(image_annotation)

        with open(annotation_file, 'w') as f:
            json.dump(annotations, f, indent=4)

        print(f"Saved {len(annotations)} plots to {output_dir}")
        print(f"Annotations written to {annotation_file}")
