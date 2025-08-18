import os
import json
import numpy as np
import pandas as pd
import altair as alt
from pc.plot_gen.plot_utils import normalize_column, hsv_to_rgb, create_ticks_labels, calculate_pixel_positions, \
    safe_join, generate_hsv_pool
import random
from pc.plot_gen.coordinate_extraction import CoordinateExtraction

# plot library : https://altair-viz.github.io/user_guide/customization.html
class MultiCatPCPGenerator:
    def __init__(self, width=600, height=300, seed =42):
        self.width = width
        self.height = height
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)

    def generate_plot(self, df, filename=None, background_value=255,
                      grid_on=False, show_ticks_labels=False, category_hsv_map=None, save_png=False):

        column_names = sorted(list(df.columns)[:-1])
        color_column = df.columns[-1]

        normalized_columns = [normalize_column(df, col) for col in column_names]

        unique_categories = sorted(df[color_column].unique())

        # Step 1: Generate HSV pool of 100 vibrant colors
        if category_hsv_map is None:
            # original random assignment (kept as-is)
            hsv_pool = generate_hsv_pool(30)
            selected_indices = np.random.choice(len(hsv_pool), len(unique_categories), replace=False)
            selected_hsvs = [hsv_pool[i] for i in selected_indices]
            category_hsv_map = dict(zip(unique_categories, selected_hsvs))
        # Step 3: Assign these random HSV colors to each category
        category_colors = {
            category: hsv_to_rgb(hsv['h'], hsv['s'], hsv['v'])
            for category, hsv in category_hsv_map.items()
        }

        base = alt.Chart(df).transform_window(index="count()").transform_fold(
            normalized_columns
        ).transform_calculate(
            mid="(datum.value + datum.value) / 2"
        ).properties(
            width=self.width,
            height=self.height
        )

        lines = base.mark_line(opacity=0.4).encode(
            x=alt.X('key:N', axis=alt.Axis(
                title=None,
                domain=False,
                labels=show_ticks_labels,
                labelAngle=0,
                ticks=False)),
            y=alt.Y('value:Q', axis=alt.Axis(
                title=None,
                domain =False,
                labels=False,
                ticks= False,
                grid=grid_on)),
            color=alt.Color(f"{color_column}:N", scale=alt.Scale(
                domain=list(category_colors.keys()),
                range=['rgb({},{},{})'.format(*color) for color in category_colors.values()]
            ),legend=None),
            detail="index:N",
            tooltip=column_names
        )

        rules = base.mark_rule(color="#ccc", tooltip=None).encode(
            x=alt.X('key:N', axis=alt.Axis(title=None, labels=False, ticks=False))
        )

        tick_dfs = [create_ticks_labels(df, norm_col, orig_col)
                    for norm_col, orig_col in zip(normalized_columns, column_names)]
        ticks_labels_df = pd.concat(tick_dfs)

        if show_ticks_labels:
            ticks = alt.Chart(ticks_labels_df).mark_tick(size=8, color="#ccc", orient="horizontal").encode(
                x='variable:N', y='value:Q')
            labels = alt.Chart(ticks_labels_df).mark_text(
                align='center', baseline='middle', dx=-10).encode(
                x='variable:N', y='value:Q', text='label:N')
        else:
            ticks = alt.Chart(ticks_labels_df).mark_tick(size=8, opacity=0, orient="horizontal").encode(
                x='variable:N', y='value:Q')
            labels = alt.Chart(ticks_labels_df).mark_text(
                align='center', baseline='middle', dx=-10, opacity=0).encode(
                x='variable:N', y='value:Q', text='label:N')

        # Axis configuration
        axis_config_x = {
            "domain": False,
            "labelAngle": 0,
            "title": None,
            "tickColor": "#ccc" if show_ticks_labels else "transparent",
            "labelColor": "#000" if show_ticks_labels else "transparent",
            "grid": grid_on,
            "gridColor": "#ccc" if grid_on else "transparent"
        }

        axis_config_y = {
            "domain": False,
            "labelAngle": 0,
            "title": None,
            "tickColor": "#ccc" if show_ticks_labels else "transparent",
            "labelColor": "#000" if show_ticks_labels else "transparent",
            "grid": grid_on,
            "gridColor": "#ccc" if grid_on else "transparent"
        }

        chart = alt.layer(lines, rules, ticks, labels).configure_axisX(
            **axis_config_x
        ).configure_axisY(
            **axis_config_y
        ).configure_view(
            stroke=None
        )

        if background_value is not None and int(background_value) < 255:
            background_rgb = f"rgb({background_value},{background_value},{background_value})"
            chart = chart.configure(background=background_rgb)

        if filename:
            chart.save(filename)
            if save_png:
                base, _ = os.path.splitext(filename)
                png_dir = os.path.join(os.path.dirname(base), "rasterized_images")
                os.makedirs(png_dir, exist_ok=True)
                png_filename = os.path.join(png_dir, os.path.basename(base) + ".png")
                chart.save(png_filename, format="png")
                print(f"Saved PNG: {png_filename}")

        return chart, normalized_columns, category_hsv_map

    def generate_individual_plots(self, df, output_dir, filename_prefix,background_value=255,
                      grid_on=False, show_ticks_labels=False, category_hsv_map=None):
        """Generate separate plots for each category in the DataFrame."""
        os.makedirs(output_dir, exist_ok=True)

        column_names = sorted(list(df.columns)[:-1])
        color_column = df.columns[-1]
        normalized_columns = [normalize_column(df, col) for col in column_names]

        # If no color map provided, fall back to generate_plot logic
        if category_hsv_map is None:
            _, _, category_hsv_map = self.generate_plot(df)

        # Convert HSV map to RGB map
        category_colors = {
            category: hsv_to_rgb(hsv['h'], hsv['s'], hsv['v'])
            for category, hsv in category_hsv_map.items()
        }


        for category in df[color_column].unique():
            df_category = df[df[color_column] == category]
            if df_category.empty:
                continue

            output_filename = os.path.join(output_dir, f"image_{filename_prefix}_{category}.svg")

            # Make a single-category plot with fixed color
            base = alt.Chart(df_category).transform_window(index="count()").transform_fold(
                normalized_columns
            ).transform_calculate(
                mid="(datum.value + datum.value) / 2"
            ).properties(width=self.width, height=self.height)

            lines = base.mark_line(opacity=0.4).encode(
                x='key:N',
                y=alt.Y('value:Q', axis=None),
                color=alt.value(f'rgb({category_colors[category][0]},'
                                f'{category_colors[category][1]},'
                                f'{category_colors[category][2]})'),
                detail="index:N"
            )

            rules = base.mark_rule(color="#ccc", tooltip=None).encode(x="key:N")

            # Shared tick labels
            tick_dfs = [create_ticks_labels(df, norm_col, orig_col)
                        for norm_col, orig_col in zip(normalized_columns, column_names)]
            ticks_labels_df = pd.concat(tick_dfs)

            if show_ticks_labels:
                ticks = alt.Chart(ticks_labels_df).mark_tick(size=8, color="#ccc", orient="horizontal").encode(
                    x='variable:N', y='value:Q')
                labels = alt.Chart(ticks_labels_df).mark_text(
                    align='center', baseline='middle', dx=-10).encode(
                    x='variable:N', y='value:Q', text='label:N')
            else:
                ticks = alt.Chart(ticks_labels_df).mark_tick(size=8, opacity=0, orient="horizontal").encode(
                    x='variable:N', y='value:Q')
                labels = alt.Chart(ticks_labels_df).mark_text(
                    align='center', baseline='middle', dx=-10, opacity=0).encode(
                    x='variable:N', y='value:Q', text='label:N')

            # Axis configuration
            axis_config_x = {
                "domain": False,
                "labelAngle": 0,
                "title": None,
                "tickColor": "#ccc" if show_ticks_labels else "transparent",
                "labelColor": "#000" if show_ticks_labels else "transparent",
                "grid": grid_on,
                "gridColor": "#ccc" if grid_on else "transparent"
            }

            axis_config_y = {
                "domain": False,
                "labelAngle": 0,
                "title": None,
                "tickColor": "#ccc" if show_ticks_labels else "transparent",
                "labelColor": "#000" if show_ticks_labels else "transparent",
                "grid": grid_on,
                "gridColor": "#ccc" if grid_on else "transparent"
            }

            chart = alt.layer(lines, rules, ticks, labels).configure_axisX(
                **axis_config_x
            ).configure_axisY(
                **axis_config_y
            ).configure_view(
                stroke=None
            )

            if background_value is not None and int(background_value) < 255:
                background_rgb = f"rgb({background_value},{background_value},{background_value})"
                chart = chart.configure(background=background_rgb)


            chart.save(output_filename)
            print(f"Saved {output_filename}")

    def generate_batch(
            self,
            input_dir,
            output_dir,
            num_files,
            annotation_file="dist_annotations.json",
            background_distribution=None,
            grid_distribution=None,
            ticks_labels_distribution=None,
            no_ticks_output_dir=None,
    ):
        """Generate plots and annotations for a directory of CSV files with real-world style distributions."""
        os.makedirs(output_dir, exist_ok=True)
        if no_ticks_output_dir is not None:
            os.makedirs(no_ticks_output_dir, exist_ok=True)
        annotation_file = safe_join(output_dir, annotation_file)
        annotations = []

        for i in range(1, num_files + 1):
            csv_path = os.path.join(input_dir, f'synthetic_data_{i}.csv')
            if not os.path.exists(csv_path):
                print(f"File not found: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            df1 = pd.read_csv(csv_path)
            df2 = pd.read_csv(csv_path)
            df3 = pd.read_csv(csv_path)
            color_column = df.columns[-1]
            unique_categories = df[color_column].unique()
            selected_categories = unique_categories

            df = df[df[color_column].isin(selected_categories)]
            if df.empty:
                print(f"No valid categories in {csv_path}")
                continue

            # Sample style attributes from real distributions
            background_value = None
            grid_on = False
            show_ticks_labels = False

            if background_distribution is not None:
                background_value = np.random.choice(background_distribution.index, p=background_distribution.values)
            if grid_distribution is not None:
                grid_on = bool(np.random.choice(grid_distribution.index, p=grid_distribution.values))
            if ticks_labels_distribution is not None:
                show_ticks_labels = bool(np.random.choice(ticks_labels_distribution.index, p=ticks_labels_distribution.values))

            output_file = os.path.join(output_dir, f'image_{i}.svg')
            _, normalized_columns, category_hsv_map = self.generate_plot(
                df, filename=output_file, background_value=background_value, grid_on=grid_on,
                show_ticks_labels=show_ticks_labels, svg_png=True
            )


            if no_ticks_output_dir:
                output_file_no_ticks = os.path.join(no_ticks_output_dir, f'image_{i}.svg')
                indiv_dir_with_ticks= os.path.join(no_ticks_output_dir, "per_category")
                indiv_dir_no_ticks = os.path.join(no_ticks_output_dir, "per_category_noticks")
                _, _, _ = self.generate_plot(
                    df1, filename=output_file_no_ticks, background_value=background_value, grid_on=False,
                    show_ticks_labels=False, category_hsv_map=category_hsv_map
                )
                self.generate_individual_plots(
                    df2, output_dir=indiv_dir_with_ticks, filename_prefix=i, background_value=background_value, grid_on=grid_on,
                    show_ticks_labels=show_ticks_labels, category_hsv_map=category_hsv_map)

                self.generate_individual_plots(
                    df3, output_dir=indiv_dir_no_ticks, filename_prefix=i,background_value=background_value, grid_on=False,
                    show_ticks_labels=False,  category_hsv_map=category_hsv_map)



            extractor = CoordinateExtraction(normalize_y_to_plot=False)
            # lines_by_region = extractor.extract_line_coordinates(output_file)["lines_by_region"]

            vertical_axes = extractor.extract_vertical_axes(output_file)
            category_colors = {
                str(cat): {
                    "h": round(hsv['h'], 4),
                    "s": round(hsv['s'], 4),
                    "v": round(hsv['v'], 4)
                }
                for cat, hsv in category_hsv_map.items()
            }
            # normalize category_colors: ensure rgb
            rgb_category_colors = {}
            for cat, val in category_colors.items():
                if isinstance(val, dict) and "h" in val:  # HSV dict
                    rgb_category_colors[cat] = hsv_to_rgb(val["h"], val["s"], val["v"])
                else:  # already rgb tuple
                    rgb_category_colors[cat] = val

            lines_by_region = extractor.extract_line_coordinates_by_category(
                output_file,
                category_colors=rgb_category_colors
            )["lines_by_region"]

            ann = {
                "filename": os.path.basename(output_file),
                "vertical_axes": [round(float(x), 2) for x in vertical_axes],
                "category_colors": category_colors,
                "lines_by_region": {
                    crop: {
                        cat: [[round(float(a), 2) for a in line] for line in lines]
                        for cat, lines in categories.items()
                    }
                    for crop, categories in lines_by_region.items()
                }
            }

            json_path = os.path.join(
                output_dir,
                os.path.splitext(os.path.basename(output_file))[0] + ".json"
            )
            with open(json_path, "w") as jf:
                json.dump(ann, jf, indent=4)


            image_annotation = {
                'image_name': os.path.basename(output_file),
                'image_style': {
                    'background_rgb': int(background_value) if background_value is not None else None,
                    'grid': grid_on,
                    'ticks_labels': show_ticks_labels
                },
                'categories': []
            }

            for category in unique_categories:
                df_cat = df[df[color_column] == category]
                pixel_positions = calculate_pixel_positions(df_cat, normalized_columns, self.height)

                if category not in selected_categories:
                    continue

                # h_index = np.where(selected_categories == category)[0][0]
                # color_hsv = {'h': round(h_index / len(unique_categories), 2), 's': 1, 'v': 1}

                color_hsv = category_hsv_map[category]

                category_stats = {
                    'category': category,
                    'color_hsv': color_hsv,
                    'color_rgb': hsv_to_rgb(color_hsv['h'], color_hsv['s'], color_hsv['v']),
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

