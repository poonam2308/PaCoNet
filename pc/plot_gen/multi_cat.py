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
        self.extractor = CoordinateExtraction(normalize_y_to_plot=False)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def generate_plot(self, df, filename=None, background_value=255,
                      grid_on=False, show_ticks_labels=False, category_hsv_map=None,
                      save_png=False,svg_dir=None, do_extraction=False):

        column_names = sorted(list(df.columns)[:-1])
        color_column = df.columns[-1]

        normalized_columns = [normalize_column(df, col) for col in column_names]
        unique_categories = sorted(df[color_column].unique())

        if category_hsv_map is None:
            hsv_pool = generate_hsv_pool(30)
            selected_indices = np.random.choice(len(hsv_pool), len(unique_categories), replace=False)
            selected_hsvs = [hsv_pool[i] for i in selected_indices]
            category_hsv_map = dict(zip(unique_categories, selected_hsvs))
        category_colors = {
            category: hsv_to_rgb(hsv['h'], hsv['s'], hsv['v'])
            for category, hsv in category_hsv_map.items()
        }

        # altair chart components: base, lines, rules,
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

        axis_config_y = axis_config_x.copy()

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


        base, _ = os.path.splitext(filename)
        basename = os.path.basename(base)
        output_dir = os.path.dirname(filename)

        if svg_dir:
            os.makedirs(svg_dir, exist_ok=True)
            svg_filename = os.path.join(svg_dir, basename + ".svg")
        else:
            svg_filename = base + ".svg"

        png_filename = os.path.join(output_dir, basename + ".png")
        json_filename = os.path.join(output_dir, basename + ".json")
        chart.save(svg_filename)
        if save_png:
            chart.save(png_filename, format="png")

        if do_extraction:
            vertical_axes = self.extractor.extract_vertical_axes(svg_filename)

            rgb_category_colors = {
                str(cat): hsv_to_rgb(hsv["h"], hsv["s"], hsv["v"])
                for cat, hsv in category_hsv_map.items()
            }
            lines_by_region = self.extractor.extract_line_coordinates_by_category(
                svg_filename, category_colors=rgb_category_colors
            )["lines"]

            ann = {
                "filename": basename + ".png",
                "vertical_axes": [round(float(x), 2) for x in vertical_axes],
                "category_colors": {
                    str(cat): {
                        "h": round(hsv['h'], 4),
                        "s": round(hsv['s'], 4),
                        "v": round(hsv['v'], 4)
                    }
                    for cat, hsv in category_hsv_map.items()
                },
                "lines": {
                    crop: {
                        cat: [[round(float(a), 2) for a in line] for line in lines]
                        for cat, lines in categories.items()
                    }
                    for crop, categories in lines_by_region.items()
                }
            }
            with open(json_filename, "w") as jf:
                json.dump(ann, jf, indent=4)

        return chart, normalized_columns, category_hsv_map

    def generate_individual_plots(self, df, output_dir, filename_prefix,background_value=255,
                      grid_on=False, show_ticks_labels=False, category_hsv_map=None):

        column_names = sorted(list(df.columns)[:-1])
        color_column = df.columns[-1]
        normalized_columns = [normalize_column(df, col) for col in column_names]

        category_colors = {
            category: hsv_to_rgb(hsv['h'], hsv['s'], hsv['v'])
            for category, hsv in category_hsv_map.items()
        }
        for category in df[color_column].unique():
            df_category = df[df[color_column] == category]
            if df_category.empty:
                continue

            output_filename = os.path.join(output_dir, f"image_{filename_prefix}_{category}.svg")

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

            axis_config_y = axis_config_x.copy()

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

    def generate_batch(
            self, input_dir,output_dir,svg_dir,num_files,
            background_distribution=None,
            grid_distribution=None,
            ticks_labels_distribution=None,
            no_ticks_output_dir=None,
            per_cat_dir=None,
            per_cat_ntl_dir=None
    ):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(svg_dir, exist_ok=True)
        if no_ticks_output_dir is not None:
            os.makedirs(no_ticks_output_dir, exist_ok=True)
        if per_cat_dir is not None:
            os.makedirs(per_cat_dir, exist_ok=True)
        if per_cat_ntl_dir is not None:
            os.makedirs(per_cat_ntl_dir, exist_ok=True)

        annotation_file = safe_join(svg_dir, "dist_annotations.json")
        annotations = []

        for i in range(1, num_files + 1):
            csv_path = os.path.join(input_dir, f'synthetic_data_{i}.csv')
            file_name = f"image_{i}.svg"
            if not os.path.exists(csv_path):
                print(f"File not found: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            df1 = pd.read_csv(csv_path)
            df2 = pd.read_csv(csv_path)
            df3 = pd.read_csv(csv_path)

            color_column = df.columns[-1]
            unique_categories = df[color_column].unique()

            background_value = None
            grid_on = False
            show_ticks_labels = False

            if background_distribution is not None:
                background_value = np.random.choice(background_distribution.index, p=background_distribution.values)
            if grid_distribution is not None:
                grid_on = bool(np.random.choice(grid_distribution.index, p=grid_distribution.values))
            if ticks_labels_distribution is not None:
                show_ticks_labels = bool(
                    np.random.choice(ticks_labels_distribution.index, p=ticks_labels_distribution.values))


            output_file = os.path.join(output_dir, file_name)
            chart, normalized_columns, category_hsv_map = self.generate_plot(
                df, filename=output_file, background_value=background_value, grid_on=grid_on,
                show_ticks_labels=show_ticks_labels,save_png=True,svg_dir=svg_dir, do_extraction=True)

            if no_ticks_output_dir:
                output_file_no_ticks = os.path.join(no_ticks_output_dir, file_name)
                self.generate_plot(
                    df1,
                    filename=output_file_no_ticks,
                    background_value=background_value,
                    grid_on=False,
                    show_ticks_labels=False,
                    category_hsv_map=category_hsv_map)

            if per_cat_dir:
                self.generate_individual_plots(
                    df2, output_dir=per_cat_dir, filename_prefix=i,
                    background_value=background_value, grid_on=grid_on,
                    show_ticks_labels=show_ticks_labels, category_hsv_map=category_hsv_map)
            if per_cat_ntl_dir:
                self.generate_individual_plots(
                    df3, output_dir=per_cat_ntl_dir, filename_prefix=i,
                    background_value=background_value, grid_on=False,
                    show_ticks_labels=False, category_hsv_map=category_hsv_map)


            # --- SVG Statistical annotations ---
            image_annotation = {
                'image_name': file_name,
                'image_style': {
                    'background_rgb': int(background_value) if background_value is not None else None,
                    'grid': grid_on,
                    'ticks_labels': show_ticks_labels
                },
                'categories': []
            }

            for category in unique_categories:
                df_cat = df[df[color_column] == category]
                if df_cat.empty:
                    continue

                pixel_positions = calculate_pixel_positions(df_cat, normalized_columns, self.height)
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

        # Save all annotations
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f, indent=4)
