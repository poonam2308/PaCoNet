import numpy as np
import pandas as pd
import random
import string
import os
import json

def generate_column_name():
    length = random.randint(4, 7)
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def generate_category_name():
    length = random.randint(4, 7)
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def test_generate_synthetic_datasets(
    directory_path,
    k,
    annotation_file_name='annotations.json',
    num_rows=100,
    seed=None,
    num_columns=None,
    column_definitions=None,  # list of (name, (min, max), scale) if needed
    category_names=None
):
    all_annotations = []

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    for i in range(k):
        if column_definitions:
            columns_info = column_definitions
            num_columns = len(columns_info)
            col_names = [col[0] for col in columns_info]
        else:
            num_columns = num_columns or random.randint(3, 6)
            col_names = [generate_column_name() for _ in range(num_columns)]
            columns_info = [(name, (0, 10), 1) for name in col_names]  # default range and scale

        A = np.random.rand(num_columns, num_columns)
        cov = np.dot(A, A.transpose()) * 0.9
        mean = np.random.rand(num_columns) * 10
        data = np.random.multivariate_normal(mean, cov, num_rows)
        df = pd.DataFrame(data, columns=col_names)

        scaling_factors = [col[2] if len(col) > 2 else 1 for col in columns_info]
        for j in range(num_columns):
            df.iloc[:, j] *= scaling_factors[j % len(scaling_factors)]

        if category_names:
            categories = category_names
        else:
            num_categories = random.randint(4, 6)
            categories = [generate_category_name() for _ in range(num_categories)]

        df['Category'] = np.random.choice(categories, num_rows)

        image_annotations = {
            "image_id": f"image_{i + 1}",
            "image_path": f"image_{i + 1}.svg",
            "attributes": list(df.columns[:-1]),
            "range_of_values": {},
            "average_of_values": {}
        }

        for category in categories:
            mask = df['Category'] == category
            cat_mean = np.random.rand(num_columns) * 10
            cat_data = np.random.multivariate_normal(cat_mean, cov, mask.sum())

            for j in range(num_columns):
                cat_data[:, j] *= scaling_factors[j % len(scaling_factors)]

            df.loc[mask, df.columns[:-1]] = cat_data

            for col in df.columns[:-1]:
                col_min = df.loc[mask, col].min()
                col_max = df.loc[mask, col].max()
                col_avg = df.loc[mask, col].mean()

                if col not in image_annotations["range_of_values"]:
                    image_annotations["range_of_values"][col] = {"min": col_min, "max": col_max}
                    image_annotations["average_of_values"][col] = col_avg
                else:
                    image_annotations["range_of_values"][col]["min"] = min(
                        image_annotations["range_of_values"][col]["min"], col_min)
                    image_annotations["range_of_values"][col]["max"] = max(
                        image_annotations["range_of_values"][col]["max"], col_max)
                    image_annotations["average_of_values"][col] = (
                        image_annotations["average_of_values"][col] + col_avg) / 2

        df.to_csv(os.path.join(directory_path, f'synthetic_data_{i + 1}.csv'), index=False)
        all_annotations.append(image_annotations)

    with open(os.path.join(directory_path, annotation_file_name), 'w') as f:
        json.dump(all_annotations, f, indent=4)

    print(f"Saved annotations to: {os.path.join(directory_path, annotation_file_name)}")