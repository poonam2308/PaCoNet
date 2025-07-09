import numpy as np
import pandas as pd
import random
import string
import os
import json

"""
Generate k synthetic datasets with strong correlation 
and minimal deviation from mean for each category

using this python class, we generate synthetic raw data which would be used
for creation of synthetic plots

"""


def generate_column_name():
    length = random.randint(4, 7)
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_category_name():
    length = random.randint(4, 7)
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

"""
Generate k synthetic datasets with strong correlation 
and minimal deviation from mean for each category
"""

def generate_synthetic_datasets(directory_path, k, annotation_file_name='annotations.json', num_rows=100, seed=None):
    all_annotations = []
    max_attributes = 0

    # Set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Ensure the directory exists
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    for i in range(k):
        # Randomly determine the number of columns (3 to 6)
        num_columns = random.randint(3, 6)
        max_attributes = max(max_attributes, num_columns)

        # Generate the mean vector and covariance matrix
        mean = np.random.rand(num_columns) * 10
        A = np.random.rand(num_columns, num_columns)
        cov = np.dot(A, A.transpose())  # Make it symmetric and positive-definite
        cov *= 0.9  # Scale the covariance to ensure strong correlation

        # Generate synthetic data
        data = np.random.multivariate_normal(mean, cov, num_rows)

        # Create DataFrame
        df = pd.DataFrame(data, columns=[generate_column_name() for _ in range(num_columns)])

        # Randomly apply different scaling factors to each column
        scaling_factors = [1, 10, 100, 1000]
        random.shuffle(scaling_factors)

        for j in range(num_columns):
            scaling_factor = scaling_factors[j % len(scaling_factors)]
            df.iloc[:, j] = df.iloc[:, j] * scaling_factor
            if j % 2 == 1:
                df.iloc[:, j] = df.iloc[:, j].round(2)

        # Generate categorical column with random names
        num_categories = random.randint(4, 6)
        categories = [generate_category_name() for _ in range(num_categories)]
        df['Category'] = np.random.choice(categories, num_rows)

        # Initialize annotations dictionary for this image
        image_annotations = {
            "image_id": f"image_{i + 1}",
            "image_path": f"image_{i + 1}.svg",
            "attributes": list(df.columns[:-1]),  # Exclude the 'Category' column
            "range_of_values": {},
            "average_of_values": {}
        }

        # Generate data for each category with minimal deviation from the mean
        for category in categories:
            category_mask = df['Category'] == category
            category_mean = np.random.rand(num_columns) * 10
            category_data = np.random.multivariate_normal(category_mean, cov, category_mask.sum())

            # Apply the same random scaling factors to category data
            for j in range(num_columns):
                scaling_factor = scaling_factors[j % len(scaling_factors)]
                category_data[:, j] *= scaling_factor

            df.loc[category_mask, df.columns[:-1]] = category_data

            # Calculate range and average for each feature in this category
            for col in df.columns[:-1]:
                col_min = df.loc[category_mask, col].min()
                col_max = df.loc[category_mask, col].max()
                col_avg = df.loc[category_mask, col].mean()

                if col not in image_annotations["range_of_values"]:
                    image_annotations["range_of_values"][col] = {"min": col_min, "max": col_max}
                    image_annotations["average_of_values"][col] = col_avg
                else:
                    image_annotations["range_of_values"][col]["min"] = min(
                        image_annotations["range_of_values"][col]["min"], col_min)
                    image_annotations["range_of_values"][col]["max"] = max(
                        image_annotations["range_of_values"][col]["max"], col_max)
                    image_annotations["average_of_values"][col] = (image_annotations["average_of_values"][
                                                                       col] + col_avg) / 2

        # Save to CSV and JSON with unique filenames
        csv_path = os.path.join(directory_path, f'synthetic_data_{i + 1}.csv')
        df.to_csv(csv_path, index=False)

        # Append this image's annotations to the list
        all_annotations.append(image_annotations)

    # Save all annotations to a single JSON file
    annotation_path = os.path.join(directory_path, annotation_file_name)
    with open(annotation_path, 'w') as f:
        json.dump(all_annotations, f, indent=4)

    print(f"Saved annotations to: {annotation_path}")

