def extract_distributions_from_excel(excel_path):
    import pandas as pd
    import numpy as np

    # Load Excel data
    df = pd.read_excel(excel_path)

    # Clean column names
    df.columns = [col.strip() for col in df.columns]

    # Identify the column with bin codes for number of rows
    bin_code_column = df.columns[2]

    # Define mapping from bin codes to approximate row counts
    bin_to_rows = {
        1: 75,
        2: 125,
        3: 175,
        4: 225,
        5: 275,
        6: 325,
        7: 400
    }

    # Create new column for approximate row counts
    df["approx_num_rows"] = df[bin_code_column].astype(int).map(bin_to_rows)

    # Build and normalize distributions
    axes_distribution = df["Number of vertical axes"].value_counts(normalize=True).sort_index()
    categories_distribution = df["Number of categories"].value_counts(normalize=True).sort_index()
    rows_distribution = df["approx_num_rows"].value_counts(normalize=True).sort_index()

    return axes_distribution, categories_distribution, rows_distribution
