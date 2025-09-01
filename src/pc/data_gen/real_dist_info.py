import pandas as pd

def extract_distributions_from_excel(excel_path):

    df = pd.read_excel(excel_path)
    df.columns = [col.strip() for col in df.columns]
    bin_code_column = df.columns[2]

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


def extract_dist_plots_from_excel(excel_path):
    df = pd.read_excel(excel_path)
    df.columns = [col.strip() for col in df.columns]

    def to_bool(val):
        val = str(val).strip().lower()
        if val in {"yes", "true", "present", "1"}:
            return True
        elif val in {"no", "false", "absent", "0"}:
            return False
        else:
            return None  # Or raise ValueError if unexpected

    df["Grid"] = df["Grid"].apply(to_bool)
    df["Presence of Ticks, labels"] = df["Presence of Ticks, labels"].apply(to_bool)

    background_distribution = df["Background RGB"].value_counts(normalize=True).sort_index()
    grid_distribution = df["Grid"].value_counts(normalize=True).sort_index()
    ticks_labels_distribution = df["Presence of Ticks, labels"].value_counts(normalize=True).sort_index()


    return background_distribution,grid_distribution, ticks_labels_distribution
