import pandas as pd

processed_dir = "../data/ml_ready/"

input_filename = "DISCRETIZED_matrix_data_dir_rec_6class.csv"

input_csv = f"{processed_dir}{input_filename}"
output_csv = f"{processed_dir}pruned_multilabel_synthetics_{input_filename}"


def prune_multilabel_or_unlabeled(
    csv_path, save_path=None, pattern="_expansion"
):
    df = pd.read_csv(csv_path)

    # Detect class columns (those starting with 'dir' or 'rec')
    class_cols = [
        col
        for col in df.columns
        if col.startswith("dir") or col.startswith("rec")
    ]

    # Count how many class columns are set to 1 in each row
    class_counts = df[class_cols].sum(axis=1)

    # Keep only rows with exactly one class active
    df = df[class_counts == 1]

    # Keep only non-expansion matrices
    df = df[~df["Matrix"].str.contains(pattern)]

    if save_path:
        df.to_csv(save_path, index=False)

    return df


# Example usage
df_clean = prune_multilabel_or_unlabeled(
    input_csv, output_csv, pattern="_expansion"
)
print(f"Done! Wrote: {output_csv}")
