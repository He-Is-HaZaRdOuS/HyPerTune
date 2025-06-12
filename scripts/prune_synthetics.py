import pandas as pd

processed_dir = "../data/ml_ready/"

input_filename = "DISCRETIZED_matrix_data_dir_rec_6class.csv"

input_csv = f"{processed_dir}{input_filename}"
output_csv = f"{processed_dir}pruned_synthetics_{input_filename}"


def prune_expansions(csv_path, save_path=None, pattern="_expansion"):
    df = pd.read_csv(csv_path)
    # Keep only non-expansion matrices
    df_clean = df[~df["Matrix"].str.contains(pattern)]
    if save_path:
        df_clean.to_csv(save_path, index=False)
    return df_clean


# Example usage
df_clean = prune_expansions(input_csv, output_csv, pattern="_expansion")
print(f"Done! Wrote: {output_csv}")
