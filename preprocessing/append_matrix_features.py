import numpy as np
import pandas as pd

# Load the two CSV files into pandas DataFrames
raw_dir = "../data/raw/"
processed_dir = "../data/processed/"

input_labels_filename = "6class.csv"
input_features_filename = "features_with_cosine_similarity_local.csv"

input_labelled_data = f"{raw_dir}{input_labels_filename}"
input_features_data = f"{processed_dir}{input_features_filename}"

output_csv = f"{processed_dir}matrix_data_{input_labels_filename}"

csv_data = pd.read_csv(input_labelled_data)
matrix_features = pd.read_csv(input_features_data)

# Fill N/A values in label columns with int64 max
columns = [col for col in csv_data.columns if col.startswith(("dir", "rec"))]
csv_data[columns] = (
    csv_data[columns]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(np.iinfo(np.int64).max)
)

# Find the intersection of matrix names (ignoring case and potential extensions)
intersecting_matrices = pd.merge(
    csv_data,
    matrix_features,
    left_on="Matrix",
    right_on="Variant Name",
    how="inner",
)

# Replace NaN values with "N/A"
intersecting_matrices = intersecting_matrices.fillna(0)

# Drop the redundant 'Matrix Name' column
intersecting_matrices = intersecting_matrices.drop(columns=["Matrix Name"])

# Save the intersecting matrices to a new CSV
intersecting_matrices.to_csv(output_csv, index=False)

print(
    f"A new CSV containing only intersecting matrices has been saved as {output_csv}."
)
