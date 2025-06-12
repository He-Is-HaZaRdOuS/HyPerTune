import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

processed_dir = "../data/processed/"

input_filename = "expanded_synthetic_matrix_features.csv"
output_filename = "features_with_cosine_similarity_local.csv"

input_csv = f"{processed_dir}{input_filename}"
output_csv = f"{processed_dir}{output_filename}"

df = pd.read_csv(input_csv)

selected_features = [
    "NNZ",
    "Density",
    "Row NNZ Min / N",
    "Column NNZ Min / M",
    "Row NNZ Max / N",
    "Column NNZ Max / M",
    "Row NNZ STD / N",
    "Column NNZ STD / M",
    "Row NNZ Mean",
    "Column NNZ Mean",
    "Row NNZ Median",
    "Column NNZ Median",
    "Psym",
    "Bandwidth",
    "Profile",
    "avg_distance_to_diagonal",
    "num_diagonals_with_nonzeros",
    "nnz_bandwidth_std",
    "nnz_diagonal",
    "nnz_off_diagonal",
    "num_structurally_unsymmetric_elements",
    "num_empty_rows",
    "num_empty_cols",
    "row_sparsity_skew",
    "col_sparsity_skew",
    "row_nnz_entropy",
    "col_nnz_entropy",
]

statistical_features = [
    "Row NNZ Min / N",
    "Column NNZ Min / M",
    "Row NNZ Max / N",
    "Column NNZ Max / M",
    "Row NNZ STD / N",
    "Column NNZ STD / M",
    "Row NNZ Mean",
    "Column NNZ Mean",
    "Row NNZ Median",
    "Column NNZ Median",
]

structural_features = [
    "Bandwidth",
    "Profile",
    "avg_distance_to_diagonal",
    "num_diagonals_with_nonzeros",
    "nnz_bandwidth_std",
    "nnz_diagonal",
    "nnz_off_diagonal",
    "num_structurally_unsymmetric_elements",
]

# Initialize a dictionary to store cosine similarity for each row
cosine_sim_dict = {}

# Identify original matrices
original_matrix_names = df[~df["Variant Name"].str.contains("_expansion")][
    "Variant Name"
].values

# Process each original matrix group
for original_name in original_matrix_names:
    # Find all related matrices (including expansions, if any)
    group_rows = df[df["Variant Name"].str.startswith(original_name)].copy()

    # If the original itself isn’t included in this group, add it
    if original_name not in group_rows["Variant Name"].values:
        original_row = df[df["Variant Name"] == original_name]
        group_rows = pd.concat([group_rows, original_row], ignore_index=True)

    # Local scaling
    scaler = MinMaxScaler()
    group_rows[selected_features] = scaler.fit_transform(
        group_rows[selected_features]
    )

    # Boost statistical and structural features
    group_rows[statistical_features] *= 1.0
    group_rows[structural_features] *= 5.0

    # Get the original matrix vector
    original_vec = group_rows[group_rows["Variant Name"] == original_name][
        selected_features
    ].values[0]

    # Compute similarity for each matrix in this group
    for idx, row in group_rows.iterrows():
        matrix_name = row["Variant Name"]
        if matrix_name == original_name:
            cos_sim = 1.0
        else:
            synthetic_vec = row[selected_features].values
            cos_sim = cosine_similarity(
                original_vec.reshape(1, -1), synthetic_vec.reshape(1, -1)
            )[0][0]

        # Store it in the dict
        cosine_sim_dict[matrix_name] = cos_sim

# Map similarities to the DataFrame
df["cosine_similarity_local"] = df["Variant Name"].map(cosine_sim_dict)

# Any matrix not part of any group (no expansions) should get NaN
# If you want them to get 1.0 for themselves, fill with 1.0:
df["cosine_similarity_local"].fillna(1.0, inplace=True)

df.to_csv(output_csv, index=False)
print(f"Done! Wrote: {output_csv}")
