import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# --------- Load Data ---------
ml_ready_dir = "../data/ml_ready/"

input_filename = "DISCRETIZED_matrix_data_dir_rec_6class.csv"

input_csv = f"{ml_ready_dir}{input_filename}"

output_filename = f"{input_filename}multilabel_tsne.html"

df = pd.read_csv(input_csv)

# Automatically detect target binary label columns (all with only 0/1 values)
target_columns = [
    col
    for col in df.columns
    if set(df[col].unique()) <= {0, 1} and col.startswith(("dir", "rec"))
]

# Drop non-feature columns
drop_columns = (
    [
        "Matrix",
        "Variant Name",
        "Group",
        "Kind",
        "Operation",
        "Method",
        "Generation Time(S)",
        "Match NNZ",
        "cosine_similarity_local",
    ]
    + target_columns
    + [
        "value_min",
        "value_max",
        "value_avg",
        "value_std",
        "row_min_min",
        "row_min_max",
        "row_min_mean",
        "row_min_std",
        "row_max_min",
        "row_max_max",
        "row_max_mean",
        "row_max_std",
        "row_mean_min",
        "row_mean_max",
        "row_mean_mean",
        "row_mean_std",
        "row_std_min",
        "row_std_max",
        "row_std_mean",
        "row_std_std",
        "row_median_min",
        "row_median_max",
        "row_median_mean",
        "row_median_std",
        "col_min_min",
        "col_min_max",
        "col_min_mean",
        "col_min_std",
        "col_max_min",
        "col_max_max",
        "col_max_mean",
        "col_max_std",
        "col_mean_min",
        "col_mean_max",
        "col_mean_mean",
        "col_mean_std",
        "col_std_min",
        "col_std_max",
        "col_std_mean",
        "col_std_std",
        "col_median_min",
        "col_median_max",
        "col_median_mean",
        "col_median_std",
        "norm_1",
        "norm_inf",
        "frobenius_norm",
        "estimated_condition_number",
        # "num_empty_rows",
        # "num_empty_cols",
    ]
)
print(f"df.shape[0]: {df.shape[1]}")
X = df.drop(columns=drop_columns)
print(f"X.shape[0]: {X.shape[1]}")
print(f"dropped columns: {len(drop_columns)}")
y = df[target_columns]

# --------- Standardize Features ---------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------- Analyze Label Types ---------
label_sums = y.sum(axis=1)
mask_clean = label_sums == 1
mask_ambiguous = label_sums > 1
mask_noclass = label_sums == 0

# --------- Prepare clean data for training ---------
X_clean = X_scaled[mask_clean]
y_clean_raw = y[mask_clean]

# Assign clean labels: get the label column where value is 1
final_labels = np.full(X.shape[0], fill_value="Unknown", dtype=object)
final_labels[mask_noclass] = "no_class"
point_type = np.full(X.shape[0], fill_value="Unknown", dtype=object)
point_type[mask_noclass] = "no_class"

# Create string label names for clean points
clean_label_names = y_clean_raw.idxmax(axis=1).values
final_labels[mask_clean] = clean_label_names
point_type[mask_clean] = "clean"

# --------- Train KNN and predict ambiguous ---------
if mask_ambiguous.any():
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_clean, clean_label_names)

    X_ambiguous = X_scaled[mask_ambiguous]
    y_pred_ambiguous = knn.predict(X_ambiguous)

    final_labels[mask_ambiguous] = y_pred_ambiguous
    point_type[mask_ambiguous] = "ambiguous"

# --------- 3D t-SNE ---------
tsne = TSNE(n_components=3, random_state=0, perplexity=10, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

df_plot = pd.DataFrame(
    {
        "tsne_1": X_tsne[:, 0],
        "tsne_2": X_tsne[:, 1],
        "tsne_3": X_tsne[:, 2],
        "label": final_labels,
        "point_type": point_type,
    }
)

fig = px.scatter(
    df_plot,
    x="tsne_1",
    y="tsne_2",
    color="label",
    symbol="point_type",
    symbol_map={"clean": "circle", "ambiguous": "diamond", "no_class": "x"},
    color_discrete_sequence=px.colors.qualitative.Set2,
    title="t-SNE Multi-Label Visualization with Reassigned Ambiguous Points",
)

# fig = px.scatter_3d(
#    df_plot,
#    x="tsne_1", y="tsne_2", z="tsne_3",
#    color="label",
#    symbol="point_type",
#    symbol_map={"clean": "circle", "ambiguous": "diamond", "no_class": "x"},
#    color_discrete_sequence=px.colors.qualitative.Set2,
#    title="t-SNE Multi-Label Visualization with Reassigned Ambiguous Points"
# )

fig.update_traces(marker=dict(size=5, opacity=0.8))
fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))

fig.write_html(output_filename)
fig.show()

# --------- Print Label Counts ---------
from collections import Counter

label_counts = Counter(final_labels)
print(f"Final label counts for {input_filename}:")
for label, count in label_counts.items():
    print(f"{label}: {count}")
