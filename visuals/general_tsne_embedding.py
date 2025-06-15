import os

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load precomputed embeddings and labels
embeddings_dir = "../data/dl_features/6class_bitiledlstm"  # Change if needed

output_filename = "multilabel_tsne_nopos6c.html"
pos_data = np.load(
    os.path.join(embeddings_dir, "data_True.npz"), allow_pickle=True
)
nopos_data = np.load(
    os.path.join(embeddings_dir, "data_False.npz"), allow_pickle=True
)
print(pos_data.files)
print(pos_data["embeddings"].shape)


X_pos = pos_data["embeddings"]
y_pos = pos_data["labels"]
X_nopos = nopos_data["embeddings"]
y_nopos = nopos_data["labels"]

print("X_pos shape:", X_pos.shape)
print("X_nopos shape:", X_nopos.shape)


# Load embeddings and labels
# X_pos = np.load(os.path.join(embeddings_dir, "embeddings_pos.npy"))
# X_nopos = np.load(os.path.join(embeddings_dir, "embeddings_nopos.npy"))
# y_pos = np.load(os.path.join(embeddings_dir, "labels_pos.npy"))
# y_nopos = np.load(os.path.join(embeddings_dir, "labels_nopos.npy"))
X_nopos = X_nopos[-515:]
X_pos = X_pos[-515:]
y_nopos = y_nopos[-515:]
y_pos = y_pos[-515:]

X = X_nopos
y = y_nopos

scaler = StandardScaler()
X_pos = scaler.fit_transform(X_pos)
X_nopos = scaler.fit_transform(X_nopos)

X_scaled = X_nopos
print("X_scaled.shape:", X_scaled.shape)


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
# If original columns are available (e.g., via y_df.columns)
if isinstance(y, pd.DataFrame):
    label_names = y.columns
    clean_label_names = label_names[y_clean_raw.to_numpy().argmax(axis=1)]
else:
    clean_label_names = y_clean_raw.argmax(axis=1)  # integer labels

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
print("Final label counts for Embedded Features:")
for label, count in label_counts.items():
    print(f"{label}: {count}")
