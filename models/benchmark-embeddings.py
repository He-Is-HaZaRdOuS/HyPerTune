import os

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import (
    KFold,
    cross_val_score,
    cross_validate,
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Load the CSV dataset
ml_ready_dir = "../data/ml_ready/"
embeddings_dir = "../data/dl_features/"

input_filename = "DISCRETIZED_matrix_data_dir_rec_2class.csv"

input_csv = f"{ml_ready_dir}{input_filename}"

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
        "num_empty_rows",
        "num_empty_cols",
    ]
)

matrix_names_from_csv = df["Matrix"].tolist()

matrix_names = [line.strip() for line in matrix_names_from_csv]

df = df.set_index("Matrix").loc[matrix_names].reset_index()

X = df.drop(columns=drop_columns)
y = df[target_columns]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load embeddings and labels
# Load from NPZ files
pos_data = np.load(
    os.path.join(embeddings_dir, "data_True.npz"), allow_pickle=True
)
nopos_data = np.load(
    os.path.join(embeddings_dir, "data_False.npz"), allow_pickle=True
)


print(pos_data.files)  # Should include 'embeddings' and 'labels'


X_pos = pos_data["embeddings"]
y_pos = pos_data["labels"]

X_nopos = nopos_data["embeddings"]
y_nopos = nopos_data["labels"]

print("X_pos:", X_pos.shape, "y_pos:", y_pos.shape)
print("X_nopos:", X_nopos.shape, "y_nopos:", y_nopos.shape)

# Load matrix names from .npz
matrix_names_pos = pos_data["matrix_names"]
matrix_names_nopos = nopos_data["matrix_names"]

# Decode if needed (NumPy saves as bytes sometimes)
matrix_names_pos = [
    name.decode("utf-8") if isinstance(name, bytes) else name
    for name in matrix_names_pos
]
matrix_names_nopos = [
    name.decode("utf-8") if isinstance(name, bytes) else name
    for name in matrix_names_nopos
]

# Align y to match matrix order in the .npz
df_aligned_pos = df.set_index("Matrix").loc[matrix_names_pos]
df_aligned_nopos = df.set_index("Matrix").loc[matrix_names_nopos]

# Get corresponding label values
y_pos_csv = df_aligned_pos[target_columns].values
y_nopos_csv = df_aligned_nopos[target_columns].values

# Now compare
# print("Label mismatch (POS):", np.sum(y_pos != y_pos_csv))
# print("Label mismatch (NOPOS):", np.sum(y_nopos != y_nopos_csv))

# Optional: Standardize (helps for MLP, usually not needed for tree models)
scaler = StandardScaler()
X_pos = scaler.fit_transform(X_pos)
X_nopos = scaler.fit_transform(X_nopos)

print("First few matrices from CSV:", df["Matrix"].head().tolist())
print("Shape of POS embeddings:", X_pos.shape)

# Just to be safe — verify shape consistency
# assert X_pos.shape[0] == y_pos.shape[0], "Mismatch: POS embeddings and labels"
# assert X_nopos.shape[0] == y_nopos.shape[0], (
#     "Mismatch: NOPOS embeddings and labels"
# )

# Classifiers to test
classifiers = {
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
        random_state=42,
    ),
    "RandomForest": RandomForestClassifier(random_state=42),
    "MLP": MLPClassifier(random_state=42, max_iter=1000),
}

# Scoring metrics
scoring = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
}

# 5-Fold Cross Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)


def evaluate(X, y, label):
    print(f"\n=== Evaluating {label} embeddings ===")

    imp_threshold = 30
    print(
        f"Performing feature sampling to keep top {100 - imp_threshold}% of original features using ExtraTreesClassifier"
    )
    model = ExtraTreesClassifier(random_state=42).fit(X, y)
    importances = model.feature_importances_
    threshold = np.percentile(
        importances, imp_threshold
    )  # keep top 80% features
    important_indices = np.where(importances >= threshold)[0]
    if isinstance(X, pd.DataFrame):
        X = X.iloc[:, important_indices]
    else:
        X = X[:, important_indices]

    for name, clf in classifiers.items():
        print(f"\nClassifier: {name}")
        model = MultiOutputClassifier(clf, n_jobs=-1)
        scores = cross_validate(
            model, X, y, cv=kfold, scoring=scoring, n_jobs=-1
        )
        for metric in scoring:
            print(f"{metric}: {np.mean(scores[f'test_{metric}']):.4f}")


# X_nopos = X_nopos[-515:]
# X_pos = X_pos[-515:]

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# A. Original features
scores_raw = cross_val_score(XGBClassifier(), X, y, cv=cv)

# B. LSTM embeddings (only last 515!)
scores_emb = cross_val_score(XGBClassifier(), X_pos, y, cv=cv)

print("Raw accuracy:", scores_raw.mean())
print("LSTM-embedded accuracy:", scores_emb.mean())

# Run evaluations
evaluate(X, y, "VANILLA")
evaluate(X_pos, y, "POS")
evaluate(X_nopos, y, "NOPOS")
