import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# Helper: extract base name (handles expansions)
def base_matrix_name(name):
    match = re.match(r"(.+?)_expansion_\d+", name)
    return match.group(1) if match else name


def infer_class_count(filename):
    """
    Infers class count (e.g., 2 or 6) from a file name.
    Returns an integer or None if unknown.
    """
    basename = os.path.basename(filename).lower()

    # Try regex first
    match = re.search(r"(\d)[-_]?class", basename)
    if match:
        return int(match.group(1))

    # Fallback to substring checks
    if "2class" in basename or "binary" in basename:
        return 2
    if "6class" in basename or "multiclass" in basename:
        return 6

    return None  # Unknown


# Load the CSV dataset
ml_ready_dir = "../data/ml_ready/"
# embeddings_dir = "../data/dl_features/"

input_filename = "DISCRETIZED_matrix_data_dir_rec_2class.csv"

embeddings_subdir = None
classes = infer_class_count(input_filename)
if classes == 2:
    embeddings_subdir = "2class_bitiledlstm/"
elif classes == 6:
    embeddings_subdir = "6class_bitiledlstm/"
else:
    print("CSV not supported!")
    sys.exit(0)

embeddings_dir = f"../data/dl_features/{embeddings_subdir}"

input_csv = f"{ml_ready_dir}{input_filename}"

df = pd.read_csv(input_csv)

# Identify binary target columns
target_columns = [
    col
    for col in df.columns
    if set(df[col].dropna().unique()) <= {0, 1}
    and (col.startswith("dir") or col.startswith("rec"))
]

# Define columns to drop before modeling
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
        col
        for col in df.columns
        if any(
            prefix in col
            for prefix in [
                "value_",
                "row_",
                "col_",
                "norm_",
                "frobenius_norm",
                "estimated_condition_number",
                "num_empty_rows",
                "num_empty_cols",
            ]
        )
    ]
)

# Build grouping map for expansions
matrix_to_base = {m: base_matrix_name(m) for m in df["Matrix"]}
base_to_all = defaultdict(list)
for m, b in matrix_to_base.items():
    base_to_all[b].append(m)

# Load embeddings (.npz)
# ------------------ Load and Align Embeddings ------------------


def load_embeddings(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    names = [
        n.decode("utf-8") if isinstance(n, bytes) else n
        for n in data["matrix_names"]
    ]
    X = data["embeddings"]
    return names, X


# Load both POS and NOPOS embeddings
matrix_names_pos, X_pos = load_embeddings(
    os.path.join(embeddings_dir, "data_True.npz")
)
matrix_names_nopos, X_nopos = load_embeddings(
    os.path.join(embeddings_dir, "data_False.npz")
)

# Filter to only matrices that are present in the CSV
valid_set = set(df["Matrix"])
matrix_names_pos = [m for m in matrix_names_pos if m in valid_set]
matrix_names_nopos = [m for m in matrix_names_nopos if m in valid_set]

# Align embeddings with CSV labels
df_pos = df.set_index("Matrix").loc[matrix_names_pos]
df_nopos = df.set_index("Matrix").loc[matrix_names_nopos]

y_pos = df_pos[target_columns].values
y_nopos = df_nopos[target_columns].values

# Normalize embeddings
scaler_pos = StandardScaler()
X_pos = scaler_pos.fit_transform(
    X_pos[: len(matrix_names_pos)]
)  # ensure same shape
scaler_nopos = StandardScaler()
X_nopos = scaler_nopos.fit_transform(X_nopos[: len(matrix_names_nopos)])

# Groups for POS and NOPOS
groups_pos = [matrix_to_base[m] for m in matrix_names_pos]
groups_nopos = [matrix_to_base[m] for m in matrix_names_nopos]


# Utility to filter embeddings arrays
def filter_embeddings(names, X, y):
    filtered = [(n, x, yy) for n, x, yy in zip(names, X, y) if n in valid_set]
    if not filtered:
        return [], np.empty((0, X.shape[1])), np.empty((0, y.shape[1]))
    names_f, X_f, y_f = zip(*filtered)
    return list(names_f), np.array(X_f), np.array(y_f)


matrix_names_pos, X_pos, y_pos = filter_embeddings(
    matrix_names_pos, X_pos, y_pos
)
matrix_names_nopos, X_nopos, y_nopos = filter_embeddings(
    matrix_names_nopos, X_nopos, y_nopos
)

# Align CSV labels with embeddings
df_pos = df.set_index("Matrix").loc[matrix_names_pos]
df_nopos = df.set_index("Matrix").loc[matrix_names_nopos]
y_pos_csv = df_pos[target_columns].values
y_nopos_csv = df_nopos[target_columns].values

# Check label alignment
assert np.array_equal(y_pos, y_pos_csv), "POS labels mismatch"
assert np.array_equal(y_nopos, y_nopos_csv), "NOPOS labels mismatch"

# Prepare feature matrix X and labels y for vanilla CSV features
X = df.drop(columns=drop_columns, errors="ignore")
y = df[target_columns].values

# Scale CSV features
scaler_vanilla = StandardScaler()
X_scaled = scaler_vanilla.fit_transform(X)

# Scale embeddings
scaler_pos = StandardScaler()
X_pos = scaler_pos.fit_transform(X_pos)
scaler_nopos = StandardScaler()
X_nopos = scaler_nopos.fit_transform(X_nopos)

# ----------------- Prepare group-aware cross-validation -----------------
groups = [matrix_to_base[name] for name in df["Matrix"]]
group_kfold = GroupKFold(n_splits=5)

# Classifiers
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
scoring = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
}


def evaluate_cv(X_data, y_data, name, groups_override=None):
    print(f"\n=== Evaluating {name} embeddings/features ===")
    et = ExtraTreesClassifier(random_state=42).fit(X_data, y_data)
    thr = np.percentile(et.feature_importances_, 30)
    idx = np.where(et.feature_importances_ >= thr)[0]
    X_sel = X_data[:, idx]

    groups_used = groups_override if groups_override else groups

    for clf_name, clf in classifiers.items():
        print(f"\nClassifier: {clf_name}")
        model = MultiOutputClassifier(clf, n_jobs=-1)
        scores = cross_validate(
            model,
            X_sel,
            y_data,
            cv=group_kfold.split(X_data, y_data, groups_used),
            scoring=scoring,
            n_jobs=-1,
        )
        for metric in scoring:
            print(f"{metric}: {scores['test_' + metric].mean():.4f}")


# ----------------- High-level raw vs embedded comparison -----------------
scores_raw = cross_validate(
    XGBClassifier(),
    X_scaled,
    y,
    cv=group_kfold.split(X_scaled, y, groups),
    scoring=["accuracy"],
    n_jobs=-1,
)
scores_pos = cross_validate(
    XGBClassifier(),
    X_pos,
    y_pos,
    cv=group_kfold.split(X_pos, y_pos, matrix_names_pos),
    scoring=["accuracy"],
    n_jobs=-1,
)

print(f"Dataset: {input_filename}")
print("Raw CSV-feature accuracy:", scores_raw["test_accuracy"].mean())
print("POS-embedding accuracy:", scores_pos["test_accuracy"].mean())

# for fold_idx, (train_idx, test_idx) in enumerate(
#     group_kfold.split(X_scaled, y, groups)
# ):
#     train_matrices = df.iloc[train_idx]["Matrix"].unique()
#     test_matrices = df.iloc[test_idx]["Matrix"].unique()

#     print(f"\nFold {fold_idx + 1}")
#     print(f"  Train matrix count: {len(train_matrices)}")
#     print(f"  Test matrix count: {len(test_matrices)}")
#     print(f"  Train matrices: {train_matrices}")
#     print(f"  Test matrices: {test_matrices}")


# ----------------- Run detailed evaluations -----------------
evaluate_cv(X_scaled, y, "VANILLA")
if len(X_pos) > 0:
    evaluate_cv(X_pos, y_pos, "POS", groups_pos)
if len(X_nopos) > 0:
    evaluate_cv(X_nopos, y_nopos, "NOPOS", groups_nopos)
