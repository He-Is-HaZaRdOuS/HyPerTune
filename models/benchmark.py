import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ml_ready_dir = "../data/ml_ready/"
results_dir = "../data/results/"
output_filename = "benchmark_results_summary.csv"
output_pivot = "results_summary_pivot.csv"
output_results_csv = f"{results_dir}{output_filename}"
output_pivot_csv = f"{results_dir}{output_pivot}"

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

group_kfold = GroupKFold(n_splits=5)


def base_matrix_name(name):
    match = re.match(r"(.+?)_expansion_\d+", name)
    return match.group(1) if match else name


def infer_class_count(filename):
    basename = os.path.basename(filename).lower()
    match = re.search(r"(\d)[-_]?class", basename)
    if match:
        return int(match.group(1))
    if "2class" in basename or "binary" in basename:
        return 2
    if "6class" in basename or "multiclass" in basename:
        return 6
    return None


def load_embeddings(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    names = [
        n.decode("utf-8") if isinstance(n, bytes) else n
        for n in data["matrix_names"]
    ]
    X = data["embeddings"]
    return names, X


def evaluate_cv(X_data, y_data, name, groups_override, dataset_name):
    et = ExtraTreesClassifier(random_state=42).fit(X_data, y_data)
    thr = np.percentile(et.feature_importances_, 30)
    idx = np.where(et.feature_importances_ >= thr)[0]
    X_sel = X_data[:, idx]

    results = []
    for clf_name, clf in classifiers.items():
        model = MultiOutputClassifier(clf, n_jobs=-1)
        scores = cross_validate(
            model,
            X_sel,
            y_data,
            cv=group_kfold.split(X_data, y_data, groups_override),
            scoring=scoring,
            n_jobs=-1,
        )
        for metric in scoring:
            mean_score = scores["test_" + metric].mean()
            results.append(
                {
                    "dataset": dataset_name,
                    "embedding_type": name,
                    "classifier": clf_name,
                    "metric": metric,
                    "score": mean_score,
                }
            )
    return results


all_results = []

for input_filename in os.listdir(ml_ready_dir):
    if not input_filename.endswith(".csv"):
        continue

    print(f"\n=== Processing {input_filename} ===")
    input_csv = os.path.join(ml_ready_dir, input_filename)
    classes = infer_class_count(input_filename)
    if classes not in [2, 6]:
        print(f"Skipping {input_filename}: Unsupported number of classes")
        continue

    embeddings_subdir = (
        "2class_bitiledlstm" if classes == 2 else "6class_bitiledlstm"
    )
    embeddings_dir = f"../data/dl_features/{embeddings_subdir}"

    df = pd.read_csv(input_csv)

    # Identify binary target columns
    target_columns = [
        col
        for col in df.columns
        if set(df[col].dropna().unique()) <= {0, 1}
        and (col.startswith("dir") or col.startswith("rec"))
    ]

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

    matrix_to_base = {m: base_matrix_name(m) for m in df["Matrix"]}
    base_to_all = defaultdict(list)
    for m, b in matrix_to_base.items():
        base_to_all[b].append(m)

    # Load embeddings
    try:
        matrix_names_pos, X_pos = load_embeddings(
            os.path.join(embeddings_dir, "data_True.npz")
        )
        matrix_names_nopos, X_nopos = load_embeddings(
            os.path.join(embeddings_dir, "data_False.npz")
        )
    except Exception as e:
        print(f"Warning: Failed to load embeddings for {input_filename}: {e}")
        matrix_names_pos, X_pos = [], np.empty((0, 0))
        matrix_names_nopos, X_nopos = [], np.empty((0, 0))

    valid_set = set(df["Matrix"])
    matrix_names_pos = [m for m in matrix_names_pos if m in valid_set]
    matrix_names_nopos = [m for m in matrix_names_nopos if m in valid_set]

    df_pos = df.set_index("Matrix").loc[matrix_names_pos]
    df_nopos = df.set_index("Matrix").loc[matrix_names_nopos]

    y_pos = (
        df_pos[target_columns].values
        if len(matrix_names_pos) > 0
        else np.empty((0, len(target_columns)))
    )
    y_nopos = (
        df_nopos[target_columns].values
        if len(matrix_names_nopos) > 0
        else np.empty((0, len(target_columns)))
    )

    scaler_pos = StandardScaler()
    if len(X_pos) > 0:
        X_pos = scaler_pos.fit_transform(X_pos[: len(matrix_names_pos)])
    else:
        X_pos = np.empty((0, 0))

    scaler_nopos = StandardScaler()
    if len(X_nopos) > 0:
        X_nopos = scaler_nopos.fit_transform(X_nopos[: len(matrix_names_nopos)])
    else:
        X_nopos = np.empty((0, 0))

    groups_pos = [matrix_to_base[m] for m in matrix_names_pos]
    groups_nopos = [matrix_to_base[m] for m in matrix_names_nopos]

    # Prepare vanilla CSV features
    X = df.drop(columns=drop_columns, errors="ignore")
    y = df[target_columns].values

    scaler_vanilla = StandardScaler()
    X_scaled = scaler_vanilla.fit_transform(X)

    groups = [matrix_to_base[name] for name in df["Matrix"]]

    # Run cross_validate for raw CSV features and POS embeddings (if available)
    try:
        scores_raw = cross_validate(
            XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
                random_state=42,
            ),
            X_scaled,
            y,
            cv=group_kfold.split(X_scaled, y, groups),
            scoring=["accuracy"],
            n_jobs=-1,
        )
        raw_acc = scores_raw["test_accuracy"].mean()
    except Exception as e:
        print(f"Failed raw CSV evaluation for {input_filename}: {e}")
        raw_acc = None

    try:
        if len(X_pos) > 0:
            scores_pos = cross_validate(
                XGBClassifier(
                    use_label_encoder=False,
                    eval_metric="logloss",
                    verbosity=0,
                    random_state=42,
                ),
                X_pos,
                y_pos,
                cv=group_kfold.split(X_pos, y_pos, groups_pos),
                scoring=["accuracy"],
                n_jobs=-1,
            )
            pos_acc = scores_pos["test_accuracy"].mean()
        else:
            pos_acc = None
    except Exception as e:
        print(f"Failed POS embedding evaluation for {input_filename}: {e}")
        pos_acc = None

    # Collect summary results for raw and POS (no NOPOS here for brevity)
    all_results.append(
        {
            "dataset": input_filename,
            "embedding_type": "VANILLA",
            "classifier": "XGBoost",
            "metric": "accuracy",
            "score": raw_acc,
        }
    )
    if pos_acc is not None:
        all_results.append(
            {
                "dataset": input_filename,
                "embedding_type": "POS",
                "classifier": "XGBoost",
                "metric": "accuracy",
                "score": pos_acc,
            }
        )

    # Run detailed evaluation with other classifiers
    all_results.extend(
        evaluate_cv(X_scaled, y, "VANILLA", groups, input_filename)
    )
    if len(X_pos) > 0:
        all_results.extend(
            evaluate_cv(X_pos, y_pos, "POS", groups_pos, input_filename)
        )
    if len(X_nopos) > 0:
        all_results.extend(
            evaluate_cv(X_nopos, y_nopos, "NOPOS", groups_nopos, input_filename)
        )


# Save results summary to CSV
df_results = pd.DataFrame(all_results)

# Create a pivot table
pivot_df = df_results.pivot_table(
    index=["dataset", "embedding_type", "classifier"],
    columns="metric",
    values="score",
).reset_index()

# Optional: sort by dataset and classifier for better readability
pivot_df = pivot_df.sort_values(by=["dataset", "embedding_type", "classifier"])

# Print the pivot table
print(pivot_df)

# Save to CSV for easier viewing if needed
pivot_df.to_csv(output_pivot_csv, index=False)
df_results.to_csv(output_results_csv, index=False)

print(f"\nAll results saved to {output_results_csv} and {output_pivot_csv}")
