import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Classifiers
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

# Classifiers
from sklearn.metrics import (
    classification_report,
    multilabel_confusion_matrix,
)
from sklearn.model_selection import (
    GroupKFold,
    cross_validate,
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ml_ready_dir = "../data/ml_ready/"
results_dir = "../results/"
os.makedirs(f"{results_dir}/confusion_matrices/", exist_ok=True)
os.makedirs(f"{results_dir}/feature_importance/", exist_ok=True)
output_filename = "benchmark_results_summary.csv"
output_pivot = "results_summary_pivot.csv"
output_results_csv = f"{results_dir}{output_filename}"
output_pivot_csv = f"{results_dir}{output_pivot}"

params = {
    "random_state": 42,
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_bin": 64,  # default is 255
    "max_depth": 4,
    "num_leaves": 20,
    "min_data_in_leaf": 30,  # lowered from 120
    "feature_fraction": 0.8,  # randomly select 80% of features per tree
    "bagging_fraction": 0.8,  # randomly select 80% of data per iteration
    "bagging_freq": 5,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "verbosity": 0,
    "n_jobs": -1,
}

# Define base models
base_learners = [
    ("gb", GradientBoostingClassifier(n_estimators=100, max_depth=3)),
    ("rf", RandomForestClassifier(n_estimators=100, max_depth=3)),
    (
        "xgb",
        XGBClassifier(
            n_estimators=100,
            max_depth=3,
            base_score=0.5,
        ),
    ),
    ("lgb", LGBMClassifier(n_estimators=100, max_depth=3)),
    (
        "cat",
        CatBoostClassifier(
            iterations=100, depth=3, learning_rate=0.1, verbose=0
        ),
    ),
]

# Meta-model for stacking
meta_model = LogisticRegression()

# Classifier dictionary
classifiers = {
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
        random_state=42,
        base_score=0.5,
    ),
    "LightGBM": LGBMClassifier(**params),
    "RandomForest": RandomForestClassifier(random_state=42),
    # "GradientBoosting": GradientBoostingClassifier(random_state=42),
    # "KNN": KNeighborsClassifier(n_jobs=-1),
    #    'Dummy': DummyClassifier(strategy="most_frequent"),
    # "SVM": SVC(probability=True, random_state=42),
    # "SVM_linear": SVC(kernel="linear", probability=True, random_state=42),
    # "SVM_poly": SVC(kernel="poly", probability=True, random_state=42),
    # "SVM_sigmoid": SVC(kernel="sigmoid", probability=True, random_state=42),
    #    'LogisticRegression': LogisticRegression(random_state=42),
    #    'AdaBoost': AdaBoostClassifier(random_state=42),
    # "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
    # "ExtraTrees": ExtraTreesClassifier(random_state=42),
    #    'NaiveBayes': GaussianNB(),
    #    'LDA': LinearDiscriminantAnalysis(),
    #    'QDA': QuadraticDiscriminantAnalysis(),
    "MLP": MLPClassifier(random_state=42, max_iter=1000, solver="adam"),
    "MLP_Optimized": MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=5000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=42,
    ),
    # "DecisionTree": DecisionTreeClassifier(random_state=42),
    #    'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=42),
    #    'Voting': VotingClassifier(estimators=[
    #        ('rf', RandomForestClassifier(random_state=42)),
    #        ('svm', SVC(probability=True, random_state=42)),
    #        ('lr', LogisticRegression(random_state=42))
    #    ], voting='soft'),
    #    'Stacking': StackingClassifier(estimators=[
    #        ('rf', RandomForestClassifier(random_state=42)),
    #        ('svm', SVC(probability=True, random_state=42)),
    #        ('lr', LogisticRegression(random_state=42))
    #    ], final_estimator=LogisticRegression()),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
    # "Stacking": MultiOutputClassifier(
    #     StackingClassifier(estimators=base_learners, final_estimator=meta_model)
    # ),
    # "voting_soft": MultiOutputClassifier(
    #     VotingClassifier(
    #         estimators=[
    #             ("xgb", XGBClassifier(n_estimators=100, max_depth=3)),
    #             ("lgb", LGBMClassifier(n_estimators=100, max_depth=3)),
    #             ("svm", SVC(probability=True)),
    #         ],
    #         voting="soft",
    #     )
    # ),
    #    'Ridge': RidgeClassifier(random_state=42),
    #    'GaussianProcess': GaussianProcessClassifier(kernel=RBF(), random_state=42),
    #    'PassiveAggressive': PassiveAggressiveClassifier(random_state=42),
    #    'NearestCentroid': NearestCentroid(),
}

# Classifiers
# classifiers = {
#     "XGBoost": XGBClassifier(
#         use_label_encoder=False,
#         eval_metric="logloss",
#         verbosity=0,
#         random_state=42,
#         base_score=0.5,
#     ),
#     "RandomForest": RandomForestClassifier(random_state=42),
#     "MLP": MLPClassifier(random_state=42, max_iter=1000),
# }
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


def evaluate_detailed(
    X_data, y_data, name, groups_override, dataset_name, feature_names=None
):
    et = ExtraTreesClassifier(random_state=42).fit(X_data, y_data)
    thr = np.percentile(et.feature_importances_, 0)
    idx = np.where(et.feature_importances_ >= thr)[0]
    X_sel = X_data[:, idx]

    if feature_names is not None:
        selected_feature_names = np.array(feature_names)[idx]
    else:
        selected_feature_names = [f"F{i}" for i in idx]

    results = []

    for clf_name, clf in classifiers.items():
        print(f"Model: {clf_name}, Type: {name}")
        model = MultiOutputClassifier(clf, n_jobs=-1)

        # Accumulate confusion matrices per label across folds
        all_conf_matrices = {
            label: np.zeros((2, 2), dtype=int) for label in target_columns
        }

        fold = 0
        for train_idx, test_idx in group_kfold.split(
            X_sel, y_data, groups_override
        ):
            model.fit(X_sel[train_idx], y_data[train_idx])
            y_pred = model.predict(X_sel[test_idx])
            y_true = y_data[test_idx]

            # === Per-class F1 metrics ===
            report = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
            for i, label in enumerate(report.keys()):
                if label in ["accuracy", "macro avg", "weighted avg"]:
                    continue
                results.append(
                    {
                        "dataset": dataset_name,
                        "embedding_type": name,
                        "feature_count": len(idx),
                        "classifier": clf_name,
                        "metric": f"f1_class_{label}",
                        "score": report[label]["f1-score"],
                    }
                )

            # === Accumulate confusion matrices ===
            cm = multilabel_confusion_matrix(y_true, y_pred)
            for i, (label, matrix) in enumerate(zip(target_columns, cm)):
                all_conf_matrices[label] += matrix

            fold += 1

        # === Save final confusion matrices after all folds ===
        confmat_dir = os.path.join(results_dir, "confusion_matrices")
        os.makedirs(confmat_dir, exist_ok=True)
        for label, matrix in all_conf_matrices.items():
            plt.figure(figsize=(3, 3))
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Total Confusion - {label}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            filename = f"{dataset_name}_{name}_{clf_name}_{label.replace(' ', '_')}.png"
            plt.savefig(os.path.join(confmat_dir, filename))
            plt.close()

        # === Average cross-validation scores ===
        scores = cross_validate(
            MultiOutputClassifier(clf, n_jobs=-1),
            X_sel,
            y_data,
            cv=group_kfold.split(X_sel, y_data, groups_override),
            scoring=scoring,
            n_jobs=-1,
        )
        for metric in scoring:
            mean_score = scores["test_" + metric].mean()
            results.append(
                {
                    "dataset": dataset_name,
                    "embedding_type": name,
                    "feature_count": len(idx),
                    "classifier": clf_name,
                    "metric": metric,
                    "score": mean_score,
                }
            )

        # Ensure y_data is 1D for dir_ml_worst
        if isinstance(y_data, pd.DataFrame):
            y_label = y_data["dir_ml_worst"].values
        else:
            y_label = y_data[:, 0]  # assuming dir_ml_worst is first column

        # === Feature importance visualization ===
        try:
            # Common tree-based model handler
            importances = None

            if isinstance(clf, (RandomForestClassifier, ExtraTreesClassifier)):
                clf.fit(X_sel, y_label)
                importances = np.mean(
                    [est.feature_importances_ for est in clf.estimators_],
                    axis=0,
                )

            elif clf_name.lower().startswith("lightgbm"):
                clf = LGBMClassifier(**clf.get_params())
                clf.fit(X_sel, y_label)
                importances = clf.booster_.feature_importance(
                    importance_type="gain"
                )
                importances = importances / importances.sum()

            elif clf_name.lower().startswith("xgboost"):
                clf = XGBClassifier(
                    use_label_encoder=False, eval_metric="logloss"
                )
                clf.fit(X_sel, y_label)
                importances = clf.feature_importances_
                importances = importances / importances.sum()

            # Plot if importances were computed
            if importances is not None:
                top_k = np.argsort(importances)[::-1][:20]
                top_features = np.array(selected_feature_names)[top_k]

                plt.figure(figsize=(14, 6))
                sns.barplot(x=importances[top_k], y=top_features)
                plt.title(
                    f"Feature Importances - {clf_name} - {dataset_name} ({name})"
                )
                plt.tight_layout()

                os.makedirs(f"{results_dir}/feature_importance/", exist_ok=True)
                plt.savefig(
                    f"{results_dir}/feature_importance/{dataset_name}_{name}_{clf_name}.png"
                )
                plt.close()

        except Exception as e:
            print(f"[!] Feature importance plotting failed: {e}")

        # # === Feature importance visualization ===
        # try:
        #     if isinstance(
        #         clf,
        #         (
        #             RandomForestClassifier,
        #             ExtraTreesClassifier,
        #         ),
        #     ):
        #         clf.fit(X_sel, y_data)
        #         importances = np.mean(
        #             [est.feature_importances_ for est in clf.estimators_],
        #             axis=0,
        #         )
        #         top_k = np.argsort(importances)[::-1][:20]
        #         plt.figure(figsize=(14, 6))
        #         sns.barplot(
        #             x=importances[top_k],
        #             y=np.array(selected_feature_names)[top_k],
        #         )
        #         plt.title(
        #             f"Feature Importances - {clf_name} - {dataset_name} ({name})"
        #         )
        #         plt.tight_layout()
        #         os.makedirs(f"{results_dir}/feature_importance/", exist_ok=True)
        #         plt.savefig(
        #             f"{results_dir}/feature_importance/{dataset_name}_{name}_{clf_name}.png"
        #         )
        #         plt.close()

        # elif clf_name.lower().startswith("lightgbm"):
        #     importances_list = []

        #     for i, label in enumerate(target_columns):
        #         clf_single = LGBMClassifier(
        #             random_state=42, **clf.get_params()
        #         )
        #         clf_single.fit(X_sel, y_data[:, i])
        #         importances_list.append(clf_single.feature_importances_)

        #     # Now safely average across all label-specific models
        #     importances = np.mean(importances_list, axis=0)

        #     top_k = np.argsort(importances)[::-1][:20]
        #     plt.figure(figsize=(14, 6))
        #     sns.barplot(
        #         x=importances[top_k],
        #         y=np.array(selected_feature_names)[top_k],
        #     )
        #     plt.title(
        #         f"Feature Importances - {clf_name} - {dataset_name} ({name})"
        #     )
        #     plt.tight_layout()
        #     os.makedirs(f"{results_dir}/feature_importance/", exist_ok=True)
        #     plt.savefig(
        #         f"{results_dir}/feature_importance/{dataset_name}_{name}_{clf_name}.png"
        #     )
        #     plt.close()

        # elif clf_name.lower().startswith("xgboost"):
        #     from xgboost import XGBClassifier
        #     from xgboost import plot_importance as xgb_plot_importance

        #     model = XGBClassifier()
        #     model.fit(X_sel, y_data)

        #     # Get top-k features
        #     importances = model.feature_importances_
        #     top_k = np.argsort(importances)[::-1][:20]
        #     top_features = np.array(selected_feature_names)[top_k]

        #     # Plot with custom figsize
        #     fig, ax = plt.subplots(figsize=(14, 6))
        #     xgb_plot_importance(
        #         model, max_num_features=20, height=0.5, ax=ax
        #     )
        #     ax.set_yticklabels(top_features)
        #     plt.title(f"XGBoost Feature Importance - {clf_name}")
        #     plt.tight_layout()
        #     plt.savefig(
        #         f"{results_dir}/feature_importance/{dataset_name}_{name}_{clf_name}_xgb.png"
        #     )
        #     plt.close()

        except Exception as e:
            print(f"Failed to compute feature importances for {clf_name}: {e}")

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

    # Drop non-feature columns
    # drop_columns = (
    #     [
    #         "Matrix",
    #         "Variant Name",
    #         "Group",
    #         "Kind",
    #         "Operation",
    #         "Method",
    #         "Generation Time(S)",
    #         "Match NNZ",
    #         "cosine_similarity_local",
    #     ]
    #     + target_columns
    #     + [
    #         "value_min",
    #         "value_max",
    #         "value_avg",
    #         "value_std",
    #         "row_min_min",
    #         "row_min_max",
    #         "row_min_mean",
    #         "row_min_std",
    #         "row_max_min",
    #         "row_max_max",
    #         "row_max_mean",
    #         "row_max_std",
    #         "row_mean_min",
    #         "row_mean_max",
    #         "row_mean_mean",
    #         "row_mean_std",
    #         "row_std_min",
    #         "row_std_max",
    #         "row_std_mean",
    #         "row_std_std",
    #         "row_median_min",
    #         "row_median_max",
    #         "row_median_mean",
    #         "row_median_std",
    #         "col_min_min",
    #         "col_min_max",
    #         "col_min_mean",
    #         "col_min_std",
    #         "col_max_min",
    #         "col_max_max",
    #         "col_max_mean",
    #         "col_max_std",
    #         "col_mean_min",
    #         "col_mean_max",
    #         "col_mean_mean",
    #         "col_mean_std",
    #         "col_std_min",
    #         "col_std_max",
    #         "col_std_mean",
    #         "col_std_std",
    #         "col_median_min",
    #         "col_median_max",
    #         "col_median_mean",
    #         "col_median_std",
    #         "norm_1",
    #         "norm_inf",
    #         "frobenius_norm",
    #         "estimated_condition_number",
    #         # "num_empty_rows",
    #         # "num_empty_cols",
    #     ]
    # )

    # drop_columns = (
    #     [
    #         "Matrix",
    #         "Variant Name",
    #         "Group",
    #         "Kind",
    #         "Operation",
    #         "Method",
    #         "Generation Time(S)",
    #         "Match NNZ",
    #     ]
    #     + target_columns
    #     + [
    #         "value_min",
    #         "value_max",
    #         "value_avg",
    #         "value_std",
    #         "row_min_min",
    #         "row_min_max",
    #         "row_min_mean",
    #         "row_min_std",
    #         "row_max_min",
    #         "row_max_max",
    #         "row_max_mean",
    #         "row_max_std",
    #         "row_mean_min",
    #         "row_mean_max",
    #         "row_mean_mean",
    #         "row_mean_std",
    #         "row_std_min",
    #         "row_std_max",
    #         "row_std_mean",
    #         "row_std_std",
    #         "row_median_min",
    #         "row_median_max",
    #         "row_median_mean",
    #         "row_median_std",
    #         "col_min_min",
    #         "col_min_max",
    #         "col_min_mean",
    #         "col_min_std",
    #         "col_max_min",
    #         "col_max_max",
    #         "col_max_mean",
    #         "col_max_std",
    #         "col_mean_min",
    #         "col_mean_max",
    #         "col_mean_mean",
    #         "col_mean_std",
    #         "col_std_min",
    #         "col_std_max",
    #         "col_std_mean",
    #         "col_std_std",
    #         "col_median_min",
    #         "col_median_max",
    #         "col_median_mean",
    #         "col_median_std",
    #         "avg_distance_to_diagonal",
    #         "Bandwidth",
    #         "Profile",
    #         # "avg_distance_to_diagonal / N",
    #         # "num_diagonals_with_nonzeros",
    #         # "nnz_bandwidth_std",
    #         # "nnz_diagonal",
    #         # "nnz_off_diagonal",
    #         # "num_structurally_unsymmetric_elements",
    #         "norm_1",
    #         "norm_inf",
    #         "frobenius_norm",
    #         "estimated_condition_number",
    #         # "num_empty_rows",
    #         # "num_empty_cols",
    #         # "row_sparsity_skew",
    #         # "col_sparsity_skew",
    #         # "row_nnz_entropy",
    #         # "col_nnz_entropy",
    #         # "Bandwidth / N",
    #         # "Profile / N",
    #         # "Row NNZ Median",
    #         # "Column NNZ Median",
    #         # "Row NNZ Max",
    #         # "Row NNZ Mean",
    #         # "Row NNZ STD",
    #         # "Density",
    #         # "Bandwidth STD",
    #         "cosine_similarity_local",
    #     ]
    # )

    # Drop non-feature columnsAdd commentMore actions

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
            "avg_distance_to_diagonal",
            # "avg_distance_to_diagonal / N",
            # "num_diagonals_with_nonzeros",
            # "nnz_bandwidth_std",
            # "nnz_diagonal",
            # "nnz_off_diagonal",
            # "num_structurally_unsymmetric_elements",
            "norm_1",
            "norm_inf",
            "frobenius_norm",
            "estimated_condition_number",
            "num_empty_rows",
            "num_empty_cols",
            "row_sparsity_skew",
            "col_sparsity_skew",
            # "row_nnz_entropy",
            # "col_nnz_entropy",
            # "Bandwidth / N",
            # "Profile / N",
            "Bandwidth",
            "Profile",
            # "Row NNZ Median",
            # "Column NNZ Median",
            # "Row NNZ Max",
            # "Row NNZ Mean",
            "Row NNZ STD",
            # "Density",
            "Bandwidth STD",
            "cosine_similarity_local",
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

    print(f"Selected {len(X.columns.tolist())} Features: {X.columns.tolist()}")

    scaler_vanilla = StandardScaler()
    X_scaled = scaler_vanilla.fit_transform(X)

    groups = [matrix_to_base[name] for name in df["Matrix"]]

    # # Run cross_validate for raw CSV features and POS embeddings (if available)
    # try:
    #     scores_raw = cross_validate(
    #         XGBClassifier(
    #             use_label_encoder=False,
    #             eval_metric="logloss",
    #             verbosity=0,
    #             random_state=42,
    #             base_score=0.5,
    #         ),
    #         X_scaled,
    #         y,
    #         cv=group_kfold.split(X_scaled, y, groups),
    #         scoring=["accuracy"],
    #         n_jobs=-1,
    #     )
    #     raw_acc = scores_raw["test_accuracy"].mean()
    # except Exception as e:
    #     print(f"Failed raw CSV evaluation for {input_filename}: {e}")
    #     raw_acc = None

    # try:
    #     if len(X_pos) > 0:
    #         scores_pos = cross_validate(
    #             XGBClassifier(
    #                 use_label_encoder=False,
    #                 eval_metric="logloss",
    #                 verbosity=0,
    #                 random_state=42,
    #                 base_score=0.5,
    #             ),
    #             X_pos,
    #             y_pos,
    #             cv=group_kfold.split(X_pos, y_pos, groups_pos),
    #             scoring=["accuracy"],
    #             n_jobs=-1,
    #         )
    #         pos_acc = scores_pos["test_accuracy"].mean()
    #     else:
    #         pos_acc = None
    # except Exception as e:
    #     print(f"Failed POS embedding evaluation for {input_filename}: {e}")
    #     pos_acc = None

    # # Collect summary results for raw and POS (no NOPOS here for brevity)
    # all_results.append(
    #     {
    #         "dataset": input_filename,
    #         "embedding_type": "VANILLA",
    #         "classifier": "XGBoost",
    #         "metric": "accuracy",
    #         "score": raw_acc,
    #     }
    # )
    # if pos_acc is not None:
    #     all_results.append(
    #         {
    #             "dataset": input_filename,
    #             "embedding_type": "POS",
    #             "classifier": "XGBoost",
    #             "metric": "accuracy",
    #             "score": pos_acc,
    #         }
    #     )

    # Run detailed evaluation with other classifiers
    all_results.extend(
        evaluate_detailed(
            X_scaled,
            y,
            "VANILLA",
            groups,
            input_filename,
            feature_names=X.columns.tolist(),
        )
    )
    if len(X_pos) > 0:
        all_results.extend(
            evaluate_detailed(X_pos, y_pos, "POS", groups_pos, input_filename)
        )
    if len(X_nopos) > 0:
        all_results.extend(
            evaluate_detailed(
                X_nopos, y_nopos, "NOPOS", groups_nopos, input_filename
            )
        )

    common_matrices = sorted(set(matrix_names_pos) & set(df["Matrix"]))

    if common_matrices:
        df_combined = df.set_index("Matrix").loc[common_matrices]
        X_vanilla_combined = scaler_vanilla.transform(
            df_combined.drop(columns=drop_columns, errors="ignore")
        )
        X_pos_combined = scaler_pos.transform(
            X_pos[[matrix_names_pos.index(m) for m in common_matrices]]
        )

        X_concat = np.hstack([X_vanilla_combined, X_pos_combined])
        y_concat = df_combined[target_columns].values
        groups_concat = [matrix_to_base[m] for m in common_matrices]

        # try:
        #     scores_combined = cross_validate(
        #         XGBClassifier(
        #             use_label_encoder=False,
        #             eval_metric="logloss",
        #             verbosity=0,
        #             random_state=42,
        #             base_score=0.5,
        #         ),
        #         X_concat,
        #         y_concat,
        #         cv=group_kfold.split(X_concat, y_concat, groups_concat),
        #         scoring=["accuracy"],
        #         n_jobs=-1,
        #     )
        #     combined_acc = scores_combined["test_accuracy"].mean()
        # except Exception as e:
        #     print(f"Failed COMBINED evaluation for {input_filename}: {e}")
        #     combined_acc = None

        # if combined_acc is not None:
        #     all_results.append(
        #         {
        #             "dataset": input_filename,
        #             "embedding_type": "VANILLA+POS",
        #             "classifier": "XGBoost",
        #             "metric": "accuracy",
        #             "score": combined_acc,
        #         }
        #     )

        # Run with all classifiers
        all_results.extend(
            evaluate_detailed(
                X_concat, y_concat, "VANILLA+POS", groups_concat, input_filename
            )
        )

    common_matrices_nopos = sorted(set(matrix_names_nopos) & set(df["Matrix"]))

    if common_matrices_nopos:
        df_combined = df.set_index("Matrix").loc[common_matrices_nopos]
        X_vanilla_combined = scaler_vanilla.transform(
            df_combined.drop(columns=drop_columns, errors="ignore")
        )
        X_nopos_combined = scaler_nopos.transform(
            X_pos[[matrix_names_nopos.index(m) for m in common_matrices_nopos]]
        )

        X_concat = np.hstack([X_vanilla_combined, X_nopos_combined])
        y_concat = df_combined[target_columns].values
        groups_concat = [matrix_to_base[m] for m in common_matrices_nopos]

        # try:
        #     scores_combined = cross_validate(
        #         XGBClassifier(
        #             use_label_encoder=False,
        #             eval_metric="logloss",
        #             verbosity=0,
        #             random_state=42,
        #             base_score=0.5,
        #         ),
        #         X_concat,
        #         y_concat,
        #         cv=group_kfold.split(X_concat, y_concat, groups_concat),
        #         scoring=["accuracy"],
        #         n_jobs=-1,
        #     )
        #     combined_acc = scores_combined["test_accuracy"].mean()
        # except Exception as e:
        #     print(f"Failed COMBINED evaluation for {input_filename}: {e}")
        #     combined_acc = None

        # if combined_acc is not None:
        #     all_results.append(
        #         {
        #             "dataset": input_filename,
        #             "embedding_type": "VANILLA+NOPOS",
        #             "classifier": "XGBoost",
        #             "metric": "accuracy",
        #             "score": combined_acc,
        #         }
        #     )

        # Run with all classifiers
        all_results.extend(
            evaluate_detailed(
                X_concat,
                y_concat,
                "VANILLA+NOPOS",
                groups_concat,
                input_filename,
            )
        )


# Save results summary to CSV
df_results = pd.DataFrame(all_results)

# Create a pivot table
pivot_df = df_results.pivot_table(
    index=["dataset", "embedding_type", "classifier", "feature_count"],
    columns="metric",
    values="score",
).reset_index()

# Optional: sort by dataset and classifier for better readability
pivot_df = pivot_df.sort_values(
    by=["dataset", "embedding_type", "classifier", "feature_count"]
)

# Print the pivot table
print(pivot_df)

# Save to CSV for easier viewing if needed
pivot_df.to_csv(output_pivot_csv, index=False)
df_results.to_csv(output_results_csv, index=False)

print(f"\nAll results saved to {output_results_csv} and {output_pivot_csv}")
