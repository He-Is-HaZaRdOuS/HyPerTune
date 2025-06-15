import gc
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Classifiers
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Load your dataset
ml_ready_dir = "../data/ml_ready/"
file_name = "pruned_multilabel_DISCRETIZED_matrix_data_dir_rec_2class.csv"
input_csv = f"{ml_ready_dir}{file_name}"
df = pd.read_csv(input_csv, low_memory=False)
df = df.apply(pd.to_numeric, errors="coerce")  # Non-numeric data becomes NaN
df = df.fillna(0)  # Replace NaN with 0, or another meaningful default

# Automatically detect target binary label columns (all with only 0/1 values)
target_columns = [
    col
    for col in df.columns
    if set(df[col].unique()) <= {0, 1} and col.startswith(("dir", "rec"))
]

print("Detected target columns:", target_columns)

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
        # "num_empty_rows",
        # "num_empty_cols",
    ]
)

X = df.drop(columns=drop_columns)
y = df[target_columns]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

params = {
    "random_state": 42,
    "n_estimators": 300,  # max number of trees (safe limit)
    "learning_rate": 0.1,  # standard rate, not too slow
    "max_depth": 6,  # limit tree depth (less overfitting, faster)
    "num_leaves": 31,  # default is fine
    "n_jobs": -1,  # use all cores
    "verbosity": 0,  # show some info but not flood logs
}

# Define base models
base_learners = [
    ("gb", GradientBoostingClassifier(n_estimators=100, max_depth=3)),
    ("rf", RandomForestClassifier(n_estimators=100, max_depth=3)),
    ("xgb", XGBClassifier(n_estimators=100, max_depth=3)),
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
    ),
    "LightGBM": LGBMClassifier(**params),
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_jobs=-1),
    #    'Dummy': DummyClassifier(strategy="most_frequent"),
    "SVM": SVC(probability=True, random_state=42),
    "SVM_linear": SVC(kernel="linear", probability=True, random_state=42),
    "SVM_poly": SVC(kernel="poly", probability=True, random_state=42),
    "SVM_sigmoid": SVC(kernel="sigmoid", probability=True, random_state=42),
    #    'LogisticRegression': LogisticRegression(random_state=42),
    #    'AdaBoost': AdaBoostClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
    "ExtraTrees": ExtraTreesClassifier(random_state=42),
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
    "DecisionTree": DecisionTreeClassifier(random_state=42),
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
    "Stacking": MultiOutputClassifier(
        StackingClassifier(estimators=base_learners, final_estimator=meta_model)
    ),
    "voting_soft": MultiOutputClassifier(
        VotingClassifier(
            estimators=[
                ("xgb", XGBClassifier(n_estimators=100, max_depth=3)),
                ("lgb", LGBMClassifier(n_estimators=100, max_depth=3)),
                ("svm", SVC(probability=True)),
            ],
            voting="soft",
        )
    ),
    #    'Ridge': RidgeClassifier(random_state=42),
    #    'GaussianProcess': GaussianProcessClassifier(kernel=RBF(), random_state=42),
    #    'PassiveAggressive': PassiveAggressiveClassifier(random_state=42),
    #    'NearestCentroid': NearestCentroid(),
}

# Scorers
scoring = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
    "roc_auc_ovr": "roc_auc_ovr",
}

# Directory to save results
output_dir = "../results/model_evaluation_results"
os.makedirs(output_dir, exist_ok=True)

# Save results
results = []

# K-Fold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

imp_threshold = 20
print(
    f"Performing feature sampling to keep top {100 - imp_threshold}% of original features using ExtraTreesClassifier"
)
model = ExtraTreesClassifier(random_state=42).fit(X_scaled, y)
importances = model.feature_importances_
threshold = np.percentile(importances, imp_threshold)  # keep top 80% features
important_indices = np.where(importances >= threshold)[0]
X_scaled = X_scaled[:, important_indices]

# Benchmark loop
for name, model in classifiers.items():
    print(f"Training and evaluating {name}...")
    try:
        multi_model = MultiOutputClassifier(model, n_jobs=-1)
        scores = cross_validate(
            multi_model, X_scaled, y, cv=kfold, scoring=scoring, n_jobs=-1
        )

        result = {
            "Classifier": name,
            "Accuracy Mean": np.mean(scores["test_accuracy"]),
            "Precision Mean": np.mean(scores["test_precision_macro"]),
            "Recall Mean": np.mean(scores["test_recall_macro"]),
            "F1 Mean": np.mean(scores["test_f1_macro"]),
            "ROC AUC Mean": np.mean(scores["test_roc_auc_ovr"]),
            "Train Time (s)": np.sum(scores["fit_time"]),
        }
        results.append(result)

        # Train on the full training set
        multi_model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = multi_model.predict(X_test)

        # Confusion Matrix
        confusion_matrices = multilabel_confusion_matrix(y_test, y_pred)

        # Plot all confusion matrices in one grid
        print("\nConfusion Matrices for All Labels:")
        fig, axes = plt.subplots(
            2, 3, figsize=(15, 10)
        )  # adjust grid size to match your labels
        axes = axes.flatten()

        for i, target in enumerate(target_columns):
            cm = confusion_matrices[i]
            ax = axes[i]
            ax.matshow(cm, cmap="Blues")
            for (j, k), val in np.ndenumerate(cm):
                ax.text(k, j, f"{val}", ha="center", va="center", color="red")
            ax.set_title(f"{target}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

        # Hide unused axes if any
        for j in range(len(target_columns), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{name}_combined_confusion_matrices.png")
        plt.close(fig)

        # Feature Importance (only for tree-based models)
        if name in [
            "XGBoost",
            "LightGBM",
            "RandomForest",
            "ExtraTrees",
            "CatBoost",
        ]:
            for i, target in enumerate(target_columns):
                estimator = multi_model.estimators_[i]
                importance = estimator.feature_importances_

                # Create a unified DataFrame to store all feature importances for all labels
                all_feature_importance_df = pd.DataFrame({"Feature": X.columns})

                for i, target in enumerate(target_columns):
                    estimator = multi_model.estimators_[i]
                    importance = estimator.feature_importances_

                    # Add the importance for this label as a new column
                    all_feature_importance_df[f"Importance_{target}"] = (
                        importance
                    )

                # Save the combined feature importance DataFrame as a CSV
                all_feature_importance_df.to_csv(
                    f"{output_dir}/{name}_combined_feature_importances.csv",
                    index=False,
                )

                # Combined feature importance plot for all labels
                num_labels = len(target_columns)
                cols = 3
                rows = (num_labels + cols - 1) // cols

                fig, axes = plt.subplots(
                    rows, cols, figsize=(cols * 5, rows * 4)
                )
                axes = axes.flatten()

                for i, target in enumerate(target_columns):
                    estimator = multi_model.estimators_[i]
                    importance = estimator.feature_importances_

                    feature_importance_df = pd.DataFrame(
                        {"Feature": X.columns, "Importance": importance}
                    ).sort_values("Importance", ascending=False)

                    ax = axes[i]
                    ax.barh(
                        feature_importance_df["Feature"][:10][
                            ::-1
                        ],  # top 10 features
                        feature_importance_df["Importance"][:10][::-1],
                    )
                    ax.set_title(f"{target}")
                    ax.set_xlabel("Importance")

                # Hide unused axes
                for j in range(num_labels, len(axes)):
                    axes[j].axis("off")

                plt.tight_layout()
                plt.savefig(
                    f"{output_dir}/{name}_combined_feature_importance.png"
                )
                plt.close(fig)

                # Create a combined DataFrame where each row corresponds to a feature
                stacked_importance_df = all_feature_importance_df.set_index(
                    "Feature"
                )

                # Plot the stacked bar plot
                stacked_importance_df.plot(
                    kind="barh", stacked=True, figsize=(10, 8)
                )
                plt.xlabel("Importance")
                plt.title("Feature Importance Across Labels (Stacked)")
                plt.tight_layout()
                plt.savefig(
                    f"{output_dir}/{name}_stacked_feature_importance.png"
                )
                plt.close()

        #                # Prepare data for radar chart
        #                labels = all_feature_importance_df['Feature']
        #                num_labels = len(target_columns)
        #                angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        #
        #                # Create radar chart for each label
        #                for i, target in enumerate(target_columns):
        #                    importance = all_feature_importance_df[f'Importance_{target}']
        #                    values = importance.tolist()
        #
        #                    # Make the plot circular
        #                    values += values[:1]
        #                    angles += angles[:1]
        #
        #                    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        #                    ax.plot(angles, values, label=target, linewidth=2, linestyle='solid')
        #                    ax.fill(angles, values, alpha=0.25)
        #                    ax.set_yticklabels([])  # hide radial ticks
        #                    ax.set_xticks(angles[:-1])
        #                    ax.set_xticklabels(labels, rotation=90)
        #                    ax.set_title(f'Feature Importance for {target}')
        #
        #                    plt.tight_layout()
        #                    plt.savefig(f"{output_dir}/{name}_radar_feature_importance_{target}.png")
        #                    plt.close(fig)
        #

        gc.collect()

    except Exception as e:
        print(f"Failed on {name}: {e}")
        continue

# Results to DataFrame
results_df = pd.DataFrame(results)
results_df.sort_values(by="F1 Mean", ascending=False, inplace=True)

# Save results to CSV
results_df.to_csv(f"{output_dir}/benchmark_results.csv", index=False)

print("\n=== Benchmark Finished ===\n")
print(results_df)
