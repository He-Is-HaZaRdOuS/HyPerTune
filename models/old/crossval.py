import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    multilabel_confusion_matrix,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, plot_importance

# Uyarıları kapat
warnings.filterwarnings("ignore")

# Dosya yolu ve veri yükleme
ml_ready_dir = "../../data/ml_ready/"
file_name = "DISCRETIZED_matrix_data_dir_rec_2class.csv"
input_csv = f"{ml_ready_dir}{file_name}"
df = pd.read_csv(input_csv)

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
        "avg_distance_to_diagonal",
        "avg_distance_to_diagonal / N",
        "num_diagonals_with_nonzeros",
        "nnz_bandwidth_std",
        "nnz_diagonal",
        "nnz_off_diagonal",
        "num_structurally_unsymmetric_elements",
        "norm_1",
        "norm_inf",
        "frobenius_norm",
        "estimated_condition_number",
        "num_empty_rows",
        "num_empty_cols",
        "row_sparsity_skew",
        "col_sparsity_skew",
        "row_nnz_entropy",
        "col_nnz_entropy",
        "Bandwidth / N",
        "Profile / N",
        "Row NNZ Median",
        "Column NNZ Median",
        "Row NNZ Max",
        "Row NNZ Mean",
        "Row NNZ STD",
        "Density",
        "Bandwidth STD",
        "cosine_similarity_local",
    ]
)

X_all = df.drop(columns=drop_columns)
y = df[target_columns]

# Print the features being used
print("Features being used in the model:")
print(X_all.columns.tolist())
print(f"\nTotal number of features: {len(X_all.columns)}")

# Normalize et
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

classifiers = {
    "XGBoost": xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
        random_state=42,
    ),
}

parallel_values = [22]

# Count the number of 1.0 and 0.0 in each target column
print("\nCount of 1.0 and 0.0 in Target Columns:")
for target in target_columns:
    count_1 = (y[target] == 1.0).sum()
    count_0 = (y[target] == 0.0).sum()
    print(f"{target}: 1.0 -> {count_1}, 0.0 -> {count_0}")

# Initialize the model
base_model = XGBClassifier(random_state=42)
multi_xgb_model = MultiOutputClassifier(base_model)

# Set up cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)


# Since accuracy_score doesn't directly support multioutput, we need to wrap it
def multioutput_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


# Perform cross-validation
scoring = {
    "accuracy": make_scorer(multioutput_accuracy),
    # You can add other metrics here if needed
}

cv_results = cross_validate(
    multi_xgb_model,
    X_all_scaled,
    y,
    cv=kfold,
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1,  # Use all available cores
)

# Print cross-validation results
print("\nCross-Validation Results:")
print(f"Mean Train Accuracy: {np.mean(cv_results['train_accuracy']):.4f}")
print(f"Std Train Accuracy: {np.std(cv_results['train_accuracy']):.4f}")
print(f"Mean Test Accuracy: {np.mean(cv_results['test_accuracy']):.4f}")
print(f"Std Test Accuracy: {np.std(cv_results['test_accuracy']):.4f}")

# Now train on the full training set and evaluate on a hold-out test set
X_train, X_test, y_train, y_test = train_test_split(
    X_all_scaled, y, test_size=0.2, random_state=42
)

# Train the MultiOutputClassifier with XGBoost
multi_xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred = multi_xgb_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nHold-out Test Set Accuracy: {accuracy:.4f}")

# Generate a multilabel confusion matrix
confusion_matrices = multilabel_confusion_matrix(y_test, y_pred)

# Print the confusion matrix for each label
print("\nConfusion Matrices for Each Label:")
for i, target in enumerate(target_columns):
    print(f"\nConfusion Matrix for {target}:")
    print(confusion_matrices[i])

# Combine all confusion matrices into a single matrix (optional)
combined_confusion_matrix = sum(confusion_matrices)
print("\nCombined Confusion Matrix:")
print(combined_confusion_matrix)

# Feature Importance Analysis
print("\nFeature Importance Analysis:")
feature_names = X_all.columns.tolist()

# For MultiOutputClassifier, we need to check each estimator
for i, target in enumerate(target_columns):
    print(f"\nFeature Importance for target: {target}")
    estimator = multi_xgb_model.estimators_[i]

    # Get feature importance
    importance = estimator.feature_importances_

    # Create a DataFrame for better visualization
    feature_importance = pd.DataFrame(
        {"Feature": feature_names, "Importance": importance}
    ).sort_values("Importance", ascending=False)

    print(feature_importance)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importance for {target}")
    plt.barh(feature_importance["Feature"], feature_importance["Importance"])
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

# Alternatively, you can use XGBoost's built-in plot_importance
for i, target in enumerate(target_columns):
    print(f"\nXGBoost Feature Importance Plot for {target}:")
    plot_importance(multi_xgb_model.estimators_[i], max_num_features=20)
    plt.title(f"XGBoost Feature Importance for {target}")
    plt.tight_layout()
    plt.show()
