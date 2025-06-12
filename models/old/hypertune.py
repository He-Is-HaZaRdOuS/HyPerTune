import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputClassifier

# Load the dataset
file_path = 'dir_rec_2class_synthetic_reduced_DISSSS.csv'
df = pd.read_csv(file_path)

# Define target columns
target_columns = ["dir_ml_worst", "rec_lazy_worst"]

# Columns to drop
drop_columns = [
    "Matrix", "Group", 
    "Kind",
    "dir_ml_worst", "rec_lazy_worst",
    "BandwidthNormalized", "ProfileNormalized",
]

# Features and targets
X = df.drop(columns=drop_columns)
y = df[target_columns]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial feature expansion
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# Base models for stacking
base_learners = [
    ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=3)),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=3)),
    ('hist_gb', HistGradientBoostingClassifier(max_iter=100)),
    ('cat', CatBoostClassifier(iterations=100, depth=3, learning_rate=0.1, verbose=0))
]

# Meta-model
final_estimator = LogisticRegression()

# Stacking classifier
stacking_classifier = StackingClassifier(
    estimators=base_learners,
    final_estimator=final_estimator,
    n_jobs=-1
)

# Multi-output wrapper
multi_output_model = MultiOutputClassifier(stacking_classifier, n_jobs=-1)

# Full pipeline
pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    multi_output_model
)

# Train
print("Training the model...")
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
auc = roc_auc_score(y_test, y_pred, average='macro')
cm = confusion_matrix(y_test.values.argmax(axis=1), y_pred.argmax(axis=1))

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1 Score (macro): {f1:.4f}")
print(f"AUC (macro): {auc:.4f}")
print("Confusion Matrix:")
print(cm)

# Save evaluation metrics
with open('metrics.txt', 'w') as f:
    f.write(f"Test Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision (macro): {precision:.4f}\n")
    f.write(f"Recall (macro): {recall:.4f}\n")
    f.write(f"F1 Score (macro): {f1:.4f}\n")
    f.write(f"AUC (macro): {auc:.4f}\n")
    f.write(f"Confusion Matrix:\n{cm}\n")

# Cross-validation
print("Running cross-validation...")
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, n_jobs=-1)
print(f"Cross-validation accuracy: {cv_scores.mean():.4f}")

with open('cv_scores.txt', 'w') as f:
    f.write(f"Cross-validation accuracy: {cv_scores.mean():.4f}\n")
    f.write(f"All scores: {cv_scores}\n")

print("\nTraining standalone Random Forest and HistGradientBoosting for feature importances...")

# Train Random Forests separately per output
rf_importances_total = np.zeros(X.shape[1])
hist_gb_importances_total = np.zeros(X.shape[1])

for i in range(y_train.shape[1]):
    # Train RF
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train.iloc[:, i])
    rf_importances_total += rf.feature_importances_

    # Train HistGB
    hist_gb = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    hist_gb.fit(X_train, y_train.iloc[:, i])
    hist_gb_importances_total += hist_gb.feature_importances_

# Average importances
rf_importances_avg = rf_importances_total / y_train.shape[1]
hist_gb_importances_avg = hist_gb_importances_total / y_train.shape[1]

# Sort features by importance
rf_indices = np.argsort(rf_importances_avg)[::-1]
hist_gb_indices = np.argsort(hist_gb_importances_avg)[::-1]

# Plot RF feature importance
plt.figure(figsize=(12, 8))
plt.title("Random Forest Feature Importances (averaged across outputs)")
plt.bar(range(X.shape[1]), rf_importances_avg[rf_indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[rf_indices], rotation=90)
plt.tight_layout()
plt.savefig("rf_feature_importances.png")
plt.close()

# Plot HistGB feature importance
plt.figure(figsize=(12, 8))
plt.title("HistGradientBoosting Feature Importances (averaged across outputs)")
plt.bar(range(X.shape[1]), hist_gb_importances_avg[hist_gb_indices], align="center", color="green")
plt.xticks(range(X.shape[1]), X.columns[hist_gb_indices], rotation=90)
plt.tight_layout()
plt.savefig("histgb_feature_importances.png")
plt.close()

# Also print top 10 features
print("\nTop 10 Random Forest Important Features (averaged):")
for idx in rf_indices[:10]:
    print(f"{X.columns[idx]}: {rf_importances_avg[idx]:.4f}")

print("\nTop 10 HistGradientBoosting Important Features (averaged):")
for idx in hist_gb_indices[:10]:
    print(f"{X.columns[idx]}: {hist_gb_importances_avg[idx]:.4f}")

print("\nFeature importance plots saved: 'rf_feature_importances.png' and 'histgb_feature_importances.png'.")


