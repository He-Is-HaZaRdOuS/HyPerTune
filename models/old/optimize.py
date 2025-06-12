import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
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
#    "Row Max", "Row Mean", "Row STD", "Density"
]

# Features and target data
X = df.drop(columns=drop_columns)
y = df[target_columns]

# Split the data into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (if not already done)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Engineering: Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Base models for stacking
base_learners = [
    ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=3)),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=3)),
    ('hist_gb', HistGradientBoostingClassifier(max_iter=100)),
    ('cat', CatBoostClassifier(iterations=100, depth=3, learning_rate=0.1, verbose=0))
]

# Meta-model for stacking (using a GradientBoostingClassifier)
final_estimator = LogisticRegression()

# Create StackingClassifier
stacking_classifier = StackingClassifier(estimators=base_learners, final_estimator=final_estimator)

# Wrap stacking model with MultiOutputClassifier for multi-target output
multi_output_model = MultiOutputClassifier(stacking_classifier)

# Create a pipeline that first applies polynomial feature expansion, then the stacking classifier
pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    multi_output_model
)

# Hyperparameter tuning for final estimator
param_grid = {
    'multioutputclassifier__estimator__final_estimator': [LogisticRegression(), GradientBoostingClassifier()],
    'multioutputclassifier__estimator__n_jobs': [-1]  # Use all cores for faster computation
}

# Initialize GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1, n_jobs=-1)

# Fit the grid search model
grid_search.fit(X_train, y_train)

# Get the best model and print the results
best_model = grid_search.best_estimator_
print("Best parameters found: ", grid_search.best_params_)

# Test the model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Cross-validation on the final model
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
print(f"Cross-validation accuracy: {cv_scores.mean():.4f}")

# Output the final results
print("Model evaluation complete.")

