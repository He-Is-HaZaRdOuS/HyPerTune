from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

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
    "Row Max", "Row Mean", "Row STD", "Density"
]

# Features and target data
X = df.drop(columns=drop_columns)
y = df[target_columns]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define base models
base_learners = [
    ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=3)),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=3)),
    ('xgb', XGBClassifier(n_estimators=100, max_depth=3)),
    ('lgb', LGBMClassifier(n_estimators=100, max_depth=3)),
    ('cat', CatBoostClassifier(iterations=100, depth=3, learning_rate=0.1, verbose=0))
]

# Meta-model for stacking
meta_model = LogisticRegression()

# Wrap the stacking model with MultiOutputClassifier
stacking_model = MultiOutputClassifier(
    StackingClassifier(estimators=base_learners, final_estimator=meta_model)
)

# Voting classifier examples
voting_models = [
    # Hard Voting with XGBoost, LightGBM, RandomForest
    ('voting_hard', MultiOutputClassifier(VotingClassifier(estimators=[('xgb', XGBClassifier(n_estimators=100, max_depth=3)),
                                                ('lgb', LGBMClassifier(n_estimators=100, max_depth=3)),
                                                ('rf', RandomForestClassifier(n_estimators=100, max_depth=3))],
                                    voting='hard'))),
    
    # Soft Voting with XGBoost, LightGBM, SVM
    ('voting_soft', MultiOutputClassifier(VotingClassifier(estimators=[('xgb', XGBClassifier(n_estimators=100, max_depth=3)),
                                                ('lgb', LGBMClassifier(n_estimators=100, max_depth=3)),
                                                ('svm', SVC(probability=True))],
                                    voting='soft'))),
    
    # Hard Voting with ExtraTrees, RandomForest, CatBoost
    ('voting_et_rf_cat', MultiOutputClassifier(VotingClassifier(estimators=[('et', ExtraTreesClassifier(n_estimators=100, max_depth=3)),
                                                     ('rf', RandomForestClassifier(n_estimators=100, max_depth=3)),
                                                     ('cat', CatBoostClassifier(iterations=100, depth=3, learning_rate=0.1, verbose=0))],
                                    voting='hard'))),
    
    # Soft Voting with LogisticRegression, KNN, GradientBoosting
    ('voting_lr_knn_gb', MultiOutputClassifier(VotingClassifier(estimators=[('lr', LogisticRegression()),
                                                     ('knn', KNeighborsClassifier(n_neighbors=5)),
                                                     ('gb', GradientBoostingClassifier(n_estimators=100))],
                                    voting='soft')))
]

# Define list to store benchmark results
benchmark_results = []

# Train and evaluate models
models_to_evaluate = [('stacking', stacking_model)] + [(name, model) for name, model in voting_models]

for name, model in models_to_evaluate:
    # Perform cross-validation
    cv_results = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    accuracy_mean = np.mean(cv_results)
    
    # Train and test final model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Save results
    benchmark_results.append({
        'Classifier': name,
        'Accuracy Mean': accuracy_mean,
        'Test Accuracy': test_accuracy
    })

# Convert benchmark results to DataFrame
benchmark_df = pd.DataFrame(benchmark_results)

# Save the results to a CSV file
benchmark_df.to_csv("stacking-voting_benchmark_results.csv", index=False)

# Print the results
print(benchmark_df)
