import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Dosya yolunu belirle
file_path = 'matrix_data_processed.csv'

# CSV dosyasını pandas DataFrame olarak oku
df = pd.read_csv(file_path)

# Hedef sütunları belirle
target_columns = ['dir_ml_worst', 'rec_lazy_worst']

# Özellik ve hedef değişkenleri ayır
columns_to_drop = target_columns + ["Matrix",
                                     "Group",
                                     "Kind",
                                     "Count",
                                     "Min",
                                     "Max",
                                     "PaToH-v3.3",
                                     "Average",
                                    "NNZ",
                                    "M",
                                    "N",
                                    "Row Max",
                                    "Row Mean",
                                    "Row Median",
                                    "Row Min",
                                    "Column Max",
                                    "Column Mean",
                                    "Column Median",
                                    "Column Min",
                                     "dir_full_first",
                                     "dir_full_worst",
                                     "dir_lazy_first",
                                     "dir_lazy_worst",
                                     "dir_ml_first",
                                     "dir_ml_worst",
                                     "rec_full_first",
                                     "rec_full_worst",
                                     "rec_lazy_first",
                                     "rec_lazy_worst",
                                     "rec_ml_first",
                                     "rec_ml_worst",
                                    # "Density",
                                    "Row STD",
                                    "Column STD",
                                    # "Column Max / M",
                                    # "Row Max / N",
                                    # "Column STD / M",
                                    # "Row STD / N",
                                    "Psym",
                                    "Nsym",
                                    "Kind-id",
                                    "Based-PSYM",
                                    "Based-ALL-Km7",
                                    "Based-COLMAX",
                                    "Based-logDensityEM7",
                                    "Based-Den_CsTDM_PSYM_NSYM-Km8",
                                    ]

X = df.drop(columns=columns_to_drop, errors='ignore')

# Performans karşılaştırmalarını yapalım
def compare_methods(row, target_a, target_b):
    method_a = row[target_a]
    method_b = row[target_b]
    if method_a > method_b:
        return 1  # Yöntem A daha iyi
    elif method_a < method_b:
        return 0  # Yöntem B daha iyi
    else:
        return 2  # İkisi eşitse, her ikisini de 1 al

# Y yeni etiketli veriyi oluştur
y = pd.DataFrame({
    'dir_ml_worst_vs_rec_lazy_worst': df.apply(lambda row: compare_methods(row, 'dir_ml_worst', 'rec_lazy_worst'), axis=1)
})

# Verileri normalize et
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim ve test verilerini ayır
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelleri oluştur ve eğit
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42)
}

# Modeli eğit
print(f"Training models for target: dir_ml_worst_vs_rec_lazy_worst")
y_train_target = y_train['dir_ml_worst_vs_rec_lazy_worst']
y_test_target = y_test['dir_ml_worst_vs_rec_lazy_worst']

if len(y_train_target.unique()) > 1:
    for model_name, model in models.items():
        model.fit(X_train, y_train_target)
        y_pred = model.predict(X_test)
        print(f"Classification Report for {model_name} (target: dir_ml_worst_vs_rec_lazy_worst):")
        print(classification_report(y_test_target, y_pred, zero_division=1))
        print(f"Accuracy Score for {model_name} (target: dir_ml_worst_vs_rec_lazy_worst):")
        print(accuracy_score(y_test_target, y_pred))
        print(f"Confusion Matrix for {model_name} (target: dir_ml_worst_vs_rec_lazy_worst):")
        print(confusion_matrix(y_test_target, y_pred))
        print("\n")

        # Özellik önemini yazdır
        if model_name == 'RandomForest':
            print(f"Feature Importances for {model_name}:")
            feature_importances = model.feature_importances_
            for feature_name, importance in zip(X.columns, feature_importances):
                print(f"{feature_name}: {importance}")
        elif model_name == 'LogisticRegression':
            print(f"Coefficients for {model_name}:")
            coefficients = model.coef_[0]
            for feature_name, coef in zip(X.columns, coefficients):
                print(f"{feature_name}: {coef}")
else:
    print(f"Warning: Only one class present in y_train_target for dir_ml_worst_vs_rec_lazy_worst. Skipping model training.")

# Map class labels for printing
print('\n')
label_mapping = {0: "rec_lazy_worst", 1: "dir_ml_worst"}

# Print class distributions with labels
print("Class distribution in the full dataset:")
print(y['dir_ml_worst_vs_rec_lazy_worst'].map(label_mapping).value_counts())

print("\nClass distribution in the training dataset:")
print(y_train_target.map(label_mapping).value_counts())

print("\nClass distribution in the test dataset:")
print(y_test_target.map(label_mapping).value_counts())

