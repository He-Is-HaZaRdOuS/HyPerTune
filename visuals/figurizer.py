import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# Path to your CSV data file.
data_file = "matrix_data_processed2.csv"  # Update this path if needed

# --- Data Loading ---
# Read the CSV file.
df = pd.read_csv(data_file, sep=",")

# Print column names to help with selection.
print("Available columns:", df.columns.tolist())

# --- Column Selection ---
# Specify the columns you want to use (e.g., class columns).
# Update this list with the actual column names.
class_columns = [
# "dir_ml_first",
 "dir_ml_worst",
# "dir_lazy_first",
# "dir_lazy_worst",
# "dir_full_first",
# "dir_full_worst",
# "rec_ml_first",
# "rec_ml_worst",
# "rec_lazy_first",
 "rec_lazy_worst",
# "rec_full_first",
# "rec_full_worst",
#'dir_i-algo_pool',
#'dir_i-algo_greedy_sequential',
#'dir_i-algo_greedy_round',
#'dir_i-algo_greedy_global',
#'rec_i-algo_pool',
#'rec_i-algo_greedy_sequential',
#'rec_i-algo_greedy_round',
#'rec_i-algo_greedy_global'
#'dir_ml_worst_kway_fm_hyperflow_cutter_km1',
#'dir_ml_worst_kway_fm_km1',
#'rec_lazy_worst_twoway_fm_hyperflow_cutter',
#'rec_lazy_worst_twoway_fm',
 ]

# Check if specified columns exist in the dataframe.
missing_cols = [col for col in class_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"The following columns are missing in the dataframe: {missing_cols}")

# --- Plot 1: Class Frequency Bar Plot ---
class_freq = df[class_columns].sum()

plt.figure(figsize=(12, 8))
ax = sns.barplot(x=class_freq.index, y=class_freq.values, palette="viridis")
plt.xlabel("Classes")
plt.ylabel("Number of Matrices")
plt.title("Distribution of Matrices per Class")
plt.xticks(rotation=45)

# Add annotations on top of bars
for i, value in enumerate(class_freq.values):
    ax.text(i, value + 1, str(int(value)), ha='center', va='bottom')

plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()

# --- Plot 2: Correlation Heatmap ---
corr_matrix = df[class_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=0, vmax=1)
plt.title("Correlation Heatmap of Class Labels")
plt.tight_layout()
plt.savefig("class_correlation_heatmap.png")
plt.show()

# --- Plot 3: Distribution of Label Counts per Matrix with Annotations ---
df['label_count'] = df[class_columns].sum(axis=1)

# Filter out rows where label_count is zero
df = df[df['label_count'] > 0]

plt.figure(figsize=(12, 8))
ax = sns.countplot(x='label_count', data=df, hue='label_count', palette="viridis", dodge=False)
ax.get_legend().remove()

print(df['label_count'].value_counts().sort_index())
print(df[class_columns].sum())


for patch in ax.patches:
    count = int(patch.get_height())
    x_coord = patch.get_x() + patch.get_width() / 2
    y_coord = patch.get_height()
    ax.annotate(f'{count}', (x_coord, y_coord), ha='center', va='bottom',
                xytext=(0, 3), textcoords='offset points')

plt.xlabel("Number of Classes per Matrix")
plt.ylabel("Number of Matrices")
plt.title("Distribution of Label Counts per Matrix")
plt.tight_layout()
plt.savefig("label_count_distribution.png")
plt.show()
