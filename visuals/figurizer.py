import os

import matplotlib as mpl
import matplotlib.pyplot as plt

# Global font and tick size settings
mpl.rcParams.update(
    {
        "font.size": 16,  # Default text size
        "axes.titlesize": 18,  # Plot title
        "axes.labelsize": 16,  # X/Y axis labels
        "xtick.labelsize": 14,  # Tick labels
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 20,  # Figure title if used
    }
)
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")

# --- Configuration ---
# Path to your CSV data file.

ml_ready_dir = "../data/ml_ready/"
results_dir = "../results/class_distributions/"
# input_filename = "pruned_multilabel_DISCRETIZED_matrix_data_dir_rec_2class.csv"
# input_csv = f"{ml_ready_dir}{input_filename}"
os.makedirs(results_dir, exist_ok=True)
# --- Data Loading ---
# Read the CSV file.
# df = pd.read_csv(input_csv, sep=",")

# Print column names to help with selection.
# print("Available columns:", df.columns.tolist())

for input_filename in os.listdir(ml_ready_dir):
    if not input_filename.endswith(".csv"):
        continue
    if "pruned_multilabel" in input_filename:
        continue  # skip file

    print(f"\n=== Processing {input_filename} ===")
    input_csv = os.path.join(ml_ready_dir, input_filename)

    df = pd.read_csv(input_csv)

    class_columns = [
        col
        for col in df.columns
        if set(df[col].dropna().unique()) <= {0, 1}
        and (col.startswith("dir") or col.startswith("rec"))
    ]

    print("Selected columns:", class_columns)

    # --- Column Selection ---
    # Specify the columns you want to use (e.g., class columns).
    # Update this list with the actual column names.
    # class_columns = [
    # # "dir_ml_first",
    #  "dir_ml_worst",
    # # "dir_lazy_first",
    # # "dir_lazy_worst",
    # # "dir_full_first",
    # # "dir_full_worst",
    # # "rec_ml_first",
    # # "rec_ml_worst",
    # # "rec_lazy_first",
    #  # "rec_lazy_worst",
    # "rec_full_first",
    # # "rec_full_worst",
    # #'dir_i-algo_pool',
    # #'dir_i-algo_greedy_sequential',
    # #'dir_i-algo_greedy_round',
    # #'dir_i-algo_greedy_global',
    # #'rec_i-algo_pool',
    # #'rec_i-algo_greedy_sequential',
    # #'rec_i-algo_greedy_round',
    # #'rec_i-algo_greedy_global'
    # #'dir_ml_worst_kway_fm_hyperflow_cutter_km1',
    # #'dir_ml_worst_kway_fm_km1',
    # #'rec_lazy_worst_twoway_fm_hyperflow_cutter',
    # #'rec_lazy_worst_twoway_fm',
    #  ]

    # Check if specified columns exist in the dataframe.
    missing_cols = [col for col in class_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The following columns are missing in the dataframe: {missing_cols}"
        )

    # --- Plot 1: Class Frequency Bar Plot ---
    class_freq = df[class_columns].sum()

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        x=class_freq.index,
        y=class_freq.values,
        palette="viridis",
        legend=False,
        hue=class_freq.index,
    )
    plt.xlabel("Classes")
    plt.ylabel("Number of Matrices")
    plt.title("Distribution of Matrices per Class")
    plt.xticks(rotation=45)

    # Add annotations on top of bars
    for i, value in enumerate(class_freq.values):
        ax.text(
            i, value + 1, str(int(value)), ha="center", va="bottom", fontsize=14
        )

    plt.tight_layout()
    plt.savefig(f"{results_dir}{input_filename}_class_distribution.png")
    # plt.show()

    # --- Plot 2: Correlation Heatmap ---
    corr_matrix = df[class_columns].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=0, vmax=1)
    plt.title("Correlation Heatmap of Class Labels")
    plt.tight_layout()
    plt.savefig(f"{results_dir}{input_filename}_class_correlation_heatmap.png")
    # plt.show()

    # --- Plot 3: Distribution of Label Counts per Matrix with Annotations ---
    df["label_count"] = df[class_columns].sum(axis=1)

    # Filter out rows where label_count is zero
    df = df[df["label_count"] > 0]

    plt.figure(figsize=(12, 8))
    ax = sns.countplot(
        x="label_count",
        data=df,
        hue="label_count",
        palette="viridis",
        dodge=False,
    )
    ax.get_legend().remove()

    # print(df["label_count"].value_counts().sort_index())
    # print(df[class_columns].sum())

    for patch in ax.patches:
        count = int(patch.get_height())
        x_coord = patch.get_x() + patch.get_width() / 2
        y_coord = patch.get_height()
        ax.annotate(
            f"{count}",
            (x_coord, y_coord),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=14,
        )

    plt.xlabel("Number of Classes per Matrix")
    plt.ylabel("Number of Matrices")
    plt.title("Distribution of Label Counts per Matrix")
    plt.tight_layout()
    plt.savefig(f"{results_dir}{input_filename}_label_count_distribution.png")
    # plt.show()

    # print(df[class_columns].sum())

    # --- Imbalance Analysis ---
    total_with_any_label = len(df)
    class_counts = df[class_columns].sum()

    print("\n=== Imbalance Analysis ===")
    imbalance_info = {}

    for col in class_columns:
        count = class_counts[col]
        rest = total_with_any_label - count
        imbalance_ratio = round(count / rest, 4) if rest != 0 else float("inf")
        percent_class = 100 * count / total_with_any_label
        percent_rest = 100 * rest / total_with_any_label
        percent_diff = round(abs(percent_class - percent_rest), 2)

        imbalance_info[col] = {
            "Class Count": int(count),
            "Rest Count": int(rest),
            "Imbalance Ratio (C/Rest)": round(imbalance_ratio, 3),
            "Class %": round(percent_class, 2),
            "Rest %": round(percent_rest, 2),
            "Imbalance % Difference": percent_diff,
        }

    # Print table
    from tabulate import tabulate

    print(input_filename)
    print(
        tabulate(
            [(k,) + tuple(v.values()) for k, v in imbalance_info.items()],
            headers=[
                "Class",
                "Class Count",
                "Rest Count",
                "Imbalance Ratio (C/Rest)",
                "Class %",
                "Rest %",
                "Imbalance % Difference",
            ],
            tablefmt="pretty",
        )
    )

    # --- Optional: Save imbalance stats to CSV ---
    imbalance_df = pd.DataFrame.from_dict(imbalance_info, orient="index")
    imbalance_df.to_csv(f"{results_dir}{input_filename}_imbalance_stats.csv")
