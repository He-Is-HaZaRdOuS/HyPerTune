import numpy as np
import pandas as pd

# Read the CSV file
raw_dir = "../data/raw/"
processed_dir = "../data/processed/"

filename = "matrix_data_6class.csv"

input_csv = f"{processed_dir}{filename}"
output_csv = f"{processed_dir}DISCRETIZED_{filename}"

df = pd.read_csv(input_csv)


# Process each row
def process_row(row):
    # Convert row to numeric values and get the minimum
    numeric_row = pd.to_numeric(row, errors="coerce")
    min_value = numeric_row.min()
    threshold = min_value * 1.01  # Calculate 1% more than minimum
    # Convert values and compare with threshold
    return [1 if value <= threshold else 0 for value in numeric_row]


# Automatically detect target binary label columns
columns = [col for col in df.columns if col.startswith(("dir", "rec"))]

# Create a new DataFrame with processed values
df_processed = df.copy()

# Convert columns to numeric type first
df[columns] = (
    df[columns]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(np.iinfo(np.int64).max)
)

# Apply the processing row-wise for the selected columns
for idx, row in df[columns].iterrows():
    processed_values = process_row(row)
    df_processed.loc[idx, columns] = processed_values

# Save the processed DataFrame to a new CSV file
df_processed.to_csv(output_csv, index=False)
print(f"Done! Wrote: {output_csv}")
