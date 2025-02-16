import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('matrix_data.csv')

# Process each row
def process_row(row):
    # Convert row to numeric values and get the minimum
    numeric_row = pd.to_numeric(row, errors='coerce')
    min_value = numeric_row.min()
    threshold = min_value * 1.01  # Calculate 1% more than minimum
    # Convert values and compare with threshold
    return [1 if value <= threshold else 0 for value in numeric_row]

# Apply the transformation to specified columns
columns = [
    'dir_full_first', 'dir_full_worst',
    'dir_lazy_first', 'dir_lazy_worst',
    'dir_ml_first', 'dir_ml_worst',
    'rec_full_first', 'rec_full_worst',
    'rec_lazy_first', 'rec_lazy_worst',
    'rec_ml_first', 'rec_ml_worst'
]

# Create a new DataFrame with processed values
df_processed = df.copy()

# Convert columns to numeric type first
df[columns] = df[columns].apply(pd.to_numeric, errors='coerce')

# Apply the processing row-wise for the selected columns
for idx, row in df[columns].iterrows():
    processed_values = process_row(row)
    df_processed.loc[idx, columns] = processed_values

# Save the processed DataFrame to a new CSV file
df_processed.to_csv('matrix_data_processed.csv', index=False)
