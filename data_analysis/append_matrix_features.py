import pandas as pd

# Load the two CSV files into pandas DataFrames
csv_data = pd.read_csv("input_data.csv")
matrix_features = pd.read_csv("matrix_features.csv")

csv_data = csv_data.fillna("TIMED_OUT")

# Find the intersection of matrix names (ignoring case and potential extensions)
intersecting_matrices = pd.merge(
    csv_data, matrix_features, left_on='Matrix', right_on='Matrix Name', how='inner'
)

# Replace NaN values with "N/A"
intersecting_matrices = intersecting_matrices.fillna("N/A")

# Drop the redundant 'Matrix Name' column
intersecting_matrices = intersecting_matrices.drop(columns=['Matrix Name'])

# Save the intersecting matrices to a new CSV
intersecting_matrices.to_csv("matrix_data.csv", index=False)

print("A new CSV containing only intersecting matrices has been saved as 'matrix_data.csv'.")
