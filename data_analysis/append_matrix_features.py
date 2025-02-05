import pandas as pd

# Load the two CSV files into pandas DataFrames
csv_data = pd.read_csv("matrix_info.csv")
matrix_features = pd.read_csv("matrix_features.csv")

# Merge the two DataFrames based on the Matrix name
# Assuming 'Matrix' in csv_data matches 'Matrix Name' in matrix_features
merged_data = pd.merge(csv_data, matrix_features, left_on='Matrix', right_on='Matrix Name', how='left')

# Drop the 'Matrix Name' column from the merged data, as it's not needed
merged_data = merged_data.drop(columns=['Matrix Name'])

# Save the merged DataFrame back to a CSV
merged_data.to_csv("merged_csv_data.csv", index=False)

print("Matrix features have been successfully appended to the CSV data.")
