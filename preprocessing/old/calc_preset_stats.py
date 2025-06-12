import pandas as pd
import os

# Read the input data from a CSV file (you can change this to your actual file)
input_file = '2class_synthetic_filtered.csv'  # Replace with the path to your actual input file
output_file = 'sorted_best_presets.txt'

# Function to extract just the filename from the full path, checking for valid strings
def get_filename_from_path(file_path):
    if isinstance(file_path, str):  # Only process if the value is a string
        return os.path.basename(file_path)
    return file_path  # If it's not a string (e.g., NaN), return it as is

# Load the data into a DataFrame
df = pd.read_csv(input_file)

df['Preset'] = df['Preset'].apply(get_filename_from_path)

# Assuming the columns are 'Matrix', 'Preset', and 'Cut'
# Group by matrix and preset, and find the minimum cut for each matrix
min_cut_per_matrix = df.groupby('Matrix')['Cut'].min()

# Now, for each matrix, find the preset that produces the minimum cut
matrix_preset_min_cut = pd.DataFrame(columns=['Matrix', 'Preset', 'Cut'])

for matrix in min_cut_per_matrix.index:
    # Get all rows for this matrix
    matrix_data = df[df['Matrix'] == matrix]

    # Find the preset(s) that produces the minimum cut
    min_cut_value = min_cut_per_matrix[matrix]
    min_cut_row = matrix_data[matrix_data['Cut'] == min_cut_value]

    for _, row in min_cut_row.iterrows():
        matrix_preset_min_cut = matrix_preset_min_cut._append({
            'Matrix': matrix,
            'Preset': row['Preset'],
            'Cut': min_cut_value
        }, ignore_index=True)

# Count how many times each preset produces the minimum cut across all matrices
preset_count = matrix_preset_min_cut.groupby('Preset').size().reset_index(name='Count')

preset_count = preset_count.sort_values(by='Count', ascending=False)

# Output the result to a text file
with open(output_file, 'w') as file:
    # Write the header
    file.write('Preset\tCount\n')

    # Write the rows
    for _, row in preset_count.iterrows():
        file.write(f"{row['Preset']}\t{row['Count']}\n")

print(f"Results written to {output_file}")
