import pandas as pd
import os

# Read input data
input_file = 'preset_data.csv'
output_file = 'best_vs_second_best_vs_worst.csv'

# Load data
df = pd.read_csv(input_file)

# Ensure necessary columns exist
if not {'Matrix', 'Preset', 'Cut'}.issubset(df.columns):
    raise ValueError("Input file must contain 'Matrix', 'Preset', and 'Cut' columns.")

# Function to extract just the filename from the full path, checking for valid strings
def get_filename_from_path(file_path):
    if isinstance(file_path, str):  # Only process if the value is a string
        return os.path.basename(file_path)
    return file_path  # If it's not a string (e.g., NaN), return it as is

# Apply the function to extract filenames
df['Matrix'] = df['Matrix'].apply(get_filename_from_path)
df['Preset'] = df['Preset'].apply(get_filename_from_path)

# Store results
results = []

# Process each matrix
for matrix, group in df.groupby('Matrix'):
    sorted_group = group.sort_values(by='Cut')  # Sort by Cut

    if len(sorted_group) < 3:
        continue  # Skip if there are less than 3 presets for this matrix

    best_preset = sorted_group.iloc[0]['Preset']
    best_cut = sorted_group.iloc[0]['Cut']

    second_best_preset = sorted_group.iloc[1]['Preset']
    second_best_cut = sorted_group.iloc[1]['Cut']

    worst_preset = sorted_group.iloc[-1]['Preset']  # Worst preset is the one with the highest cut
    worst_cut = sorted_group.iloc[-1]['Cut']

    # Calculate ratios
    if second_best_cut == 0:  # Avoid division by zero
        second_best_ratio = 1
    else:
        second_best_ratio = best_cut / second_best_cut

    if worst_cut == 0:  # Avoid division by zero
        worst_ratio = 1
    else:
        worst_ratio = best_cut / worst_cut

    results.append((matrix, best_preset, best_cut, second_best_preset, second_best_cut, second_best_ratio, worst_preset, worst_cut, worst_ratio))

# Convert results to a DataFrame
results_df = pd.DataFrame(results, columns=['Matrix', 'Best Preset', 'Best Cut', 'Second Best Preset', 'Second Best Cut', 'Second Best Ratio', 'Worst Preset', 'Worst Cut', 'Worst Ratio'])

# Sort by Best Cut (ascending) or any other column, for example 'Best Ratio'
results_df = results_df.sort_values(by='Best Cut', ascending=True)

# Save to a text file
results_df.to_csv(output_file, sep='\t', index=False)

print(f"Results written to {output_file}")
