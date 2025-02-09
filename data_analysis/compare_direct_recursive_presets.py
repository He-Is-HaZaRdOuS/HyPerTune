import pandas as pd

# Define the presets to filter
TARGET_PRESETS = {
    "km1_kKaHyPar_sea20_c-type_ml_style_i-bp-algorithm_worst_fit.ini",
    "km1_rKaHyPar_sea20_c-type_ml_style_i-bp-algorithm_worst_fit.ini"
}

# Read the raw CSV file
input_csv = "preset_data.csv"  # Change this to your actual input file
output_csv = "sorted_preset_cut_comparison.csv"

# Load data
df = pd.read_csv(input_csv)

# Ensure the required columns exist
if not {"Matrix", "Preset", "Average - Cut"}.issubset(df.columns):
    raise ValueError("CSV file must contain 'Matrix', 'Preset', and 'Average - Cut' columns.")

# Filter data for the target presets
filtered_df = df[df["Preset"].isin(TARGET_PRESETS)][["Matrix", "Preset", "Average - Cut"]]

# Sort by matrix name (Matrix column)
sorted_df = filtered_df.sort_values(by="Matrix")

# Save to CSV
sorted_df.to_csv(output_csv, index=False)

print(f"Processed data saved to {output_csv}")
