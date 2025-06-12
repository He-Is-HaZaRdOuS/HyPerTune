#!/bin/bash

# Directory containing the original .ini files
PRESETS_DIR="/home/hazardous/Desktop/HyPerTune/presets"

# Variable to modify and its new values
VARIABLE="i-bp-algorithm"
NEW_VALUES=("first_fit" "worst_fit")  # Add more values as needed

# Loop through all .ini files in the directory
for FILE in "${PRESETS_DIR}"/*.ini; do
    BASENAME=$(basename "$FILE" .ini)

    # Loop through each new value for the variable
    for VALUE in "${NEW_VALUES[@]}"; do
        # Create a new filename with the value appended
        NEW_FILENAME="${PRESETS_DIR}/${BASENAME}_${VARIABLE}_${VALUE}.ini"

        # Copy the original file to the new filename
        cp "$FILE" "$NEW_FILENAME"

        # Modify the variable in the new file
        sed -i "s/^${VARIABLE}=.*/${VARIABLE}=${VALUE}/" "$NEW_FILENAME"

        echo "Created: $NEW_FILENAME"
    done

    # Delete the original file
    rm "$FILE"
    echo "Deleted original file: $FILE"
done

echo "Variations created successfully!"
