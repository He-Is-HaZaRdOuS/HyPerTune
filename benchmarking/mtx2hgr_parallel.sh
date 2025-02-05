#!/bin/bash

# Base paths
export KAHYPAR_BASE="/home/hazardous/Projects/kahypar/"
export USER_BASE="/home/hazardous/Desktop/HyPerTune-tools/"
FILES=${USER_BASE}/matrices/*.mtx

# Loop through each file and convert to hgr
for FILE in $FILES; do
    echo "Converting $FILE..."
    ${KAHYPAR_BASE}/build/tools/MtxToHgr "$FILE"
done

echo "Conversion complete."
