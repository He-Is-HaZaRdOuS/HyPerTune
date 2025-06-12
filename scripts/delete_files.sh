#!/bin/bash

# Specify the directory
directory="/home/hazardous/Desktop/HyPerTune-tools/kahypar_logs"

# List of allowed substrings in filenames
allowed_strings=(
  "km1_rKaHyPar_sea20_c-type_heavy_lazy_i-bp-algorithm_worst_fit"
  "rec_i-algo_greedy_round_K64"
  "rec_i-algo_greedy_global_K64"
  "rec_i-algo_greedy_sequential_K64"
)

# Iterate through all files in the specified directory
for file in "$directory"/*; do
  # Check if it's a regular file
  if [[ -f "$file" ]]; then
    match=false
    # Check if the file name contains any of the allowed strings
    for allowed in "${allowed_strings[@]}"; do
      if [[ "$file" == *"$allowed"* ]]; then
        match=true
        break
      fi
    done
    
    # If no match found, delete the file
    if ! $match; then
      echo "Deleting file: $file"
      rm "$file"
    fi
  fi
done

