#!/bin/bash

cat <<'EOF'
          .
        /'
       //
   .  //
   |\//7
  /' " \
 .   . .
 | (    \     '._
 |  '._  '    '. '
 /    \'-'_---. ) )
.              :.'
|               \
| .    .   .     .
' .    |  |      |
 \^   /_-':     /
 / | |    '\  .'
/ /| |     \\  |
\ \( )     // /
 \ | |    // /
  L! !   // / Ballzagna0x0a
   [_]  L[_| He_Is_HaZaRdOuS

EOF

# Define the common base directory
BASE_DIR="/home/hazardous/Desktop/HyPerTune-tools/"

# Subdirectories to clean
SUBDIRS=("kahypar_logs" "kahypar_timed_out")

# Function to clean a directory
clean_directory() {
  local dir="$1"
  if [ -d "$dir" ]; then
    echo "Cleaning directory: $dir"
    # Use find to delete files and directories in smaller batches
    find "$dir" -mindepth 1 -delete
    echo "Contents of $dir removed."
  else
    echo "Directory $dir does not exist, skipping."
  fi
}

# Confirm before proceeding
echo "The following directories under $BASE_DIR will be nuked:"
for subdir in "${SUBDIRS[@]}"; do
  echo "- $BASE_DIR$subdir"
done

read -p "Are you sure you want to proceed? (yes/no): " CONFIRM
if [[ "$CONFIRM" == "yes" ]]; then
  for subdir in "${SUBDIRS[@]}"; do
    full_path="$BASE_DIR$subdir"
    clean_directory "$full_path"
  done
  echo "Nuking complete."
else
  echo "Operation canceled."
fi
