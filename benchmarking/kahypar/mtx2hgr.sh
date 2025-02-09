#!/bin/bash

# Check if thread count argument is provided
if [ -z "$1" ]; then
  echo "Error: Thread Count Argument is missing."
  exit 1
fi

# Check if the argument is a positive integer
if ! [[ "$1" =~ ^[0-9]+$ ]] || [ "$1" -le 0 ]]; then
  echo "Error: Thread Count Argument must be a positive integer."
  exit 1
fi

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

# Assign thread count
THREAD_COUNT=$1

# List of .mtx files
FILES=/home/hazardous/Desktop/HyPerTune-tools/matrices/*.mtx

# Function to convert .mtx to .hgr
convert_file() {
    FILE="$1"
    echo "Converting $FILE..."
    /software/kahypar/build/tools/MtxToHgr "$FILE"
}

export -f convert_file

# Run in parallel using user-specified thread count
parallel -j "$THREAD_COUNT" convert_file ::: $FILES

echo "Conversion complete."
