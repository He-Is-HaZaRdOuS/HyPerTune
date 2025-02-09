#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
  echo "Error: Thread Count Argument is missing."
  exit 1
fi

# Check if the argument is a positive integer
if ! [[ "$1" =~ ^[0-9]+$ ]] || [ "$1" -le 0 ]; then
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

# Thread count
THREAD_COUNT=$1

# Timeout duration in seconds
export TIMEOUT_DURATION=300 # 5 mins

# Function to handle SIGINT (Ctrl+C)
cleanup() {
  echo "Process interrupted. Exiting..."
  exit 1
}

# Trap SIGINT (Ctrl+C)
trap cleanup SIGINT

# Base paths
export PATOH_BASE="/home/hazardous/Projects/patoh/"
export USER_BASE="/home/hazardous/Desktop/HyPerTune-tools/"
export MATRICES_DIR="${USER_BASE}matrices/"
LOG_DIR="${USER_BASE}patoh_logs/"
TIMED_OUT_DIR="${USER_BASE}patoh_timed_out/"

mkdir -p "$LOG_DIR"
mkdir -p "$TIMED_OUT_DIR"

export LOG_DIR
export TIMED_OUT_DIR

# Set the numeric locale to ensure correct formatting for numbers
export LC_NUMERIC="en_US.UTF-8"

# List of .patoh files (assuming these are already preprocessed for PaToH)
FILES=$(find "${MATRICES_DIR}" -maxdepth 1 -name "*.patoh" -type f | xargs du -b | sort -n | awk '{print $2}')

# Blocks
declare -i K=64

# Seeds
declare -i SEEDS=(-1 0 1 2 3 4 5 6 7 8 9 10)

# Epsilons
declare -f EPSILONS=(0.03)

# Function to handle the execution and logging of PaToH benchmark
run_patoh() {
    # The input string as a single argument
    arg_str="$1"

    # Split the argument string into separate variables (array elements)
    IFS=' ' read -r -a args <<< "$arg_str"

    # Assign each argument to the corresponding variable
    local FILE="${args[0]}"
    local BLOCK="${args[1]}"
    local EPSILON="${args[2]}"
    local SEED="${args[3]}"

    BASENAME=$(basename "$FILE" .hgr.patoh)

    LOG_FILE="${LOG_DIR}${BASENAME}_K${BLOCK}_SEED${SEED}_Epsilon${EPSILON}.log"
    TIMED_OUT_FILE="${TIMED_OUT_DIR}${BASENAME}_K${BLOCK}_SEED${SEED}_Epsilon${EPSILON}.log"

    # Check if log file already exists
    if [ -f "$LOG_FILE" ]; then
        echo "Skipping: Log file already exists for $BASENAME with K=$BLOCK."
        return
    fi

    # The PaToH command
    COMMAND="${PATOH_BASE}patoh \
        $FILE \
        $BLOCK \
        IB=$(printf '%.2f' "$EPSILON") \
        SD=$(printf '%d' "$SEED")"

    echo "Executed Command: $COMMAND"

    # Execute the command with a timeout and log the output
    {
        echo "Running PaToH with the following command:"
        echo "$COMMAND"

        ERROR_MSG=$(timeout $TIMEOUT_DURATION bash -c "$COMMAND" 2>&1)
        EXIT_STATUS=$?

        # Check the result
        if [ $EXIT_STATUS -eq 124 ]; then
            echo "Timeout: PaToH command timed out after $TIMEOUT_DURATION seconds." > "$TIMED_OUT_FILE"
            echo "Command: $COMMAND" >> "$TIMED_OUT_FILE"
        elif [ $EXIT_STATUS -eq 0 ]; then
            echo "Success: $FILE completed." > "$LOG_FILE"
            echo "Command: $COMMAND" >> "$LOG_FILE"
            echo "$ERROR_MSG" >> "$LOG_FILE"
        else
            echo "Error: PaToH failed for $FILE." > "$LOG_FILE"
            echo "Error message: $ERROR_MSG" >> "$LOG_FILE"
        fi
    }
}

export -f run_patoh

# Create a temporary file to store parameters
PARAMS_FILE=$(mktemp)

# Generate the list of parameters (one per line)
for FILE in $FILES; do
    for BLOCK in "${K[@]}"; do
        for EPSILON in "${EPSILONS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                echo "$FILE $BLOCK $EPSILON $SEED" >> "$PARAMS_FILE"
            done
        done
    done
done

# Run the parallel command using the temporary file as input
parallel -j ${THREAD_COUNT} -a "$PARAMS_FILE" run_patoh

# Clean up the temporary file
rm "$PARAMS_FILE"

echo "Partitioning complete. Logs stored in $LOG_DIR"
