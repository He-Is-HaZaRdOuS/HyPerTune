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
export TIMEOUT_DURATION=1

# Function to handle SIGINT (Ctrl+C)
cleanup() {
  echo "Process interrupted. Exiting..."
  exit 1
}

# Trap SIGINT (Ctrl+C)
trap cleanup SIGINT

# Base paths
export KAHYPAR_BASE="/home/hazardous/Projects/kahypar/"
export USER_BASE="/home/hazardous/Desktop/HyPerTune/"
export MATRICES_DIR="${USER_BASE}matrices/"
export CONFIG_DIR="${USER_BASE}presets/"
LOG_DIR="${USER_BASE}data/logs/kahypar_logs/"
TIMED_OUT_DIR="${USER_BASE}data/logs/kahypar_timed_out/"

mkdir -p "$LOG_DIR"
mkdir -p "$TIMED_OUT_DIR"

export LOG_DIR
export TIMED_OUT_DIR

# Set the numeric locale to ensure correct formatting for numbers (especially floating point decimals)
export LC_NUMERIC="en_US.UTF-8"

# List of .hgr files, sorted by size (smallest to largest)
FILES=$(find "${MATRICES_DIR}" -maxdepth 1 -name "*.hgr" -type f | xargs du -b | sort -n | awk '{print $2}')

# Preset configuration files in the config directory (no subdirectories)
PRESETS=($(find "${CONFIG_DIR}" -maxdepth 1 -name "*.ini" -type f))

# Blocks
declare -i K=(64)

# Seeds
declare -i SEEDS=(-1 0 1 2 3 4 5 6 7 8 9 10)

# Epsilons
declare -f EPSILONS=(0.03)

# Function to handle the execution and logging
run_kahypar() {
    # The input string as a single argument
    arg_str="$1"

    # Split the argument string into separate variables (array elements)
    IFS=' ' read -r -a args <<< "$arg_str"

    # Assign each argument to the corresponding variable
    local FILE="${args[0]}"
    local PRESET="${args[1]}"
    local BLOCK="${args[2]}"
    local EPSILON="${args[3]}"
    local SEED="${args[4]}"

    BASENAME=$(basename "$FILE" .hgr)
    PRESET_NAME=$(basename "$PRESET" .ini)

    # Extract mode and objective from the .ini file
    MODE=$(grep -E '^mode\s*=' "${PRESET}" | awk -F '=' '{print $2}' | xargs)
    OBJECTIVE=$(grep -E '^objective\s*=' "${PRESET}" | awk -F '=' '{print $2}' | xargs)
    COARSENING_TYPE=$(grep -E '^c-type\s*=' "${PRESET}" | awk -F '=' '{print $2}' | xargs)

    if [ -z "$MODE" ] || [ -z "$OBJECTIVE" ] || [ -z "$COARSENING_TYPE" ]; then
        echo "Error: Mode or Objective or Coarsening Type is missing in ${PRESET}."
        exit 1
    fi

#     CURRENT_TIME=$(date '+%Y-%m-%d_%H-%M-%S')
    LOG_FILE="${LOG_DIR}${BASENAME}_${PRESET_NAME}_K${BLOCK}_SEED${SEED}_Epsilon${EPSILON}.log"
    TIMED_OUT_FILE="${TIMED_OUT_DIR}${BASENAME}_${PRESET_NAME}_K${BLOCK}_SEED${SEED}_Epsilon${EPSILON}.log"

    # Check if log file already exists
    if [ -f "$LOG_FILE" ]; then
        echo "Skipping: Log file already exists for $BASENAME with $PRESET_NAME, K=$BLOCK, Seed=$SEED, Epsilon=$EPSILON."
        return
    fi

    # Construct the KaHyPar command
    COMMAND="${KAHYPAR_BASE}build/kahypar/application/KaHyPar \
        -p ${PRESET} \
        -h $FILE \
        -k $BLOCK \
        -e $(printf '%.2f' "$EPSILON") \
        -m $MODE \
        -o $OBJECTIVE \
        --seed $(printf '%d' "$SEED")"

    echo "Executed Command: $COMMAND"

    # Execute the command with a timeout and log the output
    {
        echo "Running KaHyPar with the following command:"
        echo "$COMMAND"

        ERROR_MSG=$(timeout $TIMEOUT_DURATION bash -c "$COMMAND" 2>&1)
        EXIT_STATUS=$?

        # Check the result
        if [ $EXIT_STATUS -eq 124 ]; then
            echo "Timeout: KaHyPar command timed out after $TIMEOUT_DURATION seconds." > "$TIMED_OUT_FILE"
            echo "Command: $COMMAND" >> "$TIMED_OUT_FILE"
        elif [ $EXIT_STATUS -eq 0 ]; then
            echo "Success: $FILE with $PRESET_NAME completed." > "$LOG_FILE"
            echo "Command: $COMMAND" >> "$LOG_FILE"
            echo "$ERROR_MSG" >> "$LOG_FILE"
        else
            echo "Error: KaHyPar failed for $FILE with $PRESET_NAME." > "$LOG_FILE"
            echo "Error message: $ERROR_MSG" >> "$LOG_FILE"
        fi
    }
}

export -f run_kahypar

# Create a temporary file to store parameters
PARAMS_FILE=$(mktemp)

# Generate all parameter combinations and write them to the temporary file
for FILE in $FILES; do
    for PRESET in "${PRESETS[@]}"; do
        for BLOCK in "${K[@]}"; do
            for EPSILON in "${EPSILONS[@]}"; do
                for SEED in "${SEEDS[@]}"; do
                    echo "$FILE $PRESET $BLOCK $EPSILON $SEED" >> "$PARAMS_FILE"
                done
            done
        done
    done
done

# Run the parallel command using the temporary file as input
parallel -j ${THREAD_COUNT} -a "$PARAMS_FILE" run_kahypar

# Clean up the temporary file
rm "$PARAMS_FILE"

echo "Partitioning complete. Logs stored in $LOG_DIR"
