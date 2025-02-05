# Matrix File Conversion and Benchmarking

This guide provides detailed steps on how to convert `.mtx` files to `.hgr` files, create variations of presets, and run benchmarking tasks in parallel using the provided shell scripts.

## Notice

    KaHyPar is required for its `MtxToHgr` executable that actually does the conversion.
    Make sure to edit the `KAHYPAR_BASE` variable to point to the KaHyPar base installation path.

## Overview

This process involves:
1. Converting `.mtx` files to `.hgr` files using the `mtx2hgr_parallel.sh` script.
2. Creating variations of preset files with the `create_preset_variations.sh` script.
3. Running benchmarking tasks in parallel with the `run_smart_benchmarks_parallel.sh` script.

## Steps for Conversion, Preset Creation, and Benchmarking

### Step 1: Convert `.mtx` Files to `.hgr` Files

    To convert `.mtx` files (Matrix Market format) to `.hgr` files (Hypergraph format), use the `mtx2hgr_parallel.sh` script.
    Place your `.mtx` files in the appropriate directory.
    Open the `mtx2hgr_parallel.sh` script and check if any modifications are needed (e.g., adjusting paths or input/output directories).
    Run the script to start the conversion process:

  ```bash
  mtx2hgr_parallel.sh
  ```
    The script will convert all compatible .mtx files in the directory to .hgr files in parallel, speeding up the process.

### Step 2: Create Variations of Presets

    If you need to generate variations of presets, you can use the create_preset_variations.sh script. This allows you to create different configurations of preset files for testing or benchmarking purposes.
    Check the create_preset_variations.sh script for any necessary edits, such as specifying the directory for presets or adjusting other parameters.
    Run the script to generate preset variations:
    bash create_preset_variations.sh
    The script will create several preset variations according to the configuration settings you’ve specified.

### Step 3: Run Benchmarking Tasks

    Once you have your .hgr files and preset variations, you can run benchmarking tasks in parallel using the run_smart_benchmarks_parallel.sh script.
    Open the run_smart_benchmarks_parallel.sh script and review the settings. You may need to edit the script to match your desired benchmarking configurations (e.g., adjusting input files, parameters, or output directories).
    Execute the script to begin the benchmarking process:
    bash run_smart_benchmarks_parallel.sh
    This script will run the benchmarks on all the relevant files and configurations, processing them in parallel to save time.

### Step 4: Edit Scripts (if Necessary)

    If the default configurations in any of the scripts don’t match your specific needs, you may need to edit them.
    For example, in mtx2hgr_parallel.sh, you might want to specify a custom output directory.
    In create_preset_variations.sh, you might want to change the preset parameters or the number of variations generated.
    In run_smart_benchmarks_parallel.sh, you could modify how many processes run in parallel or specify custom parameters for the benchmarking task.

### Step 5: Monitor the Process

    During each of the steps, monitor the output logs to ensure the scripts are running correctly.
    If there are any issues (e.g., missing files, errors in conversion or benchmarking), debug by reviewing the error messages and adjusting the scripts accordingly.

### Final Notes

    Ensure that all input files (such as .mtx files and preset files) are in the correct format and located in the appropriate directories before starting the scripts.
    If you run into performance issues or need to optimize the parallel execution, consider adjusting the number of parallel processes in the scripts.
