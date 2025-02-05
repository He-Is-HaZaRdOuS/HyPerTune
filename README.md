# HyPerTune-tools

**HyPerTune-tools** is a suite of scripts designed to automate the benchmarking and data analysis of the **Kahypar** program. These tools are built to handle hundreds of matrices, enabling you to convert matrix files, generate preset variations, run performance benchmarks, and analyze the results efficiently.

## ⚠️ Usage Notice

Before running any scripts, **you must edit them to adjust file paths** to match your system's directory structure. Many scripts include absolute or relative file paths that need to be customized for your environment.

### How to Update Paths:
1. Open the relevant script in a text editor (e.g., VS Code, nano, vim).
2. Locate file paths inside the script (e.g., `/home/user/matrices/` or `./logs/`).
3. Modify them to match your local directory setup.
4. Save the changes before running the script.

---

## Overview

This repository includes scripts and tools for:

1. **Matrix Conversion**: Converting `.mtx` files (Matrix Market format) to `.hgr` files (Hypergraph format).
2. **Preset Generation**: Creating variations of preset configurations to be tested with matrices.
3. **Benchmarking**: Running benchmarking tasks on matrices with preset variations using the **Kahypar** program.
4. **Data Analysis**: Collecting benchmarking results, processing them, and performing analysis.
5. **Timeout Handling**: Managing timeouts for benchmarking tasks that may take too long.

## Repository Structure

The repository is organized as follows:

### Subfolders

- **`logs/`**: Folder where log files from benchmarking tasks are stored. Contains detailed logs of all benchmarking runs.
- **`matrices/`**: Folder containing `.mtx` files (Matrix Market format) to be used for benchmarking.
- **`presets/`**: Folder for preset configuration files used during benchmarking.
- **`data_analysis/`**: Folder where the processed results of the benchmarking tasks are stored, including any visualizations or data analysis outputs.
- **`data_gathering/`**: Folder where scripts for gathering and preparing matrix data are stored.
- **`benchmarking/`**: Folder containing the core benchmarking scripts, including scripts for initiating benchmarking tasks and handling results.
- **`timed_out/`**: Folder where benchmarking tasks that exceeded their time limit are placed.

## Installation

To use **HyPerTune-tools**, ensure you have the following:

1. **Python 3.6+** installed.
2. **Bash** available (for running shell scripts).
3. An internet connection for downloading matrices and dependencies.

To install dependencies, clone the repository and install Python requirements:

```bash
git clone https://github.com/He-Is-HaZaRdOuS/HyPerTune-tools
cd HyPerTune-tools
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
