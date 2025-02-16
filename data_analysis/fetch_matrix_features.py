import os
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix
import ssgetpy
from multiprocessing import Pool, cpu_count

# Configuration
MATRIX_DIRECTORY = "/home/hazardous/Desktop/HyPerTune-tools/matrices"
EXISTING_MATRICES_FILE = "matrix_list.txt"
FEATURES_OUTPUT_FILE = "og_matrix_features.csv"


def load_existing_matrices():
    if not os.path.exists(EXISTING_MATRICES_FILE):
        return set()
    with open(EXISTING_MATRICES_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())


def search_matrices():
    results = ssgetpy.search(limit=1000000000)
    return {matrix.name: matrix for matrix in results}


def intersect_matrices(existing_matrices, searched_matrices):
    existing_matrices_normalized = {os.path.splitext(name)[0] for name in existing_matrices}
    return {name: searched_matrices[name] for name in searched_matrices}


def calculate_matrix_properties(args):
    filename, filepath, matrix_search = args
    try:
        matrix = mmread(filepath)
        sparse_matrix = csr_matrix(matrix)

        properties = {
            'M': sparse_matrix.shape[0],
            'N': sparse_matrix.shape[1],
            'NNZ': sparse_matrix.nnz,
            'Density': sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1]),
        }

        row_nnz = sparse_matrix.getnnz(axis=1)
        col_nnz = sparse_matrix.getnnz(axis=0)

        properties.update({
            'Row Max': np.max(row_nnz),
            'Row Min': np.min(row_nnz),
            'Row Mean': np.mean(row_nnz),
            'Row Median': np.median(row_nnz),
            'Row STD': np.std(row_nnz),
            'Column Max': np.max(col_nnz),
            'Column Min': np.min(col_nnz),
            'Column Mean': np.mean(col_nnz),
            'Column Median': np.median(col_nnz),
            'Column STD': np.std(col_nnz),
        })

        properties.update({
            'Column Max / M': properties['Column Max'] / properties['M'] if properties['M'] != 0 else "N/A",
            'Row Max / N': properties['Row Max'] / properties['N'] if properties['N'] != 0 else "N/A",
            'Column STD / M': properties['Column STD'] / properties['M'] if properties['M'] != 0 else "N/A",
            'Row STD / N': properties['Row STD'] / properties['N'] if properties['N'] != 0 else "N/A",
            'Group': matrix_search.group or "N/A",
            'Kind': matrix_search.kind or "N/A",
            'Psym': matrix_search.psym if matrix_search.psym is not None else "N/A",
            'Nsym': matrix_search.nsym if matrix_search.nsym is not None else "N/A",
        })

        print(f"Processed: {filename}")
        return filename, properties

    except Exception as e:
        print(f"Skipping {filename} due to error: {e}")
        return filename, None


def process_matrices_parallel(directory, matrices):
    tasks = [(filename, os.path.join(directory, filename + '.mtx'), matrices[filename]) for filename in matrices]
    with Pool(cpu_count()) as pool:
        results = pool.map(calculate_matrix_properties, tasks)
    return {filename: properties for filename, properties in results if properties}


def save_results(results, output_file):
    with open(output_file, 'w') as f:
        f.write(",".join([
            "Matrix Name", "M", "N", "NNZ", "Density", "Row Max", "Row Min", "Row Mean", "Row Median", "Row STD",
            "Column Max", "Column Min", "Column Mean", "Column Median", "Column STD", "Column Max / M", "Row Max / N",
            "Column STD / M", "Row STD / N", "Group", "Kind", "Psym", "Nsym"
        ]) + "\n")
        for filename, properties in results.items():
            f.write(f"{filename},{','.join(map(str, properties.values()))}\n")


def main():
    existing_matrices = load_existing_matrices()
    print(f"Loaded {len(existing_matrices)} existing matrices from file.")

    searched_matrices = search_matrices()
    print(f"Found {len(searched_matrices)} matrices matching the search criteria.")

    matrices_to_process = intersect_matrices(existing_matrices, searched_matrices)
    print(f"Processing {len(matrices_to_process)} matrices that exist in both lists.")

    results = process_matrices_parallel(MATRIX_DIRECTORY, matrices_to_process)
    save_results(results, FEATURES_OUTPUT_FILE)

    print(f"Matrix properties have been saved to {FEATURES_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
