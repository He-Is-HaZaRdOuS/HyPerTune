import os
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix
import ssgetpy

# Configuration
MATRIX_DIRECTORY = "/home/hazardous/Desktop/HyPerTune-tools/matrices"
EXISTING_MATRICES_FILE = "/home/hazardous/Desktop/HyPerTune-tools/matrix_list.txt"
FEATURES_OUTPUT_FILE = "matrix_features.csv"

def load_existing_matrices():
    """Load the list of already downloaded matrices from a file."""
    if not os.path.exists(EXISTING_MATRICES_FILE):
        return set()

    with open(EXISTING_MATRICES_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())

def search_matrices():
    """Search for matrices with row count >= 10000 and filter only unsymmetric ones."""
    results = ssgetpy.search(rowbounds=(10000, None), limit=1000)
    return {matrix.name: matrix for matrix in results if matrix.psym < 1.0}

def intersect_matrices(existing_matrices, searched_matrices):
    """Find the intersection of matrices between the file list and search results, ignoring file extensions."""
    # Remove extensions from the existing matrix names
    existing_matrices_normalized = {os.path.splitext(name)[0] for name in existing_matrices}

    # Normalize searched matrices and find the intersection
    return {name: searched_matrices[name] for name in searched_matrices if name in existing_matrices_normalized}

def calculate_matrix_properties(matrix, matrix_search):
    """Calculate matrix properties, including new features and handling missing values."""
    properties = {}
    properties['M'] = matrix.shape[0]
    properties['N'] = matrix.shape[1]
    properties['NNZ'] = matrix.nnz
    Density = matrix.nnz / (matrix.shape[0])
    Density /= (matrix.shape[1])

    properties['Density'] = Density

    row_nnz = matrix.getnnz(axis=1)
    col_nnz = matrix.getnnz(axis=0)

    properties['Row Max'] = np.max(row_nnz)
    properties['Row Min'] = np.min(row_nnz)
    properties['Row Mean'] = np.mean(row_nnz)
    properties['Row Median'] = np.median(row_nnz)
    properties['Row STD'] = np.std(row_nnz)

    properties['Column Max'] = np.max(col_nnz)
    properties['Column Min'] = np.min(col_nnz)
    properties['Column Mean'] = np.mean(col_nnz)
    properties['Column Median'] = np.median(col_nnz)
    properties['Column STD'] = np.std(col_nnz)

    # Additional calculated features
    properties['Column Max / M'] = properties['Column Max'] / properties['M'] if properties['M'] != 0 else "N/A"
    properties['Row Max / N'] = properties['Row Max'] / properties['N'] if properties['N'] != 0 else "N/A"
    properties['Column STD / M'] = properties['Column STD'] / properties['M'] if properties['M'] != 0 else "N/A"
    properties['Row STD / N'] = properties['Row STD'] / properties['N'] if properties['N'] != 0 else "N/A"

    properties['Group'] = matrix_search.group or "N/A"
    properties['Psym'] = matrix_search.psym if matrix_search.psym is not None else "N/A"
    properties['Nsym'] = matrix_search.nsym if matrix_search.nsym is not None else "N/A"
    properties['Kind'] = matrix_search.kind or "N/A"

    return properties

def process_matrices(directory, matrices):
    """Process only matrices that are in the intersected list."""
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith('.mtx'):
            filename_without_ext = filename.replace('.mtx', '')

            if filename_without_ext not in matrices:
                print(f"Skipping {filename}: Not in search results or matrix list")
                continue

            filepath = os.path.join(directory, filename)
            try:
                print(f"Reading matrix: {filename}")
                matrix = mmread(filepath)
                sparse_matrix = csr_matrix(matrix)

                matrix_2 = matrices[filename_without_ext]

                properties = calculate_matrix_properties(sparse_matrix, matrix_2)
                results[filename_without_ext] = properties
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")

    return results

def save_results(results, output_file):
    """Save processed matrix properties to CSV."""
    with open(output_file, 'w') as f:
        f.write("Matrix Name,M,N,NNZ,Density,Row Max,Row Min,Row Mean,Row Median,Row STD,"
                "Column Max,Column Min,Column Mean,Column Median,Column STD,"
                "Column Max / M,Row Max / N,Column STD / M,Row STD / N,Group,Psym,Nsym,Kind\n")

        for filename, properties in results.items():
            f.write(f"{filename},{','.join(map(str, properties.values()))}\n")

def main():
    """Main function to load, process, and save matrix data."""
    existing_matrices = load_existing_matrices()
    print(f"Loaded {len(existing_matrices)} existing matrices from file.")

    searched_matrices = search_matrices()
    print(f"Found {len(searched_matrices)} matrices matching the search criteria.")

    # Find intersection of existing matrices and search results
    matrices_to_process = intersect_matrices(existing_matrices, searched_matrices)
    print(f"Processing {len(matrices_to_process)} matrices that exist in both lists.")

    results = process_matrices(MATRIX_DIRECTORY, matrices_to_process)

    save_results(results, FEATURES_OUTPUT_FILE)

    print(f"Matrix properties have been saved to {FEATURES_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
