import os
import numpy as np
import ssgetpy
from scipy.io import mmread
from scipy.sparse import csr_matrix
import concurrent.futures
import argparse
import multiprocessing

# Configuration
MATRIX_DIRECTORY = "/home/hazardous/Desktop/HyPerTune-tools/generated-matrices"
FEATURES_OUTPUT_FILE = "synthetic_matrix_features.csv"

def find_expanded_matrices():
    """Find all expanded matrices in the directory."""
    expanded_matrices = [os.path.join(MATRIX_DIRECTORY, f) for f in os.listdir(MATRIX_DIRECTORY) if f.endswith('.mtx')]
    expanded_matrices.sort()  # Sort by name (optional)
    return expanded_matrices

def fetch_original_matrix_info(matrix_name):
    """Fetch original matrix metadata (Group, Kind) from SuiteSparse using ssgetpy."""
    try:
        results = ssgetpy.search(name=matrix_name)  # Returns a list
        if results and len(results) > 0:
            matrix = results[0]  # Get the first matching result
            return matrix.group, matrix.kind
    except Exception as e:
        print(f"⚠ Error fetching metadata for {matrix_name}: {e}")
    return "Unknown", "Unknown"

def calculate_psym_nsym(matrix):
    """
    Calculate pattern symmetry (psym) and numerical symmetry (nsym) for a given sparse matrix.

    Definitions:
      - Pattern symmetry: The percent of off-diagonal entries that are mirrored across the diagonal.
        Each complete pair (i.e. both A[i,j] and A[j,i] exist) counts as two symmetric entries.
      - Numerical symmetry: Among off-diagonal entries that are mirrored (i.e. complete pairs),
        the percent that have identical numeric values.

    Returns:
        psym, nsym
    """
    # Only defined for square matrices.
    n = matrix.shape[0]
    if n != matrix.shape[1]:
        return 0.0, 0.0

    # Convert to COO format for easier iteration.
    A = matrix.tocoo()

    # Exclude diagonal entries.
    offdiag_mask = A.row != A.col
    rows = A.row[offdiag_mask]
    cols = A.col[offdiag_mask]
    data = A.data[offdiag_mask]

    # Group entries by unordered pair (min(i,j), max(i,j)).
    pairs = {}
    for i, j, v in zip(rows, cols, data):
        key = (min(i, j), max(i, j))
        if key not in pairs:
            pairs[key] = [None, None]
        if i < j:
            pairs[key][0] = v
        else:
            pairs[key][1] = v

    total_keys = len(pairs)  # each key represents one unique off-diagonal position (could be 1 or 2 entries)
    complete_count = 0      # keys where both entries exist
    matching_numeric = 0    # complete keys where the values match

    for key, (v1, v2) in pairs.items():
        if v1 is not None and v2 is not None:
            complete_count += 1
            if np.isclose(v1, v2, atol=1e-64):
                matching_numeric += 1

    # Total off-diagonal entries:
    # Each complete pair contributes 2 entries; each incomplete pair contributes 1.
    total_offdiag = 2 * complete_count + (total_keys - complete_count)

    # Pattern symmetry: mirrored entries / total off-diagonals.
    psym = (2 * complete_count / total_offdiag) if total_offdiag > 0 else 0.0

    # Numerical symmetry: only consider complete pairs.
    nsym = (matching_numeric / complete_count) if complete_count > 0 else 0.0

    return psym, nsym


def calculate_matrix_properties(matrix, filename):
    """Calculate properties of a given sparse matrix."""
    base_name = os.path.basename(filename).replace('.mtx', '').split('_expansion')[0]
    group, kind = fetch_original_matrix_info(base_name)

    properties = {
        'Matrix Name': os.path.basename(filename).replace('.mtx', ''),
        'M': matrix.shape[0],
        'N': matrix.shape[1],
        'NNZ': matrix.nnz,
        'Density': matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    }

    row_nnz = matrix.getnnz(axis=1)
    col_nnz = matrix.getnnz(axis=0)

    # Calculate psym and nsym
    psym, nsym = calculate_psym_nsym(matrix)

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
        'Column Max / M': np.max(col_nnz) / properties['M'] if properties['M'] else "N/A",
        'Row Max / N': np.max(row_nnz) / properties['N'] if properties['N'] else "N/A",
        'Column STD / M': np.std(col_nnz) / properties['M'] if properties['M'] else "N/A",
        'Row STD / N': np.std(row_nnz) / properties['N'] if properties['N'] else "N/A",
        'Group': group,
        'Kind': kind,
        'Psym': psym,
        'Nsym': nsym
    })

    return properties

def process_matrix(filepath):
    """Process a single matrix and return its properties."""
    try:
        print(f"📌 Processing {filepath} on process {os.getpid()}...")
        matrix = mmread(filepath)
        sparse_matrix = csr_matrix(matrix)
        return calculate_matrix_properties(sparse_matrix, filepath)
    except Exception as e:
        print(f"❌ Skipping {filepath} due to error: {e}")
        return None

def process_expanded_matrices(max_workers):
    """Find, process, and save properties of all expanded matrices."""
    expanded_files = find_expanded_matrices()

    if not expanded_files:
        print("⚠ No expanded matrices found.")
        return

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(process_matrix, expanded_files):
            if result:
                results.append(result)

    save_results(results)

def save_results(results):
    """Save processed matrix properties to a CSV file."""
    if not results:
        print("⚠ No results to save.")
        return

    with open(FEATURES_OUTPUT_FILE, 'w') as f:
        headers = results[0].keys()
        f.write(",".join(headers) + "\n")
        for properties in results:
            f.write(",".join(map(str, properties.values())) + "\n")

    print(f"✅ Matrix properties saved to {FEATURES_OUTPUT_FILE}")

def main():
    """Main function to process expanded matrices."""
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Process sparse matrices with parallelization.")
    parser.add_argument('cpu_count', type=int, help="Number of CPU cores to use (max workers).")
    args = parser.parse_args()

    process_expanded_matrices(args.cpu_count)

if __name__ == "__main__":
    main()
