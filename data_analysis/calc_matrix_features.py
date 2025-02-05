import os
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix, triu

# Configuration
MATRIX_DIRECTORY = "/home/hazardous/Desktop/HyPerTune-tools/matrices"  # Directory where matrices are stored
FEATURES_OUTPUT_FILE = "matrix_features.csv"  # File to store matrix features

def load_matrices_from_directory(directory):
    """Load matrix files from a given directory."""
    matrix_files = []

    # Iterate over all files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith(".mtx"):  # Check if the file is a .mtx matrix file
            matrix_files.append(file_name)

    return matrix_files

def calculate_psym_nsym_and_kind(sparse_matrix):
    """Calculate Psym, Nsym, and Kind for the given sparse matrix."""
    # Psym (symmetric entries): check if a[i,j] == a[j,i]
    symmetric_matrix = sparse_matrix + sparse_matrix.T  # Matrix + its transpose
    psym = np.sum(symmetric_matrix != 0)  # Count symmetric non-zero elements

    # Nsym (non-zero symmetric elements)
    upper_triangular = triu(sparse_matrix)  # Upper triangular part of the matrix
    nsym = np.sum(upper_triangular != 0)  # Count non-zero elements in the upper triangle

    # Kind: Check if the matrix is symmetric
    kind = 'Symmetric' if (sparse_matrix != sparse_matrix.T).nnz == 0 else 'Asymmetric'

    return psym, nsym, kind

def calculate_features(matrix_path):
    """Calculate missing features for a given matrix file."""
    try:
        # Load the matrix
        matrix = mmread(matrix_path)  # Load .mtx file
        sparse_matrix = csr_matrix(matrix)  # Convert to sparse format

        # Check for inconsistent shapes
        rows, cols = sparse_matrix.shape
        if rows != cols:
            print(f"Skipping {matrix_path} due to inconsistent shape: {rows}x{cols}")
            # Skip psym, nsym, kind calculations and proceed with other features
            psym, nsym, kind = None, None, None
        else:
            # Calculate Psym, Nsym, Kind if the matrix is square
            psym, nsym, kind = calculate_psym_nsym_and_kind(sparse_matrix)

        nnz = sparse_matrix.nnz
        density = nnz / (rows * cols)

        # Row features
        row_sums = sparse_matrix.getnnz(axis=1)  # Non-zero elements per row
        row_max = np.max(row_sums)
        row_min = np.min(row_sums)
        row_mean = np.mean(row_sums)
        row_median = np.median(row_sums)
        row_stddev = np.std(row_sums)

        # Column features
        col_sums = sparse_matrix.getnnz(axis=0)  # Non-zero elements per column
        col_max = np.max(col_sums)
        col_min = np.min(col_sums)
        col_mean = np.mean(col_sums)
        col_median = np.median(col_sums)
        col_stddev = np.std(col_sums)

        # Compile all features
        features = {
            "Rows": rows,
            "Cols": cols,
            "NNZ": nnz,
            "Density": density,
            "Row Max": row_max,
            "Row Min": row_min,
            "Row Mean": row_mean,
            "Row Median": row_median,
            "Row STD": row_stddev,
            "Column Max": col_max,
            "Column Min": col_min,
            "Column Mean": col_mean,
            "Column Median": col_median,
            "Column STD": col_stddev,
            "Psym": psym,
            "Nsym": nsym,
            "Kind": kind
        }

        return features

    except Exception as e:
        print(f"Error processing matrix at {matrix_path}: {e}")
        return None

def find_and_extract_features(matrix_files, matrix_directory):
    """Find features for matrices and calculate missing features."""
    with open(FEATURES_OUTPUT_FILE, "w") as f:
        f.write("Matrix Name,Group,M,NNZ,Density,Row Max,Row Min,Row Mean,Row Median,Row STD,Column Max,Column Min,Column Mean,Column Median,Column STD,Psym,Nsym,Kind\n")

        for matrix_name in matrix_files:
            matrix_path = os.path.join(matrix_directory, matrix_name)  # Path to the matrix file
            if os.path.exists(matrix_path):
                # Calculate features
                features = calculate_features(matrix_path)
                if features:
                    # Write features to file
                    f.write(f"{matrix_name.split('.')[0]},{','.join(map(str, features.values()))}\n")
            else:
                print(f"Matrix file for {matrix_name} not found.")

def main():
    # Load matrix files from the directory
    matrix_files = load_matrices_from_directory(MATRIX_DIRECTORY)
    print(f"Found {len(matrix_files)} matrix files in directory.")

    # Find and extract features for matrices in the directory
    print("Extracting matrix features...")
    find_and_extract_features(matrix_files, MATRIX_DIRECTORY)

if __name__ == "__main__":
    main()
