import matplotlib.pyplot as plt
from scipy.io import mmread
from scipy.sparse import csr_matrix
import sys

def visualize_matrix_market(filepath):
    """Reads a Matrix Market file and visualizes its sparsity pattern."""
    try:
        # Read the matrix
        matrix = mmread(filepath)

        # Convert to a sparse matrix (CSR format)
        sparse_matrix = csr_matrix(matrix)

        # Plot the sparsity pattern
        plt.figure(figsize=(8, 8))
        plt.spy(sparse_matrix, markersize=0.5)
        plt.title(f"Sparsity Pattern of {filepath}")
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.savefig("matrix_sparsity.png", dpi=300, bbox_inches='tight')
        print("Sparsity plot saved as matrix_sparsity.png")


    except Exception as e:
        print(f"Error reading matrix: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <matrix_file.mtx>")
    else:
        visualize_matrix_market(sys.argv[1])
