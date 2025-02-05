import os
import ssgetpy
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor

# Configuration
DOWNLOAD_DIR = "/home/hazardous/Desktop/HyPerTune-tools/matrices"  # Directory to store downloaded matrices and .mtx files
EXISTING_MATRICES_FILE = "existing_matrices.txt"  # File listing already downloaded matrices
NUM_WORKERS = 8  # Number of parallel downloads
BASE_URL = "https://suitesparse-collection-website.herokuapp.com/MM"
MAX_RETRIES = 3  # Number of retries for failed downloads

# Ensure the download directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def load_existing_matrices():
    """Load the list of already downloaded matrices from a file."""
    if not os.path.exists(EXISTING_MATRICES_FILE):
        return set()

    with open(EXISTING_MATRICES_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())

def search_matrices():
    """Search for matrices with row count >= 10000 and filter for unsymmetrical structure."""
    results = ssgetpy.search(
        rowbounds=(10000, None),  # Matrices with at least 10,000 rows
        limit=1000                # Maximum number of results to fetch
    )

    # Filter for unsymmetrical matrices
    unsymmetrical_matrices = [
        matrix for matrix in results
        if matrix.psym < 1.0  # Pattern symmetry < 1.0 indicates unsymmetrical
    ]

    return unsymmetrical_matrices

def download_and_extract(matrix, existing_matrices):
    """Download a matrix from SuiteSparse using wget/curl and extract the .mtx file."""
    if matrix.name in existing_matrices:
        print(f"Skipping {matrix.name}: Already downloaded.")
        return

    tar_url = f"{BASE_URL}/{matrix.group}/{matrix.name}.tar.gz"
    tar_path = os.path.join(DOWNLOAD_DIR, f"{matrix.name}.tar.gz")
    mtx_file = os.path.join(DOWNLOAD_DIR, f"{matrix.name}.mtx")

    # Retry mechanism for downloads and extractions
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Downloading {matrix.name} (Attempt {attempt}) from {tar_url}...")
            subprocess.run(["wget", "-q", "-O", tar_path, tar_url], check=True)

            # Extract the .mtx file
            with tarfile.open(tar_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".mtx"):
                        member.name = os.path.basename(member.name)  # Strip path structure
                        tar.extract(member, path=DOWNLOAD_DIR)
                        print(f"Extracted {mtx_file}")
                        break

            # Clean up the .tar.gz file
            os.remove(tar_path)

            # Add to existing matrices file
            with open(EXISTING_MATRICES_FILE, "a") as f:
                f.write(f"{matrix.name}\n")

            return  # Success, exit the retry loop

        except (subprocess.CalledProcessError, tarfile.TarError, Exception) as e:
            print(f"Error processing {matrix.name} on attempt {attempt}: {e}")

            # Remove the corrupted or incomplete files
            if os.path.exists(tar_path):
                os.remove(tar_path)

            if attempt == MAX_RETRIES:
                print(f"Failed to download or extract {matrix.name} after {MAX_RETRIES} attempts.")

def main():
    # Load existing matrices
    existing_matrices = load_existing_matrices()
    print(f"Loaded {len(existing_matrices)} existing matrices from file.")

    # Search for matrices
    matrices = search_matrices()
    print(f"Found {len(matrices)} matrices matching the criteria.")

    # Download matrices in parallel
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        executor.map(lambda matrix: download_and_extract(matrix, existing_matrices), matrices)

if __name__ == "__main__":
    main()
