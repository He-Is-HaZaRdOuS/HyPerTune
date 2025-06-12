import os
import random
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor

import ssgetpy

# Configuration
DOWNLOAD_DIR = "../matrices"
NUM_WORKERS = 8
BASE_URL = "https://suitesparse-collection-website.herokuapp.com/MM"
MAX_RETRIES = 3

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def get_base_name(matrix):
    # Remove trailing digits, e.g., "cage4" -> "cage"
    return "".join(filter(lambda c: not c.isdigit(), matrix.name))


def get_downloaded_matrix_names():
    """Get names of matrices already downloaded by checking .mtx files in the download directory."""
    return {
        filename.replace(".mtx", "")
        for filename in os.listdir(DOWNLOAD_DIR)
        if filename.endswith(".mtx")
    }


def search_matrices():
    """Search for 500 <= rows <= 1000, unsymmetrical, exclude binary types, return 10."""
    results = ssgetpy.search(
        rowbounds=(500, 1000),
        limit=1000,  # Fetch enough to filter down
    )

    filtered = [
        m
        for m in results
        if m.psym < 1.0  # Non-perfectly symmetrical
        and m.rows == m.cols  # Square
        and m.kind  # Has valid kind
        and "binary" not in m.kind.lower()  # Non-binary Rutherford-Boeing type
    ]

    return filtered


def download_and_extract(matrix, downloaded_names):
    """Download a matrix and extract its .mtx file."""
    if matrix.name in downloaded_names:
        print(f"Skipping {matrix.name}: Already downloaded.")
        return

    tar_url = f"{BASE_URL}/{matrix.group}/{matrix.name}.tar.gz"
    tar_path = os.path.join(DOWNLOAD_DIR, f"{matrix.name}.tar.gz")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(
                f"Downloading {matrix.name} (Attempt {attempt}) from {tar_url}..."
            )
            subprocess.run(["wget", "-q", "-O", tar_path, tar_url], check=True)

            with tarfile.open(tar_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".mtx"):
                        member.name = os.path.basename(member.name)
                        tar.extract(member, path=DOWNLOAD_DIR)
                        print(f"Extracted {matrix.name}.mtx")
                        break

            os.remove(tar_path)
            return

        except (
            subprocess.CalledProcessError,
            tarfile.TarError,
            Exception,
        ) as e:
            print(f"Error processing {matrix.name} on attempt {attempt}: {e}")
            if os.path.exists(tar_path):
                os.remove(tar_path)
            if attempt == MAX_RETRIES:
                print(
                    f"Failed to download or extract {matrix.name} after {MAX_RETRIES} attempts."
                )


def main():
    downloaded_names = get_downloaded_matrix_names()
    print(f"Detected {len(downloaded_names)} existing matrices in directory.")

    matrices = search_matrices()
    print(f"Found {len(matrices)} matrices matching criteria.")

    # Deduplicate by base name (e.g., avoid cage1, cage2, etc.)
    seen = set()
    unique_matrices = []
    for matrix in matrices:
        base = get_base_name(matrix)
        if base not in seen:
            seen.add(base)
            unique_matrices.append(matrix)

    print(f"{len(unique_matrices)} unique base matrices after filtering.")

    # Deterministic shuffle
    random.seed(42)
    random.shuffle(unique_matrices)

    selected = unique_matrices[:10]
    print(f"Selected {len(selected)} matrices for download.")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        executor.map(
            lambda matrix: download_and_extract(matrix, downloaded_names),
            selected,
        )


if __name__ == "__main__":
    main()
