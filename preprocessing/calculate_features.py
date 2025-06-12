import csv
import os
import sys
from collections import defaultdict
from multiprocessing import cpu_count, get_context

import numpy as np
import scipy.sparse.linalg as splinalg
import ssgetpy
from scipy.io import mmread

MATRIX_DIRECTORY = "../matrices/synthetic_matrices"
ORIGINAL_DIRECTORY = "../matrices/matrices"
FEATURES_OUTPUT_FILE = "../data/processed/synthetic_matrix_features.csv"

os.makedirs(ORIGINAL_DIRECTORY, exist_ok=True)


def load_matrix(path):
    return mmread(path).tocsr()


# def list_all_mtx_files():
#     """List all MTX files in the synthetic matrix directory."""
#     return sorted(
#         [
#             os.path.join(MATRIX_DIRECTORY, f)
#             for f in os.listdir(MATRIX_DIRECTORY)
#             if f.endswith(".mtx")
#         ]
#     )


def list_all_mtx_files():
    """List all MTX files in both the synthetic and original directories."""
    synthetic_files = sorted(
        [
            os.path.join(MATRIX_DIRECTORY, f)
            for f in os.listdir(MATRIX_DIRECTORY)
            if f.endswith(".mtx")
        ]
    )
    original_files = sorted(
        [
            os.path.join(ORIGINAL_DIRECTORY, f)
            for f in os.listdir(ORIGINAL_DIRECTORY)
            if f.endswith(".mtx")
        ]
    )
    return synthetic_files + original_files


def read_generation_time(matrix_filepath):
    with open(matrix_filepath, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line.startswith("% Generated in "):
                tokens = line.split()
                if len(tokens) >= 4:
                    return float(tokens[3])
            # stop searching
            if i == 9:
                break
    return -1.0


# def read_generation_time(matrix_filepath):
#     with open(matrix_filepath, "r") as f:
#         for line in f:
#             print("DEBUG: Line:", repr(line))
#             if line.lstrip().startswith(
#                 "% Generated in "
#             ):  # lstrip removes leading spaces
#                 tokens = line.strip().split()
#                 if len(tokens) >= 4:
#                     print("DEBUG: Tokens:", tokens)
#                     return float(tokens[3])
#     return np.nan


def parse_base_name(filename):
    """Extract original matrix name from a synthetic variant filename."""
    name = os.path.basename(filename).replace(".mtx", "")
    if "_expansion" in name:
        return name.split("_expansion")[0]
    return name


# def ensure_original_matrix(name):
#     """Download the original matrix if not already present."""
#     dest_path = os.path.join(ORIGINAL_DIRECTORY, f"{name}.mtx")
#     if not os.path.exists(dest_path):
#         print(f"{dest_path} does not exist locally, exiting...")
#         sys.exit(0)
#     return dest_path


def calculate_psym_nsym(matrix):
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

    total_keys = len(
        pairs
    )  # each key represents one unique off-diagonal position (could be 1 or 2 entries)
    complete_count = 0  # keys where both entries exist
    matching_numeric = 0  # complete keys where the values match

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


def calculate_bandwidth_and_total_profile(matrix):
    # Convert matrix to coordinate format for easy access to nonzero indices.
    coo = matrix.tocoo()
    rows = coo.row
    cols = coo.col
    n_rows, n_cols = matrix.shape

    # Bandwidth is defined as max(|i - j|) for all nonzero positions.
    bandwidth = np.max(np.abs(rows - cols)) if rows.size else 0

    # For the total profile, we compute the row-wise lower and upper profiles.
    # Lower profile: For each row i, difference between i and the minimum column index with a nonzero.
    # Upper profile: For each row i, difference between the maximum column index with a nonzero and i.

    # Initialize dictionaries for the first and last nonzero column indices for each row.
    # If no nonzero exists in a row, we keep default values.
    min_nonzero = {
        i: n_cols for i in range(n_rows)
    }  # default: no entry in row (n_cols is "infinite")
    max_nonzero = {
        i: -1 for i in range(n_rows)
    }  # default: no entry in row (-1 means no nonzero)

    # Update the dictionaries with the actual values.
    for i, j in zip(rows, cols):
        if j < min_nonzero[i]:
            min_nonzero[i] = j
        if j > max_nonzero[i]:
            max_nonzero[i] = j

    # Now accumulate the profile for each row.
    lower_profile = 0
    upper_profile = 0
    for i in range(n_rows):
        # Only if there is at least one nonzero entry in the row.
        if min_nonzero[i] < n_cols:
            lower_profile += i - min_nonzero[i]
        if max_nonzero[i] >= 0:
            upper_profile += max_nonzero[i] - i

    total_profile = lower_profile + upper_profile
    return bandwidth, total_profile


def entropy(x):
    p = x / np.sum(x)
    return -np.sum(p * np.log2(p + 1e-12))


# Here we assume synthetic matrices contain the keyword `expansion` in their filename
# This is not the case anymore and may be changed according to the synthetic matrix generation naming style
# The current code is "glue" code that provides backward compatibility with our current matrices from an -
# earlier version of MatGen https://github.com/He-Is-HaZaRdOuS/MatGen where we made breaking changes to the API at some point
def extract_method_name(matrix_name):
    if "_expansion_" in matrix_name:
        after = matrix_name.split("_expansion")[1]
        for method in ["double", "increment", "half", "decrement"]:
            if method in after:
                return after.split(f"_{method}")[0]
        return "unknown"
    else:
        return "original"


# Need to update to detect expansion size
def extract_operation_name(matrix_name):
    for method in ["double", "increment", "half", "decrement"]:
        if method in matrix_name:
            return method
    return "original"


def get_match_nnz_flag(matrix_filename):
    # Example filename:
    # mcfe_expansion_scale_sparse_matrix_wavelet_increment_766x766_matchnnz_NA_nnz43021.mtx

    parts = matrix_filename.split("_")
    if "matchnnz" in parts:
        idx = parts.index("matchnnz")
        if idx + 1 < len(parts):
            flag = parts[idx + 1]
            if flag == "True":
                return flag
            elif flag == "False":
                return flag
            # elif flag == "NA":
            #     return "Not Available"
    return "original"  # default to original if not found


def calculate_features(matrix, matrix_name, original_path, group, kind):
    M, N = matrix.shape
    method = extract_method_name(matrix_name)
    operation = extract_operation_name(matrix_name)
    NNZ = matrix.nnz
    row_nnz = matrix.getnnz(axis=1)
    col_nnz = matrix.getnnz(axis=0)
    psym, nsym = calculate_psym_nsym(matrix)
    bandwidth, profile = calculate_bandwidth_and_total_profile(matrix)

    features = {
        "Variant Name": matrix_name,
        "Matrix Name": matrix_name.split("_expansion")[0],
        "Operation": operation,
        "Method": method,
        "Generation Time(S)": read_generation_time(original_path),
        "Match NNZ": get_match_nnz_flag(matrix_name),
        "Group": group,
        "Kind": kind,
        "M": M,
        "N": N,
        "NNZ": NNZ,
        "Density": NNZ / (M * N),
        "Row NNZ Max": np.max(row_nnz),
        "Row NNZ Min": np.min(row_nnz),
        "Row NNZ Mean": np.mean(row_nnz),
        "Row NNZ Median": np.median(row_nnz),
        "Row NNZ STD": np.std(row_nnz),
        "Column NNZ Max": np.max(col_nnz),
        "Column NNZ Min": np.min(col_nnz),
        "Column NNZ Mean": np.mean(col_nnz),
        "Column NNZ Median": np.median(col_nnz),
        "Column NNZ STD": np.std(col_nnz),
        "Row NNZ Min / N": np.min(row_nnz) / N if N else np.nan,
        "Column NNZ Min / M": np.min(col_nnz) / M if M else np.nan,
        "Row NNZ Max / N": np.max(row_nnz) / N if N else np.nan,
        "Column NNZ Max / M": np.max(col_nnz) / M if M else np.nan,
        "Row NNZ STD / N": np.std(row_nnz) / N if N else np.nan,
        "Column NNZ STD / M": np.std(col_nnz) / M if M else np.nan,
        "Psym": psym,
        "Nsym": nsym,
        "Bandwidth": bandwidth,
        "Bandwidth / N": bandwidth / N,
        "Bandwidth STD": np.std(bandwidth),
        "Profile": profile,
        "Profile / N": profile / N,
    }

    # ––– value stats ––––––––––––––––––––––––––––––––––––––––––––––––
    data = matrix.data if matrix.data.size else np.array([0])
    features |= {
        "value_min": float(data.min()),
        "value_max": float(data.max()),
        "value_avg": float(data.mean()),
        "value_std": float(data.std()),
    }

    # ––– per‑row detailed stats ––––––––––––––––––––––––––––––––––––
    csr = matrix
    row_min = np.zeros(M)
    row_max = np.zeros(M)
    row_mean = np.zeros(M)
    row_std = np.zeros(M)
    row_median = np.zeros(M)

    for i in range(M):
        seg = csr.data[csr.indptr[i] : csr.indptr[i + 1]]
        if seg.size:
            row_min[i] = seg.min()
            row_max[i] = seg.max()
            row_mean[i] = seg.mean()
            row_std[i] = seg.std()
            row_median[i] = np.median(seg)

    features |= {
        "row_min_min": float(row_min.min()),
        "row_min_max": float(row_min.max()),
        "row_min_mean": float(row_min.mean()),
        "row_min_std": float(row_min.std()),
        "row_max_min": float(row_max.min()),
        "row_max_max": float(row_max.max()),
        "row_max_mean": float(row_max.mean()),
        "row_max_std": float(row_max.std()),
        "row_mean_min": float(row_mean.min()),
        "row_mean_max": float(row_mean.max()),
        "row_mean_mean": float(row_mean.mean()),
        "row_mean_std": float(row_mean.std()),
        "row_std_min": float(row_std.min()),
        "row_std_max": float(row_std.max()),
        "row_std_mean": float(row_std.mean()),
        "row_std_std": float(row_std.std()),
        "row_median_min": float(row_median.min()),
        "row_median_max": float(row_median.max()),
        "row_median_mean": float(row_median.mean()),
        "row_median_std": float(row_median.std()),
    }

    # ––– per‑column detailed stats ––––––––––––––––––––––––––––––––––
    csc = matrix.tocsc()
    col_min = np.zeros(N)
    col_max = np.zeros(N)
    col_mean = np.zeros(N)
    col_std = np.zeros(N)
    col_median = np.zeros(N)

    for j in range(N):
        seg = csc.data[csc.indptr[j] : csc.indptr[j + 1]]
        if seg.size:
            col_min[j] = seg.min()
            col_max[j] = seg.max()
            col_mean[j] = seg.mean()
            col_std[j] = seg.std()
            col_median[j] = np.median(seg)

    features |= {
        "col_min_min": float(col_min.min()),
        "col_min_max": float(col_min.max()),
        "col_min_mean": float(col_min.mean()),
        "col_min_std": float(col_min.std()),
        "col_max_min": float(col_max.min()),
        "col_max_max": float(col_max.max()),
        "col_max_mean": float(col_max.mean()),
        "col_max_std": float(col_max.std()),
        "col_mean_min": float(col_mean.min()),
        "col_mean_max": float(col_mean.max()),
        "col_mean_mean": float(col_mean.mean()),
        "col_mean_std": float(col_mean.std()),
        "col_std_min": float(col_std.min()),
        "col_std_max": float(col_std.max()),
        "col_std_mean": float(col_std.mean()),
        "col_std_std": float(col_std.std()),
        "col_median_min": float(col_median.min()),
        "col_median_max": float(col_median.max()),
        "col_median_mean": float(col_median.mean()),
        "col_median_std": float(col_median.std()),
    }

    # ––– diagonal distances –––––––––––––––––––––––––––––––––––––––
    row_idx, col_idx = matrix.nonzero()
    if row_idx.size:
        dist = np.abs(row_idx - col_idx)
        nnz_diagonal = np.count_nonzero(row_idx == col_idx)
        nnz_off_diagonal = NNZ - nnz_diagonal
        features["avg_distance_to_diagonal"] = float(dist.mean())
        features["avg_distance_to_diagonal / N"] = (
            features["avg_distance_to_diagonal"] / N
        )
        features["num_diagonals_with_nonzeros"] = int(np.unique(dist).size)
        features["nnz_bandwidth_std"] = float(dist.std()) if dist.size else 0.0
        features["nnz_diagonal"] = int(nnz_diagonal)
        features["nnz_off_diagonal"] = int(nnz_off_diagonal)

    else:
        features["avg_distance_to_diagonal"] = 0.0
        features["avg_distance_to_diagonal / N"] = 0.0
        features["num_diagonals_with_nonzeros"] = 0
        features["nnz_bandwidth_std"] = 0.0
        features["nnz_diagonal"] = 0.0
        features["nnz_off_diagonal"] = 0.0

    # ––– structural unsymmetry ––––––––––––––––––––––––––––––––––––
    unsym = set(zip(row_idx, col_idx)) - set(zip(col_idx, row_idx))
    features["num_structurally_unsymmetric_elements"] = int(len(unsym))

    # ––– norms ––––––––––––––––––––––––––––––––––––––––––––––––––––
    try:
        features["norm_1"] = float(splinalg.norm(matrix, 1))
        features["norm_inf"] = float(splinalg.norm(matrix, np.inf))
        features["frobenius_norm"] = float(splinalg.norm(matrix))
    except Exception:
        pass

    # ––– 1‑norm condition estimate ––––––––––––––––––––––––––––––––
    try:
        features["estimated_condition_number"] = float(
            splinalg.onenormest(matrix)
        )
    except Exception:
        features["estimated_condition_number"] = None

    # Sparsity profile
    features["num_empty_rows"] = int(np.sum(row_nnz == 0))
    features["num_empty_cols"] = int(np.sum(col_nnz == 0))

    # Sparsity skew: compare nonzero spread across rows and cols
    features["row_sparsity_skew"] = float(
        row_nnz.std() / (row_nnz.mean() + 1e-8)
    )
    features["col_sparsity_skew"] = float(
        col_nnz.std() / (col_nnz.mean() + 1e-8)
    )

    # Row/Col entropy
    features["row_nnz_entropy"] = float(entropy(row_nnz))
    features["col_nnz_entropy"] = float(entropy(col_nnz))

    return features


def fetch_matrix_metadata(name):
    try:
        results = ssgetpy.search(name=name)
        if results:
            m = results[0]
            return m.group, m.kind
    except Exception:
        pass
    return "Unknown", "Unknown"


def group_expansion_by_base_name(paths):
    groups = defaultdict(list)
    for path in paths:
        base = parse_base_name(path)
        groups[base].append(path)
    return groups


def process_group(base_name_and_paths):
    sys.stdout = open("output.log", "a", buffering=1)
    sys.stderr = sys.stdout  # redirect errors too

    base_name, syn_paths = base_name_and_paths
    print(f"Starting group: {base_name}")
    group, kind = fetch_matrix_metadata(base_name)
    local_rows = []

    # try:
    #     original_path = ensure_original_matrix(base_name)
    #     if original_path:
    #         mat = load_matrix(original_path)
    #         group, kind = fetch_matrix_metadata(base_name)
    #         features = calculate_features(
    #             mat, base_name, original_path, group, kind
    #         )
    #         local_rows.append(features)
    #         print(f"[INFO] Calculated features for original matrix {base_name}")
    # except Exception as e:
    #     print(f"[ERROR] [{base_name}] Failed original: {e}")

    for syn_path in syn_paths:
        # if syn_path == original_path:
        #     continue
        try:
            print(f"Loading {syn_path}")
            mat = load_matrix(syn_path)
            print(f"Starting feature calculation for {syn_path}")
            features = calculate_features(
                mat,
                os.path.basename(syn_path).replace(".mtx", ""),
                syn_path,
                group,
                kind,
            )
            local_rows.append(features)
            print(f"[INFO] Calculated features for matrix {syn_path}")
        except Exception as e:
            print(f"[ERROR] [{base_name}] Failed {syn_path}: {e}")

    print(f"Done group: {base_name}")
    return local_rows


def main():
    sys.stdout = open("output.log", "a", buffering=1)
    sys.stderr = sys.stdout  # redirect errors too

    all_expansion_paths = list_all_mtx_files()
    grouped = group_expansion_by_base_name(all_expansion_paths)

    print(f"[INFO] Processing {len(grouped)} groups using multiprocessing...")

    ctx = get_context("spawn")  # Use spawn for safety
    with ctx.Pool(processes=cpu_count()) as pool:
        all_results = pool.map(process_group, grouped.items())

    # Flatten results
    rows = [r for group in all_results for r in group if r]

    if rows:
        with open(FEATURES_OUTPUT_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] Saved features to {FEATURES_OUTPUT_FILE}")
    else:
        print("[INFO] No features extracted.")


if __name__ == "__main__":
    sys.stdout = open("output.log", "w", buffering=1)  # Line-buffered output
    main()
    sys.exit(0)
