import logging
import math
import os

import numpy as np
import pandas as pd
import psutil
import scipy.sparse.linalg as splinalg
import torch
from scipy.io import mmread
from scipy.sparse import csr_matrix
from torch import nn, optim

log_file = "output.log"

if os.path.exists(log_file):
    os.remove(log_file)

# Logging setup
logging.basicConfig(
    filename=log_file,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Parameters
MAX_TILES = 256  # max tiles per matrix
EMBED_DIM = 64
NUM_LAYERS = 1
EPOCHS = 5  # for LSTM classifier training
LR = 1e-3
CSV_PATH = "datasets/DISCRETIZED_matrix_data_dir_rec_6class.csv"
MATRIX_DIR = "/home/g7-2024/hypertune_matrices/all_matrices"
CACHE_DIR = "cache/tile_feats"
EMBED_DIR = "cache/embeddings"
BATCH_SIZE = 1

ALL_FEATURES = [
    "M",
    "N",
    "NNZ",
    "Density",
    "Row NNZ Max",
    "Row NNZ Min",
    "Row NNZ Mean",
    "Row NNZ Median",
    "Row NNZ STD",
    "Column NNZ Max",
    "Column NNZ Min",
    "Column NNZ Mean",
    "Column NNZ Median",
    "Column NNZ STD",
    "Row NNZ Min / N",
    "Column NNZ Min / M",
    "Row NNZ Max / N",
    "Column NNZ Max / M",
    "Row NNZ STD / N",
    "Column NNZ STD / M",
    "Psym",
    "Nsym",
    "Bandwidth",
    "Bandwidth / N",
    "Bandwidth STD",
    "Profile",
    "Profile / N",
    "value_min",
    "value_max",
    "value_avg",
    "value_std",
    "row_min_min",
    "row_min_max",
    "row_min_mean",
    "row_min_std",
    "row_max_min",
    "row_max_max",
    "row_max_mean",
    "row_max_std",
    "row_mean_min",
    "row_mean_max",
    "row_mean_mean",
    "row_mean_std",
    "row_std_min",
    "row_std_max",
    "row_std_mean",
    "row_std_std",
    "row_median_min",
    "row_median_max",
    "row_median_mean",
    "row_median_std",
    "col_min_min",
    "col_min_max",
    "col_min_mean",
    "col_min_std",
    "col_max_min",
    "col_max_max",
    "col_max_mean",
    "col_max_std",
    "col_mean_min",
    "col_mean_max",
    "col_mean_mean",
    "col_mean_std",
    "col_std_min",
    "col_std_max",
    "col_std_mean",
    "col_std_std",
    "col_median_min",
    "col_median_max",
    "col_median_mean",
    "col_median_std",
    "avg_distance_to_diagonal",
    "avg_distance_to_diagonal / N",
    "num_diagonals_with_nonzeros",
    "nnz_bandwidth_std",
    "nnz_diagonal",
    "nnz_off_diagonal",
    "num_structurally_unsymmetric_elements",
    "norm_1",
    "norm_inf",
    "frobenius_norm",
    "estimated_condition_number",
    "num_empty_rows",
    "num_empty_cols",
    "row_sparsity_skew",
    "col_sparsity_skew",
    "row_nnz_entropy",
    "col_nnz_entropy",
]

DROP_FEATURES = [
    "value_min",
    "value_max",
    "value_avg",
    "value_std",
    "row_min_min",
    "row_min_max",
    "row_min_mean",
    "row_min_std",
    "row_max_min",
    "row_max_max",
    "row_max_mean",
    "row_max_std",
    "row_mean_min",
    "row_mean_max",
    "row_mean_mean",
    "row_mean_std",
    "row_std_min",
    "row_std_max",
    "row_std_mean",
    "row_std_std",
    "row_median_min",
    "row_median_max",
    "row_median_mean",
    "row_median_std",
    "col_min_min",
    "col_min_max",
    "col_min_mean",
    "col_min_std",
    "col_max_min",
    "col_max_max",
    "col_max_mean",
    "col_max_std",
    "col_mean_min",
    "col_mean_max",
    "col_mean_mean",
    "col_mean_std",
    "col_std_min",
    "col_std_max",
    "col_std_mean",
    "col_std_std",
    "col_median_min",
    "col_median_max",
    "col_median_mean",
    "col_median_std",
    "norm_1",
    "norm_inf",
    "frobenius_norm",
    "estimated_condition_number",
    "num_empty_rows",
    "num_empty_cols",
]

# Determine which features to include
INCLUDED_FEATURES = [f for f in ALL_FEATURES if f not in DROP_FEATURES]

FEATURE_COUNT = len(INCLUDED_FEATURES)
logger.info(f"Included features ({FEATURE_COUNT}): {INCLUDED_FEATURES}")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EMBED_DIR, exist_ok=True)


# Dataset
class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        df = df[
            df["Matrix"].apply(
                lambda n: os.path.isfile(os.path.join(MATRIX_DIR, n + ".mtx"))
            )
        ]
        target_columns = [
            col
            for col in df.columns
            if set(df[col].dropna().unique()) <= {0, 1}
            and (col.startswith("dir") or col.startswith("rec"))
        ]
        self.num_classes = len(target_columns)
        self.names = df["Matrix"].tolist()
        self.labels = torch.tensor(
            df[target_columns].values, dtype=torch.float32
        )
        print(
            f"[INFO] Init'd dataset loader. Detected {self.num_classes} classes."
        )

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return self.names[idx], self.labels[idx]


# Adaptive tile size: ensure total tiles <= MAX_TILES
# Aim for a grid of at most sqrt(MAX_TILES) per side
def calc_tile_size(M, N):
    # number of tiles per side target
    k = max(1, int(math.sqrt(MAX_TILES)))
    # compute tile size so that you get at most k tiles along each dimension
    tile_x = math.ceil(M / k)
    tile_y = math.ceil(N / k)
    # take the max to guarantee tile grid <= k x k
    tile = max(tile_x, tile_y)
    return tile


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


def calculate_features(matrix):
    M, N = matrix.shape
    NNZ = matrix.nnz
    row_nnz = matrix.getnnz(axis=1)
    col_nnz = matrix.getnnz(axis=0)
    psym, nsym = calculate_psym_nsym(matrix)
    bandwidth, profile = calculate_bandwidth_and_total_profile(matrix)

    features = {
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
        # features["estimated_condition_number"] = None
        features["estimated_condition_number"] = 0

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


# Compute features for one tile
def featurize_tile(sub: csr_matrix):
    features = calculate_features(sub)
    include = None
    filtered_keys = [
        k
        for k in features
        if (include is None or k in include)
        and (DROP_FEATURES is None or k not in DROP_FEATURES)
    ]
    feature_array = np.array(
        [features[k] for k in filtered_keys], dtype=np.float64
    )
    return feature_array
    # feature_names = list(features.keys())  # preserves insertion order (Python 3.7+)
    # feature_values = np.array([features[k] for k in feature_names], dtype=np.float64)


# Positional encoding for one tile index
def sinusoidal_pos_enc_single(xy, dim):
    x, y = xy
    vec = []
    for k in range(dim // 4):
        for coord in (x, y):
            angle = coord / (10000 ** (2 * k / dim))
            vec += [math.sin(angle), math.cos(angle)]
    return np.array(vec, dtype=np.float32)


# Cache tile features with metadata
# Always save both feats and pos arrays; at load, use_pos determines concatenation
CACHE_KEY_FEATS = "feats"
CACHE_KEY_POS = "pos"
CACHE_KEY_TILE = "tile_size"
CACHE_KEY_SHAPE = "orig_shape"
CACHE_KEY_NAME = "matrix_name"


def get_tile_features(name, use_pos: bool):
    cache_file = os.path.join(CACHE_DIR, f"{name}_tiles.npz")
    if os.path.exists(cache_file):
        data = np.load(cache_file, mmap_mode="r")
        feats = data[CACHE_KEY_FEATS]
        pos = data[CACHE_KEY_POS]
        tile = int(data[CACHE_KEY_TILE])
        shape = tuple(data[CACHE_KEY_SHAPE])
        # on load, selectively include pos
        return (
            feats,
            (
                pos
                if use_pos
                else np.empty((feats.shape[0], 0), dtype=np.float32)
            ),
            tile,
            shape,
        )

    # build from scratch
    mat = csr_matrix(mmread(os.path.join(MATRIX_DIR, name + ".mtx")))
    M, N = mat.shape
    tile = calc_tile_size(M, N)
    t_x, t_y = math.ceil(M / tile), math.ceil(N / tile)
    feats, pos = [], []
    pos_dim = FEATURE_COUNT * 2

    for i in range(t_x):
        for j in range(t_y):
            sub = mat[i * tile : (i + 1) * tile, j * tile : (j + 1) * tile]
            feats.append(
                featurize_tile(sub)
                if sub.nnz
                else np.zeros(FEATURE_COUNT, dtype=np.float32)
            )
            # always compute pos
            pos.append(sinusoidal_pos_enc_single((i, j), pos_dim))

    feats = np.vstack(feats)
    pos_arr = np.vstack(pos)

    # save both vectors and metadata
    np.savez_compressed(
        cache_file,
        feats=feats,
        pos=pos_arr,
        tile_size=tile,
        orig_shape=(M, N),
        matrix_name=name,
    )
    logger.info(
        f"Cached {name}: tile={tile}, shape=({M},{N}), tiles={len(feats)}"
    )

    # return with selective concatenation
    return (
        feats,
        (
            pos_arr
            if use_pos
            else np.empty((feats.shape[0], 0), dtype=np.float32)
        ),
        tile,
        (M, N),
    )


# LSTM + classifier with sigmoid
class LSTMWithClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, EMBED_DIM, NUM_LAYERS, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(EMBED_DIM, 2), nn.Sigmoid())

    def forward(self, feats):
        out, (h, c) = self.lstm(feats)
        return self.classifier(h[-1]), h[-1]


class StackedBiLSTMClassifier(nn.Module):
    def __init__(self, in_dim, num_class):
        super().__init__()
        self.conv1d = nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(
            in_dim,
            EMBED_DIM,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(EMBED_DIM * 2, num_class)
        # self.classifier = nn.Sequential(
        #     nn.Linear(EMBED_DIM * 2, 2), nn.Sigmoid()
        # )

    def forward(self, feats):
        feats = feats.transpose(1, 2)  # (batch, feat_dim, seq_len)
        feats = self.conv1d(feats).transpose(1, 2)
        out, (h, c) = self.lstm(feats)
        # forward final hidden state
        h_forward = h[-2, :, :]  # shape (batch, EMBED_DIM)

        # backward final hidden state
        h_backward = h[-1, :, :]  # shape (batch, EMBED_DIM)

        # concatenate along feature dimension
        h_final = torch.cat(
            (h_forward, h_backward), dim=1
        )  # shape (batch, 2*EMBED_DIM)

        return self.classifier(h_final), h_final


# Train and incremental save embeddings
def train_and_save_embeddings(use_pos=False):
    ds = MatrixDataset(CSV_PATH)
    in_dim = FEATURE_COUNT + (FEATURE_COUNT * 2 if use_pos else 0)
    logger.info(f"Initializing model with in_dim={in_dim} (use_pos={use_pos})")
    model = StackedBiLSTMClassifier(in_dim, ds.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    emb_file = os.path.join(EMBED_DIR, f"embeddings_{use_pos}.npy")
    lab_file = os.path.join(EMBED_DIR, f"labels_{use_pos}.npy")
    name_file = os.path.join(EMBED_DIR, f"matrix_names_{use_pos}.npy")
    dim_file = os.path.join(EMBED_DIR, f"matrix_dims_{use_pos}.npy")
    npz_file = os.path.join(EMBED_DIR, f"data_{use_pos}.npz")

    # remove old files
    for f in (emb_file, lab_file, name_file, dim_file, npz_file):
        if os.path.exists(f):
            os.remove(f)

        # training loop
    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{EPOCHS} start (use_pos={use_pos})")
        epoch_loss = 0.0
        epoch_correct = 0
        total_samples = len(ds)

        # Clear lists to accumulate this epoch's data only
        all_embs = []
        all_labels = []
        all_names = []
        all_dims = []

        for idx, (name, label) in enumerate(ds):
            feats, pos, tile, shape = get_tile_features(name, use_pos)
            mem = psutil.Process().memory_info().rss // (1024**2)
            logger.info(
                f"Matrix={name}, tile={tile}, tiles={feats.shape[0]}, RSS={mem}MB"
            )

            data = np.hstack([feats, pos]) if use_pos else feats
            X = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
            pred, embs = model(X)
            print(pred.shape)
            loss = criterion(pred, label.to(device).unsqueeze(0))
            # loss = criterion(pred, label.unsqueeze(0).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            acc = (pred.detach().cpu().numpy() > 0.5).astype(int)
            sample_acc = np.mean(acc == label.numpy())
            epoch_correct += sample_acc
            logger.info(f"{name}: loss={loss.item():.4f}, acc={sample_acc:.2f}")

            emb_row = embs.detach().cpu().numpy()
            lab_row = label.numpy().reshape(1, -1)
            name_row = np.array([name], dtype=object)
            dim_row = np.array([shape])

            # Accumulate this epoch's samples
            all_embs.append(emb_row)
            all_labels.append(lab_row)
            all_names.append(name_row)
            all_dims.append(dim_row)

        avg_loss = epoch_loss / total_samples
        avg_acc = epoch_correct / total_samples
        logger.info(
            f"Epoch {epoch + 1} completed: avg_loss={avg_loss:.4f}, avg_acc={avg_acc:.4f}"
        )

        # Save the current epoch's accumulated data to disk
        all_embs_np = np.vstack(all_embs)
        all_labels_np = np.vstack(all_labels)
        all_names_np = np.concatenate(all_names)
        all_dims_np = np.vstack(all_dims)

        np.savez_compressed(
            npz_file,
            embeddings=all_embs_np,
            labels=all_labels_np,
            matrix_names=all_names_np,
            matrix_dims=all_dims_np,
        )
        logger.info(f"Saved epoch {epoch + 1} data to {npz_file}")


# execute
train_and_save_embeddings(use_pos=False)
train_and_save_embeddings(use_pos=True)
