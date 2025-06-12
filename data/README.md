# data/

This folder holds all generated and preprocessed data required for training and evaluation of models in the HyPerTune project.

## Folder Layout

data/
├── logs/ # Logs from partitioning tools like KaHyPar and PaToH, including timeout info.
├── raw/ # Original raw CSVs containing partitioning results
├── processed/ # Feature-processed but not yet pruned or labelled
├── ml_ready/ # Final cleaned CSVs for ML training
└── dl_features/ # Deep learning feature arrays

### DL Feature Embeddings

`dl_features/` folder contains deep feature embeddings derived from a stacked bidirectional LSTM trained on matrix tiles.

Each matrix is tiled into 256 block-wise fragments to expose local patterns. These are passed through the LSTM to extract learned embeddings.

- **`True`**: Feature vectors include sinusoidal positional encoding
- **`False`**: No positional encoding used

These DL embeddings aim to encode deeper structural properties for supervised preset prediction.
