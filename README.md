# HyPerTune: Hypergraph Partitioning Performance Tuning with ML/DL

HyPerTune is a machine learning and deep learning-based framework for high-performance hypergraph partitioning parameter tuning.

Instead of performing expensive partitioning operations at runtime, HyPerTune **predicts the optimal preset configuration** (e.g., preset type in KaHyPar or PaToH) for a given sparse matrix or hypergraph. It uses supervised learning models trained on historical performance data.

---

## Project Goals

- Automate dataset construction by downloading, converting, and synthesizing matrices
- Benchmark hypergraph partitioners with preset variations, collecting runtime and quality metrics
- Dataset construction integrates with external tools like:
  - [KaHyPar](https://github.com/kahypar/kahypar)
  - [PaToH](https://faculty.cc.gatech.edu/~umit/software.html)
  - [MatGen](https://github.com/He-Is-HaZaRdOuS/MatGen) for matrix synthesis
- Extract deep embedded features from matrices via deep learning models
- Train ML/DL models to predict the best preset configuration for a given matrix
---

## Usage

This repository focuses on **tuning** workflows — training, predicting, and analyzing preset choices. For matrix generation and dataset extension, refer to [MatGen](https://github.com/He-Is-HaZaRdOuS/MatGen).

---


## Folder Structure

- `data/` — Holds static artifacts like partitioning logs, CSV datasets and DL-learned features.
- `scripts/` — Utility scripts for downloading and pruning matrices.
- `model/` — Contains model definitions, training, and evaluation code.
- `results/` — Stores output files, logs, and metrics from experiments.
- `generation/` — Contains scripts or placeholders for synthetic data generation.
- `matrices/` — Input data matrices.
- `partitioning/` — Contains automation scripts for running partitioners (e.g., KaHyPar or PaToH) and managing their outputs.
- `preprocessing/` — Various scripts for preparing and processing raw data.
- `visuals/` — Scripts and generated figures for result visualization.

---

## Overall Workflow & Data Flow

```text
        ┌─────────────┐
        │ Matrix Data │
        │ Acquisition │
        └──────┬──────┘
               │ .mtx files (Matrix Market format)
               │
       ┌───────▼──────────┐
       │ Conversion to    │
       │ Hypergraph Files │
       │ (.hgr, .patoh)   │
       └───────▼──────────┘
               │
       ┌───────▼────────────────┐
       │ Benchmarking with      │
       │ KaHyPar & PaToH using  │
       │ many preset configs    │
       └───────▼────────────────┘
               │ Runtime, quality, memory
               │ metrics + logs
       ┌───────▼─────────────┐
       │ Data Processing &   │
       │ Label Generation    │
       └───────▼─────────────┘
               │
       ┌───────▼────────────┐
       │ Feature Extraction │
       └───────▼────────────┘
               │
       ┌───────▼──────────────┐
       │ ML/DL Model Training │
       └───────▼──────────────┘
               │
       ┌───────▼──────────┐
       │ Preset Prediction│
       └──────────────────┘
