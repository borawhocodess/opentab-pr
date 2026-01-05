# OpenTab

An open implementation of [TabPFN](https://github.com/PriorLabs/TabPFN) - Prior-Data Fitted Networks for tabular data. This implementation follows the approach described in the [TabPFN Nature paper](https://www.nature.com/articles/s41586-024-08328-6), using **Structural Causal Models (SCMs)** for synthetic data generation.

## Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Install TabArena for benchmarking (optional)
uv pip install "tabarena @ git+https://github.com/autogluon/tabarena.git#subdirectory=tabarena"

#Typical chain of executions to run a basic data generation + training + eval
python generate_data.py --n_datasets 1000000 --output data/synthetic.h5 --max_features 50
python train.py --data data/synthetic.h5 --steps 15500  --batch_size 4 --gradient_accumulation_steps 16 --n_layers 3 --log_interval 1 --eval_interval 20 --save_interval 500 --max_features 50 --embedding_size 96 --n_heads 4 --mlp_hidden 192
python evaluate.py --checkpoint checkpoints/final_model.pt --mode lite
python evaluate.py --mode leaderboard --results eval_results --method OpenTab


# Train classification model (online data generation)
python train.py --online --steps 100000 --output_dir checkpoints

# Train regression model
python train.py --online --regression --steps 100000 --output_dir checkpoints

# Evaluate on sklearn datasets (quick test)
python evaluate.py --checkpoint checkpoints/model_100000.pt --mode quick

# Evaluate on TabArena benchmark (all 51 datasets: 38 classification + 13 regression)
python evaluate.py --checkpoint checkpoints/model_100000.pt --mode lite

# Evaluate on TabArena benchmark (classification only: 38 datasets)
python evaluate.py --checkpoint checkpoints/model_100000.pt --mode lite --task-type classification

# Generate leaderboard comparing against SOTA methods
python evaluate.py --mode leaderboard --results eval_results --method OpenTab
```

## How It Works

```
Training:   Generate SCM-based data → Train Transformer → Save checkpoint
Inference:  clf.fit(X_train, y_train) → clf.predict(X_test)  # One forward pass!
```

## Use as a Classifier

```python
from model import OpenTabClassifier

clf = OpenTabClassifier("checkpoints/model.pt")
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
```

## Use as a Regressor

```python
from model import OpenTabRegressor

reg = OpenTabRegressor("checkpoints/model.pt")
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)  # Returns expected value over bins
```

## Repository Structure

```
model.py          # Transformer architecture (TabPFN encoder/decoder) + OpenTabClassifier/Regressor
generate_data.py  # SCM-based synthetic data generation
train.py          # Training loop with online data generation
evaluate.py       # TabArena benchmark + quick evaluation
```

## Data Generation

Synthetic data is generated using **Structural Causal Models (SCMs)**:

- **DAG Structure**: Growing network with redirection (preferential attachment for scale-free networks)
- **Node Representations**: Vector embeddings in ℝᵈ
- **Edge Functions**: Neural networks (with various activations), decision trees, or categorical mappings
- **Activations**: identity, log, sigmoid, abs, sin, tanh, rank, square, power (2-5), smooth ReLU, step, modulo
- **Post-processing**: Kumaraswamy warping (20% of datasets), quantization, missing values (MCAR with NaN)
- **Feature Sampling**: Beta(0.95, 8.0) distribution scaled to [1, 160] features
- **Cell Budget**: Tables capped at 75,000 cells

```bash
# Generate and save synthetic datasets to HDF5
python generate_data.py --n_datasets 100000 --output data/synthetic.h5

# Generate with visualization
python generate_data.py --n_datasets 10 --visualize

# Generate for regression
python generate_data.py --n_datasets 100000 --regression --output data/synthetic_reg.h5
```

### Data Generation Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_datasets` | 100000 | Number of datasets to generate |
| `--output` | data/synthetic.h5 | Output HDF5 file path |
| `--max_samples` | 512 | Maximum samples per dataset (paper uses up to 2048) |
| `--max_features` | 160 | Maximum features per dataset (Beta distribution, paper-aligned) |
| `--max_classes` | 10 | Maximum classes (classification) |
| `--regression` | False | Generate regression data |
| `--visualize` | False | Show visualization of generated data |
| `--seed` | 42 | Random seed |


## Training

Training can use offline data generation as described above in the quickstart, or **online data generation** - each batch is a freshly generated synthetic dataset from an SCM prior.

```bash
# Basic training (classification)
python train.py --online --steps 100000

# Training with custom architecture
python train.py --online \
    --embedding_size 128 \
    --n_heads 4 \
    --n_layers 6 \
    --mlp_hidden 256 \
    --batch_size 64 \
    --lr 3e-4

# Training for regression
python train.py --online --regression --n_bins 64

# Resume from checkpoint
python train.py --online --resume checkpoints/model_50000.pt
```

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--online` | - | Enable online data generation (recommended) |
| `--data` | None | HDF5 file with pre-generated data (alternative to --online) |
| `--steps` | 100000 | Number of training steps |
| `--batch_size` | 64 | Batch size |
| `--lr` | 3e-4 | Learning rate |
| `--warmup_steps` | 1000 | Linear warmup steps |
| `--embedding_size` | 128 | Transformer embedding dimension |
| `--n_heads` | 4 | Number of attention heads |
| `--n_layers` | 6 | Number of transformer layers |
| `--mlp_hidden` | 256 | MLP hidden layer size |
| `--regression` | False | Train for regression task |
| `--max_classes` | 10 | Maximum number of classes (classification) |
| `--n_bins` | 64 | Number of bins (regression) |
| `--output_dir` | checkpoints | Output directory for checkpoints |
| `--compile` | False | Use torch.compile() for ~10-20% faster training |


## Evaluation

### Quick Evaluation (sklearn datasets)

```bash
# Classification: Iris, Wine, Breast Cancer
python evaluate.py --checkpoint checkpoints/model.pt --mode quick

# Regression: Diabetes, California Housing
python evaluate.py --checkpoint checkpoints/model.pt --mode quick-regression
```

### TabArena Benchmark

[TabArena](https://github.com/autogluon/tabarena) provides standardized benchmarking against state-of-the-art tabular ML methods.

```bash
# TabArena-Lite (all 51 datasets, 1 fold each)
python evaluate.py --checkpoint checkpoints/model.pt --mode lite

# TabArena-Lite (classification only: 38 datasets)
python evaluate.py --checkpoint checkpoints/model.pt --mode lite --task-type classification

# TabArena-Lite (regression only: 13 datasets)
python evaluate.py --checkpoint checkpoints/model.pt --mode lite --task-type regression

# Full TabArena (all datasets, all folds)
python evaluate.py --checkpoint checkpoints/model.pt --mode full

# Full TabArena (classification only, all folds)
python evaluate.py --checkpoint checkpoints/model.pt --mode full --task-type classification

# Generate leaderboard with ELO ratings
python evaluate.py --mode leaderboard --results eval_results --method OpenTab

# Load cached leaderboard
python evaluate.py --mode leaderboard-cache --method OpenTab
```

### Evaluation Modes

| Mode | Description |
|------|-------------|
| `quick` | Test on sklearn classification datasets |
| `quick-regression` | Test on sklearn regression datasets |
| `lite` | TabArena-Lite (51 datasets, 1 fold). Use `--task-type` to filter |
| `full` | Full TabArena (all datasets, all folds). Use `--task-type` to filter |
| `leaderboard` | Generate leaderboard with ELO ratings |
| `leaderboard-cache` | Load leaderboard from cache |

### Task Type Filtering

Use `--task-type` (or `-t`) to filter tasks in `lite` and `full` modes:
- `all` (default): Evaluate on all 51 tasks (38 classification + 13 regression)
- `classification`: Only 38 classification tasks
- `regression`: Only 13 regression tasks

## Model Architecture

The model follows the TabPFN architecture with two-way attention:

1. **Input Processing**:
   - Z-normalization per feature using training statistics
   - Random feature embeddings for disambiguation
   - Missing value indicators (zero-fill + indicator embedding)

2. **TabPFN Encoder** (N layers):
   - **Inter-feature attention**: Fully connected across features
   - **Inter-sample attention**: Test samples attend to train samples only

3. **Decoder**:
   - MLP decoder for classification (softmax over classes)
   - Piecewise constant output for regression (binning approach)

## Current TabArena Leaderboard - Classification Only (05.01.2026)

Note: OpenTab is still in early development. Performance is expected to improve significantly. The current results demonstrate a working baseline trained with minimal compute resources.

**Training configuration**: 1M synthetic datasets, 512 max samples, 50 max features, 10 classes, ~15,500 steps (1 epoch) on a modest NVIDIA GeForce RTX 3050 4GB laptop GPU.

**Evaluation**: 38 classification tasks from TabArena-v0.1. The model supports both classification and regression, but these results show classification-only performance (use `--task-type classification` to reproduce).

Results on [TabArena](https://github.com/autogluon/tabarena) benchmark comparing OpenTab against state-of-the-art tabular ML methods:

| # | Model | Elo ⬆️ | Elo 95% CI | Score ⬆️ | Rank ⬇️ | Harmonic Rank ⬇️ | Improvability (%) ⬇️ |
|---:|:---|---:|:---|---:|---:|---:|---:|
| 0 | RealTabPFN-v2.5 (tuned + ensembled) | 1588 | +99/-86 | 0.646 | 9.26 | 2.8 | 4.624 |
| 1 | AutoGluon 1.4 (extreme, 4h) | 1581 | +93/-79 | 0.658 | 9.54 | 3.89 | 6.975 |
| 2 | RealTabPFN-v2.5 (tuned) | 1555 | +93/-79 | 0.592 | 10.55 | 4.91 | 7.143 |
| 3 | RealTabPFN-v2.5 (default) | 1533 | +84/-72 | 0.568 | 11.51 | 5.92 | 7.507 |
| 4 | AutoGluon 1.4 (best, 4h) | 1526 | +92/-61 | 0.565 | 11.82 | 4.64 | 8.629 |
| 5 | RealMLP_GPU (tuned + ensembled) | 1482 | +60/-50 | 0.481 | 13.89 | 7.46 | 10.302 |
| 6 | RealMLP_GPU (tuned) | 1415 | +54/-52 | 0.403 | 17.49 | 7.85 | 11.952 |
| 7 | TabM_GPU (tuned + ensembled) | 1403 | +76/-59 | 0.382 | 18.18 | 8.02 | 12.434 |
| 8 | LightGBM (tuned + ensembled) | 1392 | +55/-41 | 0.306 | 18.86 | 13.37 | 13.178 |
| 9 | TabDPT_GPU (tuned + ensembled) | 1372 | +96/-61 | 0.353 | 20.05 | 8.17 | 11.441 |
| 10 | CatBoost (tuned + ensembled) | 1371 | +73/-56 | 0.33 | 20.09 | 12.2 | 12.7 |
| 11 | TabICL_GPU (default) [5.26% IMPUTED] | 1365 | +80/-73 | 0.342 | 20.45 | 7.16 | 12.606 |
| 12 | XGBoost (tuned + ensembled) | 1362 | +72/-74 | 0.296 | 20.67 | 12.1 | 13.768 |
| 13 | ModernNCA_GPU (tuned) | 1348 | +67/-71 | 0.306 | 21.5 | 10.95 | 13.208 |
| 14 | CatBoost (default) | 1348 | +50/-53 | 0.288 | 21.5 | 10.8 | 13.343 |
| 15 | CatBoost (tuned) | 1348 | +65/-58 | 0.3 | 21.51 | 14.53 | 13.231 |
| 16 | TabM_GPU (tuned) | 1342 | +82/-63 | 0.319 | 21.88 | 11.89 | 13.403 |
| 17 | ModernNCA_GPU (tuned + ensembled) | 1341 | +95/-87 | 0.333 | 21.92 | 9.35 | 13.248 |
| 18 | TabPFNv2_GPU (tuned + ensembled) [31.58% IMPUTED] | 1333 | +95/-81 | 0.348 | 22.43 | 8.43 | 14.799 |
| 19 | LightGBM (tuned) | 1332 | +67/-48 | 0.255 | 22.47 | 16.55 | 14.12 |
| 20 | XGBoost (tuned) | 1331 | +66/-67 | 0.262 | 22.55 | 11.46 | 14.021 |
| 21 | TabDPT_GPU (tuned) | 1323 | +74/-72 | 0.293 | 23.04 | 8.26 | 13.473 |
| 22 | Mitra_GPU (default) [31.58% IMPUTED] | 1315 | +86/-83 | 0.303 | 23.53 | 9.68 | 15.138 |
| 23 | xRFM_GPU (tuned + ensembled) | 1309 | +72/-57 | 0.25 | 23.92 | 13.63 | 14.514 |
| 24 | EBM (tuned + ensembled) | 1303 | +63/-51 | 0.22 | 24.29 | 15 | 15.58 |
| 25 | TabM_GPU (default) | 1280 | +76/-64 | 0.246 | 25.76 | 16.92 | 16.062 |
| 26 | TabPFNv2_GPU (tuned) [31.58% IMPUTED] | 1272 | +76/-79 | 0.242 | 26.3 | 13.03 | 16.524 |
| 27 | TorchMLP (tuned + ensembled) | 1262 | +50/-48 | 0.172 | 26.95 | 20.99 | 15.317 |
| 28 | RealMLP_GPU (default) | 1259 | +64/-51 | 0.166 | 27.11 | 17.8 | 15.931 |
| 29 | xRFM_GPU (tuned) | 1258 | +70/-55 | 0.176 | 27.2 | 13.05 | 16.168 |
| 30 | EBM (tuned) | 1249 | +67/-54 | 0.154 | 27.74 | 14.63 | 16.507 |
| 31 | EBM (default) | 1241 | +60/-55 | 0.163 | 28.24 | 14.3 | 17.402 |
| 32 | TabPFNv2_GPU (default) [31.58% IMPUTED] | 1238 | +81/-80 | 0.22 | 28.49 | 10.51 | 17.278 |
| 33 | TabDPT_GPU (default) | 1211 | +82/-69 | 0.189 | 30.14 | 11.05 | 16.684 |
| 34 | FastaiMLP (tuned + ensembled) | 1207 | +68/-91 | 0.149 | 30.41 | 20.45 | 18.549 |
| 35 | XGBoost (default) | 1192 | +68/-70 | 0.131 | 31.34 | 15.78 | 17.116 |
| 36 | ExtraTrees (tuned + ensembled) | 1192 | +63/-59 | 0.116 | 31.37 | 22.21 | 18.759 |
| 37 | TorchMLP (tuned) | 1191 | +69/-62 | 0.13 | 31.42 | 23.76 | 17.609 |
| 38 | ModernNCA_GPU (default) | 1167 | +65/-61 | 0.095 | 32.91 | 15.43 | 19.205 |
| 39 | RandomForest (tuned + ensembled) | 1166 | +66/-75 | 0.107 | 32.95 | 22.93 | 19.675 |
| 40 | ExtraTrees (tuned) | 1151 | +66/-76 | 0.114 | 33.86 | 18.84 | 20.059 |
| 41 | LightGBM (default) | 1141 | +58/-63 | 0.096 | 34.49 | 29.15 | 18.433 |
| 42 | FastaiMLP (tuned) | 1124 | +68/-80 | 0.097 | 35.47 | 22.82 | 20.568 |
| 43 | RandomForest (tuned) | 1122 | +57/-59 | 0.058 | 35.61 | 28.7 | 20.512 |
| 44 | TorchMLP (default) | 1029 | +60/-74 | 0.029 | 40.68 | 35.98 | 23.249 |
| 45 | RandomForest (default) | 1000 | +60/-80 | 0.019 | 42.09 | 33 | 26.601 |
| 46 | KNN (tuned + ensembled) | 996 | +72/-98 | 0.032 | 42.29 | 36.61 | 27.368 |
| 47 | FastaiMLP (default) | 986 | +72/-67 | 0.029 | 42.76 | 38.51 | 24.469 |
| 48 | xRFM_GPU (default) | 966 | +89/-88 | 0.043 | 43.63 | 37.51 | 27.398 |
| 49 | Linear (tuned + ensembled) | 962 | +81/-96 | 0.032 | 43.83 | 20.39 | 30.833 |
| 50 | Linear (tuned) | 932 | +88/-104 | 0.023 | 45.08 | 28.59 | 31.557 |
| 51 | ExtraTrees (default) | 920 | +73/-100 | 0.014 | 45.58 | 41.5 | 29.133 |
| 52 | Linear (default) | 874 | +86/-106 | 0.01 | 47.26 | 44.03 | 34.148 |
| 53 | KNN (tuned) | 825 | +91/-116 | 0.016 | 48.84 | 46.25 | 34.867 |
| 54 | **OpenTab** | **818** | **+136/-191** | **0.038** | **49.05** | **41.48** | **42.268** |
| 55 | KNN (default) | 553 | +104/-150 | 0 | 54.24 | 53.97 | 49.494 |


## References

```bibtex
@article{hollmann2025tabpfn,
  title={Accurate predictions on small data with a tabular foundation model},
  author={Hollmann, Noah and Müller, Samuel and Purucker, Lennart and others},
  journal={Nature},
  year={2025},
  publisher={Nature Publishing Group}
}

@article{hollmann2023tabpfn,
  title={TabPFN: A transformer that solves small tabular classification problems in a second},
  author={Hollmann, Noah and Müller, Samuel and Eggensperger, Katharina and Hutter, Frank},
  booktitle={ICLR 2023},
  year={2023}
}

@article{pfefferle2025nanotabpfn,
  title={nanoTabPFN: A Lightweight and Educational Reimplementation of TabPFN},
  author={Pfefferle, Alexander and Hog, Johannes and Purucker, Lennart and Hutter, Frank},
  journal={arXiv preprint arXiv:2511.03634},
  year={2025}
}

@article{erickson2025tabarena,
  title={TabArena: A Living Benchmark for Machine Learning on Tabular Data}, 
  author={Nick Erickson and Lennart Purucker and others},
  year={2025},
  journal={arXiv preprint arXiv:2506.16791},
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
