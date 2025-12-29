# OpenTab

An open implementation of TabPFN-like models (Prior-Data Fitted Network for Tabular Data) with **full synthetic data generation** from scratch.

## Quick Start

```bash
# Install
pip install -e .

# Generate synthetic data for classification
python generate_data.py --n_datasets 100000 --output data/synthetic_clf.h5

# Train classification model
python train.py --data data/synthetic_clf.h5 --task classification --epochs 100 --output checkpoints/classifier.pt

# Train regression model (separate model, following TabPFN approach)
python train.py --online --task regression --epochs 100 --output checkpoints/regressor.pt

# Evaluate classification (quick test on sklearn datasets)
python evaluate.py --checkpoint checkpoints/classifier.pt --mode quick

# Evaluate regression
python evaluate.py --checkpoint checkpoints/regressor.pt --mode quick-regression

# Evaluate on TabArena benchmark (classification, requires: pip install tabarena autogluon openml)
source .venv/bin/activate && python evaluate.py --checkpoint checkpoints/classifier.pt --mode lite
```

## Two-Model Approach (Like TabPFN)

Following [TabPFN](https://github.com/PriorLabs/TabPFN), OpenTab trains **separate models** for classification and regression:

| Task | Prior Type | Model Class | Checkpoint |
|------|------------|-------------|------------|
| Classification | `mixed` (MLP, GP, Tree, SCM) | `OpenTabClassifier` | `classifier.pt` |
| Regression | `mixed_regression` (MLP, GP, Linear) | `OpenTabRegressor` | `regressor.pt` |

This matches how TabPFN provides separate pre-trained weights:
- `tabpfn-v2.5-classifier-*.ckpt` for classification
- `tabpfn-v2.5-regressor-*.ckpt` for regression

## Use as a Classifier

```python
from model import OpenTabClassifier

clf = OpenTabClassifier("checkpoints/classifier.pt")
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
```

## Use as a Regressor

```python
from model import OpenTabRegressor

reg = OpenTabRegressor("checkpoints/regressor.pt")
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)  # Returns mean predictions
predictions = reg.predict(X_test, output_type='median')  # Or median
quantiles = reg.predict_quantiles(X_test, quantiles=[0.1, 0.5, 0.9])
```

## What is TabPFN?

TabPFN learns to **approximate Bayesian inference** on tabular data. It's trained on millions of synthetic datasets, then at inference time performs **in-context learning** - a single forward pass, no gradient updates.

```
Training:   Generate synthetic data → Train Transformer → Save checkpoint
Inference:  clf.fit(X_train, y_train) → clf.predict(X_test)  # One forward pass!
```

## Repository Structure

```
model.py          # Transformer architecture + OpenTabClassifier/Regressor
generate_data.py  # Synthetic data generation (classification & regression priors)
train.py          # Training loop with task-specific evaluation
evaluate.py       # TabArena benchmark + quick evaluation
```

## Key Commands

| Task | Command |
|------|---------|
| Generate classification data | `python generate_data.py --n_datasets 100000 --output data/synthetic.h5` |
| Train classification | `python train.py --online --task classification --epochs 100` |
| Train regression | `python train.py --online --task regression --epochs 100` |
| Quick eval (classification) | `python evaluate.py --checkpoint model.pt --mode quick` |
| Quick eval (regression) | `python evaluate.py --checkpoint model.pt --mode quick-regression` |
| TabArena eval | `python evaluate.py --checkpoint model.pt --mode lite` |

## Configuration

**Data Generation:**
- `--n_datasets`: Number of synthetic datasets (default: 100000)
- `--max_samples`: Max samples per dataset (default: 100)
- `--max_features`: Max features (default: 20)
- `--prior`: Prior type - `mlp`, `gp`, `scm`, `tree`, `mixed` for classification; `mlp_regression`, `gp_regression`, `linear_regression`, `mixed_regression` for regression

**Training:**
- `--task`: Task type - `classification` or `regression`
- `--epochs`: Training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--embedding_size`: Embedding dimension (default: 96)
- `--n_layers`: Transformer layers (default: 3)

**Evaluation:**
- `--mode quick`: Test classification on sklearn datasets (Iris, Wine, Breast Cancer)
- `--mode quick-regression`: Test regression on sklearn datasets (Diabetes, California Housing)
- `--mode lite`: TabArena-Lite (51 classification datasets, 1 fold)
- `--mode full`: Full TabArena (all folds)

## References

```bibtex
@article{hollmann2023tabpfn,
  title={TabPFN: A transformer that solves small tabular classification problems in a second},
  author={Hollmann, Noah and Müller, Samuel and Eggensperger, Katharina and Hutter, Frank},
  booktitle={ICLR 2023},
  year={2023}
}
```

## License

Apache 2.0
