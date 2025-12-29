# OpenTab

An open implementation of TabPFN-like models (Prior-Data Fitted Network for Tabular Data) with **full synthetic data generation** from scratch.

## Quick Start

```bash
# Install
pip install -e .

# Generate synthetic data
python generate_data.py --n_datasets 100000 --output data/synthetic_100k.h5

# Train
python train.py --data data/synthetic_100k.h5 --epochs 100 --output checkpoints/opentab.pt

# Evaluate (quick test on sklearn datasets)
python evaluate.py --checkpoint checkpoints/opentab.pt --mode quick

# Evaluate on TabArena benchmark (requires: pip install tabarena autogluon openml)
source .venv/bin/activate && python evaluate.py --checkpoint checkpoints/opentab.pt --mode lite
```

## Use as a Classifier

```python
from model import OpenTabClassifier

clf = OpenTabClassifier("checkpoints/opentab.pt")
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

## What is TabPFN?

TabPFN learns to **approximate Bayesian inference** on tabular data. It's trained on millions of synthetic datasets, then at inference time performs **in-context learning** - a single forward pass, no gradient updates.

```
Training:   Generate synthetic data → Train Transformer → Save checkpoint
Inference:  clf.fit(X_train, y_train) → clf.predict(X_test)  # One forward pass!
```

## Repository Structure

```
model.py          # Transformer architecture (~450 LOC)
generate_data.py  # Synthetic data generation (~770 LOC)
train.py          # Training loop (~770 LOC)
evaluate.py       # TabArena benchmark (~620 LOC)
```

## Key Commands

| Task | Command |
|------|---------|
| Generate data | `python generate_data.py --n_datasets 100000 --output data/synthetic.h5` |
| Train | `python train.py --data data/synthetic.h5 --epochs 100` |
| Quick eval | `python evaluate.py --checkpoint checkpoints/model.pt --mode quick` |
| TabArena eval | `python evaluate.py --checkpoint checkpoints/model.pt --mode lite` |

## Configuration

**Data Generation:**
- `--n_datasets`: Number of synthetic datasets (default: 100000)
- `--max_samples`: Max samples per dataset (default: 50)
- `--max_features`: Max features (default: 10)
- `--prior`: Prior type - `mlp`, `gp`, `scm`, `tree` (default: mlp)

**Training:**
- `--epochs`: Training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--embedding_size`: Embedding dimension (default: 96)
- `--n_layers`: Transformer layers (default: 3)

**Evaluation:**
- `--mode quick`: Test on sklearn datasets (Iris, Wine, Breast Cancer)
- `--mode lite`: TabArena-Lite (38 datasets, 1 fold)
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
