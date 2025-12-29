"""
generate_data.py - Synthetic Data Generation for OpenTab Training

This module implements various "priors" - ways to generate synthetic tabular
datasets that cover the space of real-world data-generating processes.

Key Insight: OpenTab is trained on synthetic data, not real datasets, like it's inspiration TabPFN. The 
quality and diversity of synthetic data determines what the model can learn.

Available Priors:
1. MLPPrior: Random neural networks as data-generating functions
2. GPPrior: Gaussian Processes with random kernels
3. TreePrior: Random decision trees / boolean functions
4. SCMPrior: Structural Causal Models (DAG-based)

Each prior generates:
- X: Features (random input points)
- y: Targets (outputs of the random function)
- Train/test split position

References:
- TabPFN paper: "TabPFN: A Transformer That Solves Small Tabular Classification Problems"
- TICL: microsoft/ticl - Prior implementations
- TabICL: soda-inria/tabicl - Alternative prior
"""

import argparse
import math
import random
from typing import Tuple, Optional, Dict, Any, Callable, List
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import h5py
from tqdm import tqdm


@dataclass
class SyntheticDataset:
    """A single synthetic dataset."""
    X: np.ndarray  # (n_samples, n_features)
    y: np.ndarray  # (n_samples,)
    train_size: int  # Number of training samples
    n_classes: int  # Number of classes (for classification), 0 for regression
    is_regression: bool = False  # Whether this is a regression task
    categorical_mask: Optional[np.ndarray] = None  # (n_features,) bool - True if feature is categorical
    missing_mask: Optional[np.ndarray] = None  # (n_samples, n_features) bool - True if value is missing
    n_categories: Optional[np.ndarray] = None  # (n_features,) int - number of categories per feature (0 if continuous)


# ============================================================================
# Feature Augmentation - Following TabPFN Paper
# ============================================================================

class FeatureAugmenter:
    """
    Augments synthetic data with realistic real-world characteristics.
    
    Following TabPFN, this class adds:
    1. Categorical features (ordinal encoding)
    2. Missing values (random masking)
    3. Feature noise and transformations
    
    This is applied AFTER generating base synthetic data from a prior.
    """
    
    def __init__(
        self,
        categorical_prob: float = 0.3,  # Probability of a feature being categorical
        missing_prob: float = 0.1,  # Probability of a value being missing
        n_categories_range: Tuple[int, int] = (2, 10),  # Range for number of categories
        missing_indicator_value: float = -999.0,  # Value to use for missing
        add_noise: bool = True,
        noise_std: float = 0.01,
    ):
        self.categorical_prob = categorical_prob
        self.missing_prob = missing_prob
        self.n_categories_range = n_categories_range
        self.missing_indicator_value = missing_indicator_value
        self.add_noise = add_noise
        self.noise_std = noise_std
    
    def augment(self, dataset: 'SyntheticDataset') -> 'SyntheticDataset':
        """
        Apply augmentations to a synthetic dataset.
        
        Args:
            dataset: A SyntheticDataset with continuous features
            
        Returns:
            Augmented SyntheticDataset with categorical/missing metadata
        """
        X = dataset.X.copy()
        n_samples, n_features = X.shape
        
        # Initialize masks
        categorical_mask = np.zeros(n_features, dtype=bool)
        n_categories = np.zeros(n_features, dtype=np.int32)
        missing_mask = np.zeros((n_samples, n_features), dtype=bool)
        
        # Apply categorical transformation to random features
        for i in range(n_features):
            if random.random() < self.categorical_prob:
                X, categorical_mask, n_categories = self._make_categorical(
                    X, i, categorical_mask, n_categories
                )
        
        # Apply missing values
        if self.missing_prob > 0:
            X, missing_mask = self._add_missing_values(X, n_samples, n_features)
        
        # Add noise to continuous features
        if self.add_noise:
            X = self._add_feature_noise(X, categorical_mask, missing_mask)
        
        return SyntheticDataset(
            X=X.astype(np.float32),
            y=dataset.y,
            train_size=dataset.train_size,
            n_classes=dataset.n_classes,
            is_regression=dataset.is_regression,
            categorical_mask=categorical_mask,
            missing_mask=missing_mask,
            n_categories=n_categories,
        )
    
    def _make_categorical(
        self,
        X: np.ndarray,
        feature_idx: int,
        categorical_mask: np.ndarray,
        n_categories: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert a continuous feature to categorical via binning."""
        n_cats = random.randint(*self.n_categories_range)
        
        # Use percentile-based binning for robustness
        feature_values = X[:, feature_idx]
        percentiles = np.linspace(0, 100, n_cats + 1)[1:-1]
        thresholds = np.percentile(feature_values, percentiles)
        
        # Ordinal encode: value becomes integer category
        X[:, feature_idx] = np.digitize(feature_values, thresholds).astype(np.float32)
        
        categorical_mask[feature_idx] = True
        n_categories[feature_idx] = n_cats
        
        return X, categorical_mask, n_categories
    
    def _add_missing_values(
        self,
        X: np.ndarray,
        n_samples: int,
        n_features: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add missing values following TabPFN patterns."""
        missing_mask = np.zeros((n_samples, n_features), dtype=bool)
        
        # Different missing patterns (TabPFN uses multiple)
        pattern = random.choice(['random', 'column', 'row', 'block'])
        
        if pattern == 'random':
            # Completely random missingness
            missing_mask = np.random.rand(n_samples, n_features) < self.missing_prob
            
        elif pattern == 'column':
            # Some columns have high missingness
            n_missing_cols = random.randint(1, max(1, n_features // 3))
            missing_cols = np.random.choice(n_features, n_missing_cols, replace=False)
            for col in missing_cols:
                col_missing_prob = random.uniform(0.1, 0.5)
                missing_mask[:, col] = np.random.rand(n_samples) < col_missing_prob
                
        elif pattern == 'row':
            # Some rows have high missingness
            n_missing_rows = random.randint(1, max(1, n_samples // 5))
            missing_rows = np.random.choice(n_samples, n_missing_rows, replace=False)
            for row in missing_rows:
                n_missing_features = random.randint(1, max(1, n_features // 2))
                missing_features = np.random.choice(n_features, n_missing_features, replace=False)
                missing_mask[row, missing_features] = True
                
        elif pattern == 'block':
            # Block missingness (correlated)
            block_size = random.randint(2, max(2, min(n_samples // 4, n_features // 2)))
            start_row = random.randint(0, max(0, n_samples - block_size))
            start_col = random.randint(0, max(0, n_features - block_size))
            n_block_rows = min(block_size, n_samples - start_row)
            n_block_cols = min(block_size, n_features - start_col)
            missing_mask[start_row:start_row+n_block_rows, start_col:start_col+n_block_cols] = True
        
        # Ensure at least some non-missing values per feature
        for col in range(n_features):
            if missing_mask[:, col].sum() > n_samples * 0.8:
                # Keep at least 20% non-missing
                missing_idx = np.where(missing_mask[:, col])[0]
                n_to_keep = int(n_samples * 0.2)
                keep_idx = np.random.choice(missing_idx, min(len(missing_idx), n_to_keep), replace=False)
                missing_mask[keep_idx, col] = False
        
        # Apply missing indicator value
        X[missing_mask] = self.missing_indicator_value
        
        return X, missing_mask
    
    def _add_feature_noise(
        self,
        X: np.ndarray,
        categorical_mask: np.ndarray,
        missing_mask: np.ndarray,
    ) -> np.ndarray:
        """Add small noise to continuous features (not categorical or missing)."""
        continuous_mask = ~categorical_mask
        for i in range(X.shape[1]):
            if continuous_mask[i]:
                # Add noise only to non-missing values
                non_missing = ~missing_mask[:, i]
                if non_missing.any():
                    feature_std = X[non_missing, i].std() + 1e-8
                    noise = np.random.randn(non_missing.sum()) * self.noise_std * feature_std
                    X[non_missing, i] += noise
        return X


class AugmentedPrior:
    """
    Wrapper that applies FeatureAugmenter to any base prior.
    
    This combines synthetic data generation with realistic augmentation,
    following the TabPFN approach.
    
    Usage:
        base_prior = MixedPrior()
        augmented = AugmentedPrior(base_prior, categorical_prob=0.3, missing_prob=0.1)
        dataset = augmented.generate(n_samples=100, n_features=10, n_classes=2)
    """
    
    def __init__(
        self,
        base_prior: Any,
        categorical_prob: float = 0.3,
        missing_prob: float = 0.1,
        n_categories_range: Tuple[int, int] = (2, 10),
        augment_prob: float = 0.7,  # Probability of applying augmentation
    ):
        self.base_prior = base_prior
        self.augmenter = FeatureAugmenter(
            categorical_prob=categorical_prob,
            missing_prob=missing_prob,
            n_categories_range=n_categories_range,
        )
        self.augment_prob = augment_prob
    
    def generate(
        self,
        n_samples: int,
        n_features: int,
        n_classes: int = 2,
        train_ratio: float = 0.7,
    ) -> SyntheticDataset:
        """Generate an augmented synthetic dataset."""
        # Generate base dataset
        dataset = self.base_prior.generate(n_samples, n_features, n_classes, train_ratio)
        
        # Apply augmentation with some probability
        if random.random() < self.augment_prob:
            dataset = self.augmenter.augment(dataset)
        else:
            # No augmentation - still create empty masks
            dataset = SyntheticDataset(
                X=dataset.X,
                y=dataset.y,
                train_size=dataset.train_size,
                n_classes=dataset.n_classes,
                is_regression=dataset.is_regression,
                categorical_mask=np.zeros(n_features, dtype=bool),
                missing_mask=np.zeros((n_samples, n_features), dtype=bool),
                n_categories=np.zeros(n_features, dtype=np.int32),
            )
        
        return dataset


class AugmentedRegressionPrior:
    """
    Wrapper that applies FeatureAugmenter to regression priors.
    """
    
    def __init__(
        self,
        base_prior: Any,
        categorical_prob: float = 0.3,
        missing_prob: float = 0.1,
        n_categories_range: Tuple[int, int] = (2, 10),
        augment_prob: float = 0.7,
    ):
        self.base_prior = base_prior
        self.augmenter = FeatureAugmenter(
            categorical_prob=categorical_prob,
            missing_prob=missing_prob,
            n_categories_range=n_categories_range,
        )
        self.augment_prob = augment_prob
    
    def generate(
        self,
        n_samples: int,
        n_features: int,
        train_ratio: float = 0.7,
    ) -> SyntheticDataset:
        """Generate an augmented synthetic regression dataset."""
        # Generate base dataset
        dataset = self.base_prior.generate(n_samples, n_features, train_ratio)
        
        # Apply augmentation with some probability
        if random.random() < self.augment_prob:
            dataset = self.augmenter.augment(dataset)
        else:
            dataset = SyntheticDataset(
                X=dataset.X,
                y=dataset.y,
                train_size=dataset.train_size,
                n_classes=0,
                is_regression=True,
                categorical_mask=np.zeros(n_features, dtype=bool),
                missing_mask=np.zeros((n_samples, n_features), dtype=bool),
                n_categories=np.zeros(n_features, dtype=np.int32),
            )
        
        return dataset


class MLPPrior:
    """
    Generate data using random Multi-Layer Perceptrons.
    
    Each MLP represents a random function from features to targets.
    The architecture (layers, activation, weights) is randomly sampled.
    
    This prior tends to create smooth, complex decision boundaries.
    """
    
    def __init__(
        self,
        n_layers_range: Tuple[int, int] = (1, 4),
        hidden_size_range: Tuple[int, int] = (16, 128),
        activation_options: List[str] = ['relu', 'tanh', 'gelu', 'sigmoid'],
        weight_scale: float = 1.0,
        noise_std: float = 0.01,
    ):
        self.n_layers_range = n_layers_range
        self.hidden_size_range = hidden_size_range
        self.activation_options = activation_options
        self.weight_scale = weight_scale
        self.noise_std = noise_std
    
    def _get_activation(self, name: str) -> Callable:
        activations = {
            'relu': lambda x: np.maximum(0, x),
            'tanh': np.tanh,
            'gelu': lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))),
            'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
            'leaky_relu': lambda x: np.where(x > 0, x, 0.01 * x),
        }
        return activations.get(name, activations['relu'])
    
    def sample_function(
        self,
        n_features: int,
        n_outputs: int = 1,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Sample a random MLP function."""
        # Sample architecture
        n_layers = random.randint(*self.n_layers_range)
        hidden_sizes = [random.randint(*self.hidden_size_range) for _ in range(n_layers)]
        activation = self._get_activation(random.choice(self.activation_options))
        
        # Sample weights
        layer_sizes = [n_features] + hidden_sizes + [n_outputs]
        weights = []
        biases = []
        
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization with scaling
            scale = self.weight_scale * np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            b = np.random.randn(layer_sizes[i+1]) * scale * 0.1
            weights.append(W)
            biases.append(b)
        
        def forward(X: np.ndarray) -> np.ndarray:
            h = X
            for i, (W, b) in enumerate(zip(weights, biases)):
                h = h @ W + b
                if i < len(weights) - 1:  # Apply activation except last layer
                    h = activation(h)
            return h
        
        return forward
    
    def generate(
        self,
        n_samples: int,
        n_features: int,
        n_classes: int = 2,
        train_ratio: float = 0.7,
    ) -> SyntheticDataset:
        """Generate a synthetic classification dataset."""
        # Sample input features
        X = self._sample_features(n_samples, n_features)
        
        # Sample and apply random function
        func = self.sample_function(n_features, n_outputs=n_classes)
        logits = func(X)
        
        # Clip to prevent overflow and handle NaN/Inf
        logits = np.clip(logits, -1e6, 1e6)
        logits = np.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Add noise and convert to classes
        logits += np.random.randn(*logits.shape) * self.noise_std
        y = logits.argmax(axis=1)
        
        # Determine train/test split
        train_size = max(1, int(n_samples * train_ratio))
        train_size = min(train_size, n_samples - 1)  # At least 1 test sample
        
        return SyntheticDataset(
            X=X.astype(np.float32),
            y=y.astype(np.int64),
            train_size=train_size,
            n_classes=n_classes,
        )
    
    def _sample_features(self, n_samples: int, n_features: int) -> np.ndarray:
        """Sample feature values from various distributions."""
        X = np.zeros((n_samples, n_features))
        
        for i in range(n_features):
            dist_type = random.choice(['uniform', 'normal', 'mixture'])
            
            if dist_type == 'uniform':
                low = random.uniform(-10, 0)
                high = random.uniform(0, 10)
                X[:, i] = np.random.uniform(low, high, n_samples)
            elif dist_type == 'normal':
                mean = random.uniform(-5, 5)
                std = random.uniform(0.1, 3)
                X[:, i] = np.random.normal(mean, std, n_samples)
            else:  # mixture
                n_components = random.randint(2, 4)
                means = np.random.uniform(-5, 5, n_components)
                stds = np.random.uniform(0.1, 2, n_components)
                component_idx = np.random.choice(n_components, n_samples)
                X[:, i] = np.array([
                    np.random.normal(means[c], stds[c])
                    for c in component_idx
                ])
        
        return X


class GPPrior:
    """
    Generate data using Gaussian Processes.
    
    GPs provide smooth function samples with controllable properties.
    Different kernel parameters create different types of functions.
    
    This prior tends to create smooth, continuous decision boundaries.
    """
    
    def __init__(
        self,
        lengthscale_range: Tuple[float, float] = (0.1, 2.0),
        outputscale_range: Tuple[float, float] = (0.5, 2.0),
        noise_std: float = 0.01,
    ):
        self.lengthscale_range = lengthscale_range
        self.outputscale_range = outputscale_range
        self.noise_std = noise_std
    
    def _rbf_kernel(
        self, 
        X1: np.ndarray, 
        X2: np.ndarray, 
        lengthscale: float,
        outputscale: float,
    ) -> np.ndarray:
        """Compute RBF (squared exponential) kernel matrix."""
        # Compute squared distances
        X1_sq = np.sum(X1**2, axis=1, keepdims=True)
        X2_sq = np.sum(X2**2, axis=1, keepdims=True)
        dist_sq = X1_sq + X2_sq.T - 2 * X1 @ X2.T
        
        # Apply RBF kernel
        K = outputscale * np.exp(-0.5 * dist_sq / lengthscale**2)
        return K
    
    def sample_function_values(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Sample function values at given points from a GP."""
        n_samples = X.shape[0]
        
        # Sample kernel hyperparameters
        lengthscale = random.uniform(*self.lengthscale_range)
        outputscale = random.uniform(*self.outputscale_range)
        
        # Compute kernel matrix
        K = self._rbf_kernel(X, X, lengthscale, outputscale)
        
        # Add jitter for numerical stability
        K += np.eye(n_samples) * 1e-6
        
        # Sample from GP (multivariate normal with covariance K)
        try:
            L = np.linalg.cholesky(K)
            z = np.random.randn(n_samples)
            f = L @ z
        except np.linalg.LinAlgError:
            # Fallback if Cholesky fails
            f = np.random.randn(n_samples)
        
        return f
    
    def generate(
        self,
        n_samples: int,
        n_features: int,
        n_classes: int = 2,
        train_ratio: float = 0.7,
    ) -> SyntheticDataset:
        """Generate a synthetic classification dataset."""
        # Sample input features (uniform for GP)
        X = np.random.uniform(-3, 3, (n_samples, n_features))
        
        # Sample GP function values for each class
        logits = np.zeros((n_samples, n_classes))
        for c in range(n_classes):
            logits[:, c] = self.sample_function_values(X)
        
        # Clip to prevent overflow and handle NaN/Inf
        logits = np.clip(logits, -1e6, 1e6)
        logits = np.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Add noise and convert to classes
        logits += np.random.randn(*logits.shape) * self.noise_std
        y = logits.argmax(axis=1)
        
        # Determine train/test split
        train_size = max(1, int(n_samples * train_ratio))
        train_size = min(train_size, n_samples - 1)
        
        return SyntheticDataset(
            X=X.astype(np.float32),
            y=y.astype(np.int64),
            train_size=train_size,
            n_classes=n_classes,
        )


class TreePrior:
    """
    Generate data using random decision trees.
    
    Creates axis-aligned decision boundaries by recursively splitting
    feature space. Good for capturing discrete/rule-based patterns.
    """
    
    def __init__(
        self,
        max_depth_range: Tuple[int, int] = (2, 6),
        noise_std: float = 0.01,
    ):
        self.max_depth_range = max_depth_range
        self.noise_std = noise_std
    
    def _build_tree(
        self,
        n_features: int,
        n_classes: int,
        max_depth: int,
        depth: int = 0,
    ) -> Dict[str, Any]:
        """Recursively build a random decision tree."""
        # Decide whether to split or create leaf
        if depth >= max_depth or random.random() < 0.3:
            # Leaf node: random class probabilities
            probs = np.random.dirichlet(np.ones(n_classes) * 0.5)
            return {'type': 'leaf', 'probs': probs}
        
        # Internal node: random split
        feature_idx = random.randint(0, n_features - 1)
        threshold = random.uniform(-2, 2)
        
        left = self._build_tree(n_features, n_classes, max_depth, depth + 1)
        right = self._build_tree(n_features, n_classes, max_depth, depth + 1)
        
        return {
            'type': 'split',
            'feature': feature_idx,
            'threshold': threshold,
            'left': left,
            'right': right,
        }
    
    def _evaluate_tree(
        self,
        tree: Dict[str, Any],
        X: np.ndarray,
        n_classes: int,
    ) -> np.ndarray:
        """Evaluate decision tree on data."""
        n_samples = X.shape[0]
        
        if tree['type'] == 'leaf':
            return np.tile(tree['probs'], (n_samples, 1))
        
        # Evaluate split
        mask = X[:, tree['feature']] < tree['threshold']
        
        result = np.zeros((n_samples, n_classes))
        
        if mask.any():
            result[mask] = self._evaluate_tree(tree['left'], X[mask], n_classes)
        if (~mask).any():
            result[~mask] = self._evaluate_tree(tree['right'], X[~mask], n_classes)
        
        return result
    
    def generate(
        self,
        n_samples: int,
        n_features: int,
        n_classes: int = 2,
        train_ratio: float = 0.7,
    ) -> SyntheticDataset:
        """Generate a synthetic classification dataset."""
        # Sample features
        X = np.random.uniform(-3, 3, (n_samples, n_features))
        
        # Build and evaluate random tree
        max_depth = random.randint(*self.max_depth_range)
        tree = self._build_tree(n_features, n_classes, max_depth)
        probs = self._evaluate_tree(tree, X, n_classes)
        
        # Ensure probs has correct shape
        if probs.shape[1] != n_classes:
            probs = np.random.dirichlet(np.ones(n_classes), n_samples)
        
        # Add noise and sample classes
        probs += np.random.randn(*probs.shape) * self.noise_std
        probs = np.clip(probs, 1e-10, None)
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        y = np.array([np.random.choice(n_classes, p=p) for p in probs])
        
        # Determine train/test split
        train_size = max(1, int(n_samples * train_ratio))
        train_size = min(train_size, n_samples - 1)
        
        return SyntheticDataset(
            X=X.astype(np.float32),
            y=y.astype(np.int64),
            train_size=train_size,
            n_classes=n_classes,
        )


class SCMPrior:
    """
    Generate data using Structural Causal Models.
    
    Creates a random DAG and assigns random causal mechanisms to each node.
    This captures causal relationships between features and target.
    
    More realistic but also more complex than other priors.
    """
    
    def __init__(
        self,
        mechanism_types: List[str] = ['linear', 'mlp', 'polynomial'],
        noise_std: float = 0.1,
        edge_prob: float = 0.3,
    ):
        self.mechanism_types = mechanism_types
        self.noise_std = noise_std
        self.edge_prob = edge_prob
    
    def _sample_dag(self, n_nodes: int) -> np.ndarray:
        """Sample a random DAG adjacency matrix."""
        # Lower triangular ensures DAG (no cycles)
        adj = np.random.rand(n_nodes, n_nodes) < self.edge_prob
        adj = np.tril(adj, k=-1).astype(float)
        return adj
    
    def _sample_mechanism(
        self, 
        mechanism_type: str,
        n_parents: int,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Sample a random causal mechanism."""
        if mechanism_type == 'linear':
            weights = np.random.randn(n_parents) * 0.5
            bias = np.random.randn() * 0.5
            return lambda x: x @ weights + bias
        
        elif mechanism_type == 'polynomial':
            degree = random.randint(1, 3)
            coeffs = [np.random.randn(n_parents) * (0.5 ** d) for d in range(degree + 1)]
            def poly(x):
                # Clip x to prevent overflow in power operation
                x_clipped = np.clip(x, -10, 10)
                result = coeffs[0].sum()  # Bias
                for d, c in enumerate(coeffs[1:], 1):
                    x_pow = np.clip(x_clipped ** d, -1e6, 1e6)
                    result += (x_pow @ c)
                return np.clip(result, -1e6, 1e6)
            return poly
        
        else:  # mlp
            hidden_size = random.randint(8, 32)
            W1 = np.random.randn(n_parents, hidden_size) * 0.5
            b1 = np.random.randn(hidden_size) * 0.1
            W2 = np.random.randn(hidden_size) * 0.5
            b2 = np.random.randn() * 0.1
            def mlp(x):
                h = np.maximum(0, x @ W1 + b1)  # ReLU
                return h @ W2 + b2
            return mlp
    
    def generate(
        self,
        n_samples: int,
        n_features: int,
        n_classes: int = 2,
        train_ratio: float = 0.7,
    ) -> SyntheticDataset:
        """Generate a synthetic classification dataset."""
        n_nodes = n_features + 1  # Features + target
        
        # Sample DAG structure
        adj = self._sample_dag(n_nodes)
        
        # Sample mechanisms for each node
        mechanisms = []
        for i in range(n_nodes):
            parents = np.where(adj[i] > 0)[0]
            if len(parents) == 0:
                # No parents: sample from prior
                mechanisms.append(lambda x: np.random.randn(n_samples))
            else:
                mech_type = random.choice(self.mechanism_types)
                mechanisms.append(self._sample_mechanism(mech_type, len(parents)))
        
        # Generate data by ancestral sampling
        data = np.zeros((n_samples, n_nodes))
        
        for i in range(n_nodes):
            parents = np.where(adj[i] > 0)[0]
            if len(parents) == 0:
                data[:, i] = np.random.randn(n_samples)
            else:
                parent_values = data[:, parents]
                data[:, i] = mechanisms[i](parent_values) + np.random.randn(n_samples) * self.noise_std
        
        # Split into features and target
        X = data[:, :n_features]
        target_continuous = data[:, -1]
        
        # Clip to prevent overflow when converting to float32
        X = np.clip(X, -1e6, 1e6)
        target_continuous = np.clip(target_continuous, -1e6, 1e6)
        
        # Handle any NaN/Inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        target_continuous = np.nan_to_num(target_continuous, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Convert to classes
        if n_classes == 2:
            y = (target_continuous > np.median(target_continuous)).astype(np.int64)
        else:
            percentiles = np.linspace(0, 100, n_classes + 1)[1:-1]
            thresholds = np.percentile(target_continuous, percentiles)
            y = np.digitize(target_continuous, thresholds)
        
        # Determine train/test split
        train_size = max(1, int(n_samples * train_ratio))
        train_size = min(train_size, n_samples - 1)
        
        return SyntheticDataset(
            X=X.astype(np.float32),
            y=y.astype(np.int64),
            train_size=train_size,
            n_classes=n_classes,
        )


class MixedPrior:
    """
    Mixture of multiple priors for diverse training data.
    
    Randomly selects from available priors for each dataset.
    This increases the diversity of training data.
    """
    
    def __init__(
        self,
        priors: Optional[List[Any]] = None,
        weights: Optional[List[float]] = None,
    ):
        if priors is None:
            priors = [MLPPrior(), GPPrior(), TreePrior(), SCMPrior()]
        if weights is None:
            weights = [1.0] * len(priors)
        
        self.priors = priors
        self.weights = np.array(weights) / sum(weights)
    
    def generate(
        self,
        n_samples: int,
        n_features: int,
        n_classes: int = 2,
        train_ratio: float = 0.7,
    ) -> SyntheticDataset:
        """Generate a synthetic dataset using a random prior."""
        prior_idx = np.random.choice(len(self.priors), p=self.weights)
        return self.priors[prior_idx].generate(
            n_samples, n_features, n_classes, train_ratio
        )


# ============================================================================
# Regression Priors
# ============================================================================

class MLPRegressionPrior:
    """
    Generate regression data using random Multi-Layer Perceptrons.
    
    Similar to MLPPrior but outputs continuous values instead of class labels.
    """
    
    def __init__(
        self,
        n_layers_range: Tuple[int, int] = (1, 4),
        hidden_size_range: Tuple[int, int] = (16, 128),
        activation_options: List[str] = ['relu', 'tanh', 'gelu', 'sigmoid'],
        weight_scale: float = 1.0,
        noise_std_range: Tuple[float, float] = (0.01, 0.3),
    ):
        self.n_layers_range = n_layers_range
        self.hidden_size_range = hidden_size_range
        self.activation_options = activation_options
        self.weight_scale = weight_scale
        self.noise_std_range = noise_std_range
    
    def _get_activation(self, name: str) -> Callable:
        activations = {
            'relu': lambda x: np.maximum(0, x),
            'tanh': np.tanh,
            'gelu': lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))),
            'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
            'leaky_relu': lambda x: np.where(x > 0, x, 0.01 * x),
        }
        return activations.get(name, activations['relu'])
    
    def sample_function(self, n_features: int) -> Callable[[np.ndarray], np.ndarray]:
        """Sample a random MLP function for regression."""
        n_layers = random.randint(*self.n_layers_range)
        hidden_sizes = [random.randint(*self.hidden_size_range) for _ in range(n_layers)]
        activation = self._get_activation(random.choice(self.activation_options))
        
        layer_sizes = [n_features] + hidden_sizes + [1]
        weights = []
        biases = []
        
        for i in range(len(layer_sizes) - 1):
            scale = self.weight_scale * np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            b = np.random.randn(layer_sizes[i+1]) * scale * 0.1
            weights.append(W)
            biases.append(b)
        
        def forward(X: np.ndarray) -> np.ndarray:
            h = X
            for i, (W, b) in enumerate(zip(weights, biases)):
                h = h @ W + b
                if i < len(weights) - 1:
                    h = activation(h)
            return h.squeeze(-1)
        
        return forward
    
    def generate(
        self,
        n_samples: int,
        n_features: int,
        train_ratio: float = 0.7,
    ) -> SyntheticDataset:
        """Generate a synthetic regression dataset."""
        # Sample input features
        X = self._sample_features(n_samples, n_features)
        
        # Sample and apply random function
        func = self.sample_function(n_features)
        y = func(X)
        
        # Add noise
        noise_std = random.uniform(*self.noise_std_range)
        y += np.random.randn(n_samples) * noise_std * (y.std() + 1e-6)
        
        # Handle NaN/Inf
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.clip(y, -1e6, 1e6)
        
        # Determine train/test split
        train_size = max(1, int(n_samples * train_ratio))
        train_size = min(train_size, n_samples - 1)
        
        return SyntheticDataset(
            X=X.astype(np.float32),
            y=y.astype(np.float32),
            train_size=train_size,
            n_classes=0,
            is_regression=True,
        )
    
    def _sample_features(self, n_samples: int, n_features: int) -> np.ndarray:
        """Sample feature values from various distributions."""
        X = np.zeros((n_samples, n_features))
        
        for i in range(n_features):
            dist_type = random.choice(['uniform', 'normal', 'mixture'])
            
            if dist_type == 'uniform':
                low = random.uniform(-10, 0)
                high = random.uniform(0, 10)
                X[:, i] = np.random.uniform(low, high, n_samples)
            elif dist_type == 'normal':
                mean = random.uniform(-5, 5)
                std = random.uniform(0.1, 3)
                X[:, i] = np.random.normal(mean, std, n_samples)
            else:
                n_components = random.randint(2, 4)
                means = np.random.uniform(-5, 5, n_components)
                stds = np.random.uniform(0.1, 2, n_components)
                component_idx = np.random.choice(n_components, n_samples)
                X[:, i] = np.array([
                    np.random.normal(means[c], stds[c])
                    for c in component_idx
                ])
        
        return X


class GPRegressionPrior:
    """
    Generate regression data using Gaussian Processes.
    
    GPs provide smooth function samples ideal for regression.
    """
    
    def __init__(
        self,
        lengthscale_range: Tuple[float, float] = (0.1, 2.0),
        outputscale_range: Tuple[float, float] = (0.5, 2.0),
        noise_std_range: Tuple[float, float] = (0.01, 0.2),
    ):
        self.lengthscale_range = lengthscale_range
        self.outputscale_range = outputscale_range
        self.noise_std_range = noise_std_range
    
    def _rbf_kernel(
        self, 
        X1: np.ndarray, 
        X2: np.ndarray, 
        lengthscale: float,
        outputscale: float,
    ) -> np.ndarray:
        X1_sq = np.sum(X1**2, axis=1, keepdims=True)
        X2_sq = np.sum(X2**2, axis=1, keepdims=True)
        dist_sq = X1_sq + X2_sq.T - 2 * X1 @ X2.T
        K = outputscale * np.exp(-0.5 * dist_sq / lengthscale**2)
        return K
    
    def sample_function_values(self, X: np.ndarray) -> np.ndarray:
        """Sample function values at given points from a GP."""
        n_samples = X.shape[0]
        
        lengthscale = random.uniform(*self.lengthscale_range)
        outputscale = random.uniform(*self.outputscale_range)
        
        K = self._rbf_kernel(X, X, lengthscale, outputscale)
        K += np.eye(n_samples) * 1e-6
        
        try:
            L = np.linalg.cholesky(K)
            z = np.random.randn(n_samples)
            f = L @ z
        except np.linalg.LinAlgError:
            f = np.random.randn(n_samples)
        
        return f
    
    def generate(
        self,
        n_samples: int,
        n_features: int,
        train_ratio: float = 0.7,
    ) -> SyntheticDataset:
        """Generate a synthetic regression dataset."""
        X = np.random.uniform(-3, 3, (n_samples, n_features))
        
        y = self.sample_function_values(X)
        
        # Add noise
        noise_std = random.uniform(*self.noise_std_range)
        y += np.random.randn(n_samples) * noise_std
        
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.clip(y, -1e6, 1e6)
        
        train_size = max(1, int(n_samples * train_ratio))
        train_size = min(train_size, n_samples - 1)
        
        return SyntheticDataset(
            X=X.astype(np.float32),
            y=y.astype(np.float32),
            train_size=train_size,
            n_classes=0,
            is_regression=True,
        )


class LinearRegressionPrior:
    """
    Generate regression data using random linear functions with optional
    polynomial features and interactions.
    """
    
    def __init__(
        self,
        include_interactions: bool = True,
        max_polynomial_degree: int = 3,
        noise_std_range: Tuple[float, float] = (0.01, 0.3),
    ):
        self.include_interactions = include_interactions
        self.max_polynomial_degree = max_polynomial_degree
        self.noise_std_range = noise_std_range
    
    def generate(
        self,
        n_samples: int,
        n_features: int,
        train_ratio: float = 0.7,
    ) -> SyntheticDataset:
        """Generate a synthetic regression dataset."""
        X = np.random.uniform(-3, 3, (n_samples, n_features))
        
        # Random coefficients
        coeffs = np.random.randn(n_features) * 2
        
        # Linear term
        y = X @ coeffs
        
        # Add polynomial terms
        if self.max_polynomial_degree > 1:
            degree = random.randint(2, self.max_polynomial_degree)
            for d in range(2, degree + 1):
                # Select random features for polynomial
                n_poly_features = random.randint(1, min(3, n_features))
                poly_features = np.random.choice(n_features, n_poly_features, replace=False)
                for f in poly_features:
                    coeff = np.random.randn() * 0.5
                    y += coeff * (X[:, f] ** d)
        
        # Add interaction terms
        if self.include_interactions and n_features > 1:
            n_interactions = random.randint(1, min(5, n_features * (n_features - 1) // 2))
            for _ in range(n_interactions):
                f1, f2 = np.random.choice(n_features, 2, replace=False)
                coeff = np.random.randn() * 0.3
                y += coeff * X[:, f1] * X[:, f2]
        
        # Add noise
        noise_std = random.uniform(*self.noise_std_range)
        y += np.random.randn(n_samples) * noise_std * (y.std() + 1e-6)
        
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.clip(y, -1e6, 1e6)
        
        train_size = max(1, int(n_samples * train_ratio))
        train_size = min(train_size, n_samples - 1)
        
        return SyntheticDataset(
            X=X.astype(np.float32),
            y=y.astype(np.float32),
            train_size=train_size,
            n_classes=0,
            is_regression=True,
        )


class MixedRegressionPrior:
    """
    Mixture of multiple regression priors for diverse training data.
    """
    
    def __init__(
        self,
        priors: Optional[List[Any]] = None,
        weights: Optional[List[float]] = None,
    ):
        if priors is None:
            priors = [MLPRegressionPrior(), GPRegressionPrior(), LinearRegressionPrior()]
        if weights is None:
            weights = [1.0] * len(priors)
        
        self.priors = priors
        self.weights = np.array(weights) / sum(weights)
    
    def generate(
        self,
        n_samples: int,
        n_features: int,
        train_ratio: float = 0.7,
    ) -> SyntheticDataset:
        """Generate a synthetic regression dataset using a random prior."""
        prior_idx = np.random.choice(len(self.priors), p=self.weights)
        return self.priors[prior_idx].generate(n_samples, n_features, train_ratio)


def get_prior(prior_type: str, **kwargs) -> Any:
    """Get a prior by name.
    
    Note: Following TabPFN, classification and regression are trained as separate models.
    Use 'mixed' for classification and 'mixed_regression' for regression.
    
    Augmented variants (e.g., 'augmented_mixed') add categorical features, 
    missing values, and noise following TabPFN's preprocessing approach.
    """
    # Extract augmentation parameters
    categorical_prob = kwargs.pop('categorical_prob', 0.3)
    missing_prob = kwargs.pop('missing_prob', 0.1)
    augment_prob = kwargs.pop('augment_prob', 0.7)
    
    priors = {
        # Classification priors
        'mlp': MLPPrior,
        'gp': GPPrior,
        'tree': TreePrior,
        'scm': SCMPrior,
        'mixed': MixedPrior,
        # Regression priors
        'mlp_regression': MLPRegressionPrior,
        'gp_regression': GPRegressionPrior,
        'linear_regression': LinearRegressionPrior,
        'mixed_regression': MixedRegressionPrior,
    }
    
    # Check for augmented variants
    if prior_type.startswith('augmented_'):
        base_type = prior_type[len('augmented_'):]
        if base_type not in priors:
            raise ValueError(f"Unknown base prior: {base_type}. Available: {list(priors.keys())}")
        
        base_prior = priors[base_type](**kwargs)
        
        # Check if it's a regression prior
        if 'regression' in base_type:
            return AugmentedRegressionPrior(
                base_prior=base_prior,
                categorical_prob=categorical_prob,
                missing_prob=missing_prob,
                augment_prob=augment_prob,
            )
        else:
            return AugmentedPrior(
                base_prior=base_prior,
                categorical_prob=categorical_prob,
                missing_prob=missing_prob,
                augment_prob=augment_prob,
            )
    
    if prior_type not in priors:
        raise ValueError(f"Unknown prior: {prior_type}. Available: {list(priors.keys())} + augmented_* variants")
    return priors[prior_type](**kwargs)


class SyntheticDataGenerator:
    """
    Generate and save synthetic datasets for training.
    
    Generates batches of synthetic datasets and saves them to HDF5 format
    for efficient loading during training.
    """
    
    def __init__(
        self,
        prior: Any,
        n_samples_range: Tuple[int, int] = (10, 100),
        n_features_range: Tuple[int, int] = (2, 20),
        n_classes_range: Tuple[int, int] = (2, 10),
        train_ratio_range: Tuple[float, float] = (0.5, 0.9),
    ):
        self.prior = prior
        self.n_samples_range = n_samples_range
        self.n_features_range = n_features_range
        self.n_classes_range = n_classes_range
        self.train_ratio_range = train_ratio_range
    
    def generate_batch(
        self,
        batch_size: int,
        max_samples: int,
        max_features: int,
        max_classes: int,
    ) -> Dict[str, np.ndarray]:
        """Generate a batch of synthetic datasets.
        
        Returns:
            Dictionary with:
            - X: (batch, max_samples, max_features) feature tensor
            - y: (batch, max_samples) target tensor
            - train_size: (batch,) train/test split positions
            - n_features: (batch,) actual number of features
            - n_samples: (batch,) actual number of samples
        """
        X_batch = np.zeros((batch_size, max_samples, max_features), dtype=np.float32)
        y_batch = np.zeros((batch_size, max_samples), dtype=np.float32)
        train_sizes = np.zeros(batch_size, dtype=np.int32)
        n_features_list = np.zeros(batch_size, dtype=np.int32)
        n_samples_list = np.zeros(batch_size, dtype=np.int32)
        
        for i in range(batch_size):
            # Sample dataset parameters
            n_samples = random.randint(*self.n_samples_range)
            n_samples = min(n_samples, max_samples)
            n_features = random.randint(*self.n_features_range)
            n_features = min(n_features, max_features)
            n_classes = random.randint(*self.n_classes_range)
            n_classes = min(n_classes, max_classes)
            train_ratio = random.uniform(*self.train_ratio_range)
            
            # Generate dataset
            dataset = self.prior.generate(n_samples, n_features, n_classes, train_ratio)
            
            # Store in batch arrays (with padding)
            X_batch[i, :n_samples, :n_features] = dataset.X
            y_batch[i, :n_samples] = dataset.y
            train_sizes[i] = dataset.train_size
            n_features_list[i] = n_features
            n_samples_list[i] = n_samples
        
        return {
            'X': X_batch,
            'y': y_batch,
            'train_size': train_sizes,
            'n_features': n_features_list,
            'n_samples': n_samples_list,
        }
    
    def generate_and_save(
        self,
        output_path: str,
        n_datasets: int,
        batch_size: int = 1000,
        max_samples: int = 100,
        max_features: int = 20,
        max_classes: int = 10,
    ):
        """Generate datasets and save to HDF5 file."""
        n_batches = (n_datasets + batch_size - 1) // batch_size
        
        with h5py.File(output_path, 'w') as f:
            # Create datasets with chunked storage
            f.create_dataset('X', shape=(0, max_samples, max_features),
                           maxshape=(None, max_samples, max_features),
                           chunks=(batch_size, max_samples, max_features),
                           dtype='float32', compression='lzf')
            f.create_dataset('y', shape=(0, max_samples),
                           maxshape=(None, max_samples),
                           chunks=(batch_size, max_samples),
                           dtype='float32')
            f.create_dataset('single_eval_pos', shape=(0,),
                           maxshape=(None,), chunks=(batch_size,), dtype='int32')
            f.create_dataset('num_features', shape=(0,),
                           maxshape=(None,), chunks=(batch_size,), dtype='int32')
            f.create_dataset('num_datapoints', shape=(0,),
                           maxshape=(None,), chunks=(batch_size,), dtype='int32')
            
            # Metadata
            f.create_dataset('max_num_classes', data=np.array([max_classes]))
            f.create_dataset('problem_type', data='classification')
            
            # Generate batches
            total_generated = 0
            pbar = tqdm(range(n_batches), desc="Generating datasets")
            
            for _ in pbar:
                current_batch_size = min(batch_size, n_datasets - total_generated)
                if current_batch_size <= 0:
                    break
                
                batch = self.generate_batch(
                    current_batch_size, max_samples, max_features, max_classes
                )
                
                # Append to HDF5
                n = f['X'].shape[0]
                for key in ['X', 'y']:
                    f[key].resize(n + current_batch_size, axis=0)
                    f[key][n:n+current_batch_size] = batch[key]
                
                f['single_eval_pos'].resize(n + current_batch_size, axis=0)
                f['single_eval_pos'][n:n+current_batch_size] = batch['train_size']
                
                f['num_features'].resize(n + current_batch_size, axis=0)
                f['num_features'][n:n+current_batch_size] = batch['n_features']
                
                f['num_datapoints'].resize(n + current_batch_size, axis=0)
                f['num_datapoints'][n:n+current_batch_size] = batch['n_samples']
                
                total_generated += current_batch_size
                pbar.set_postfix({'total': total_generated})
        
        print(f"Generated {total_generated} datasets, saved to {output_path}")


def visualize_prior(prior, n_samples: int = 200, n_features: int = 2, n_classes: int = 2):
    """Visualize samples from a prior (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    for i, ax in enumerate(axes.flat):
        dataset = prior.generate(n_samples, n_features, n_classes)
        
        if n_features >= 2:
            for c in range(n_classes):
                mask = dataset.y == c
                ax.scatter(dataset.X[mask, 0], dataset.X[mask, 1], 
                          alpha=0.6, label=f'Class {c}', s=20)
        else:
            for c in range(n_classes):
                mask = dataset.y == c
                ax.scatter(dataset.X[mask, 0], np.zeros_like(dataset.X[mask, 0]) + c*0.1,
                          alpha=0.6, label=f'Class {c}', s=20)
        
        ax.set_title(f'Sample {i+1}')
        ax.legend(fontsize=8)
    
    plt.suptitle(f'{prior.__class__.__name__} Samples', fontsize=14)
    plt.tight_layout()
    plt.savefig('prior_samples.png', dpi=150)
    plt.show()
    print("Saved visualization to prior_samples.png")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic data for training')
    
    parser.add_argument('--output', '-o', type=str, default='data/synthetic.h5',
                       help='Output HDF5 file path')
    parser.add_argument('--n_datasets', '-n', type=int, default=100000,
                       help='Number of datasets to generate')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='Batch size for generation')
    parser.add_argument('--max_samples', type=int, default=100,
                       help='Maximum samples per dataset')
    parser.add_argument('--max_features', type=int, default=20,
                       help='Maximum features per dataset')
    parser.add_argument('--max_classes', type=int, default=10,
                       help='Maximum number of classes')
    parser.add_argument('--prior', type=str, default='mixed',
                       help='Prior type to use (base: mlp, gp, tree, scm, mixed; augmented: augmented_*; regression: *_regression)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize prior samples instead of generating data')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Get prior
    prior = get_prior(args.prior)
    
    if args.visualize:
        visualize_prior(prior)
        return
    
    # Generate data
    import os
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    generator = SyntheticDataGenerator(prior)
    generator.generate_and_save(
        output_path=args.output,
        n_datasets=args.n_datasets,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_features=args.max_features,
        max_classes=args.max_classes,
    )


if __name__ == '__main__':
    main()
