"""
model.py - OpenTab Transformer Architecture

A implementation of the TabPFN (Prior-Data Fitted Network) architecture.
The model processes tabular data as a 3D tensor: (batch, rows, features+target)
and applies attention across both dimensions.

Architecture Overview:
1. FeatureEncoder: Embed each feature value into a dense vector
2. TargetEncoder: Embed target values (with padding for test samples)
3. TransformerEncoder: Apply attention across features and samples
4. Decoder: Map embeddings to class predictions

References:
- TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second
- nanoTabPFN: A Lightweight and Educational Reimplementation of TabPFN
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union


class FeatureEncoder(nn.Module):
    """Encodes scalar feature values into dense embeddings.
    
    Each feature value is normalized based on training statistics,
    then projected to an embedding space.
    """
    
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear = nn.Linear(1, embedding_size)
    
    def forward(self, x: torch.Tensor, train_size: int) -> torch.Tensor:
        """
        Args:
            x: (batch, rows, features) tensor of feature values
            train_size: number of training samples (for normalization)
            
        Returns:
            (batch, rows, features, embedding_size) tensor of embeddings
        """
        # Normalize features based on training data statistics
        x = x.unsqueeze(-1)  # (batch, rows, features, 1)
        
        # Compute mean and std from training portion only
        train_x = x[:, :train_size]
        mean = train_x.mean(dim=1, keepdim=True)
        std = train_x.std(dim=1, keepdim=True) + 1e-8
        
        # Normalize all data (train + test) using training statistics
        x = (x - mean) / std
        x = torch.clamp(x, -100, 100)  # Prevent extreme values
        
        return self.linear(x)


class TargetEncoder(nn.Module):
    """Encodes target values into dense embeddings.
    
    Training targets are embedded directly. Test positions are padded
    with the mean target value (since we don't know the true labels).
    """
    
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear = nn.Linear(1, embedding_size)
    
    def forward(self, y_train: torch.Tensor, total_rows: int) -> torch.Tensor:
        """
        Args:
            y_train: (batch, train_size) tensor of training labels
            total_rows: total number of rows (train + test)
            
        Returns:
            (batch, total_rows, 1, embedding_size) tensor of embeddings
        """
        # Convert to float for embedding
        y_train = y_train.float()
        
        # Add feature dimension
        if y_train.dim() == 2:
            y_train = y_train.unsqueeze(-1)
        
        # Pad test positions with mean of training targets
        train_size = y_train.shape[1]
        test_size = total_rows - train_size
        
        if test_size > 0:
            mean_y = y_train.mean(dim=1, keepdim=True)
            padding = mean_y.expand(-1, test_size, -1)
            y_full = torch.cat([y_train, padding], dim=1)
        else:
            y_full = y_train
        
        # Embed: (batch, rows, 1) -> (batch, rows, 1, embedding_size)
        return self.linear(y_full.unsqueeze(-1))


class TransformerEncoderLayer(nn.Module):
    """A single transformer layer with attention over both features and samples.
    
    Unlike standard transformers, TabPFN applies attention in two directions:
    1. Across features (column-wise): features can interact
    2. Across samples (row-wise): with causal masking for train→test
    
    The attention pattern ensures test samples can attend to training samples
    but not to other test samples (to prevent data leakage).
    """
    
    def __init__(
        self,
        embedding_size: int,
        n_heads: int,
        mlp_hidden_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Attention across features (columns)
        self.attn_features = nn.MultiheadAttention(
            embedding_size, n_heads, batch_first=True, dropout=dropout
        )
        
        # Attention across samples (rows)
        self.attn_samples = nn.MultiheadAttention(
            embedding_size, n_heads, batch_first=True, dropout=dropout
        )
        
        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, mlp_hidden_size),
            nn.GELU(),
            nn.Linear(mlp_hidden_size, embedding_size),
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.norm3 = nn.LayerNorm(embedding_size)
    
    def forward(self, x: torch.Tensor, train_size: int) -> torch.Tensor:
        """
        Args:
            x: (batch, rows, cols, embedding_size) tensor
            train_size: number of training samples (for attention masking)
            
        Returns:
            (batch, rows, cols, embedding_size) tensor
        """
        batch_size, n_rows, n_cols, emb_size = x.shape
        
        # 1. Attention across features (for each row)
        # Reshape: (batch * rows, cols, emb)
        x_flat = x.reshape(batch_size * n_rows, n_cols, emb_size)
        attn_out, _ = self.attn_features(x_flat, x_flat, x_flat)
        x_flat = x_flat + attn_out
        x = x_flat.reshape(batch_size, n_rows, n_cols, emb_size)
        x = self.norm1(x)
        
        # 2. Attention across samples (for each feature)
        # Reshape: (batch * cols, rows, emb)
        x = x.transpose(1, 2)  # (batch, cols, rows, emb)
        x_flat = x.reshape(batch_size * n_cols, n_rows, emb_size)
        
        # Split into train and test portions
        x_train = x_flat[:, :train_size]
        x_test = x_flat[:, train_size:]
        
        # Training samples attend to themselves
        train_attn, _ = self.attn_samples(x_train, x_train, x_train)
        
        # Test samples attend to training samples only
        if x_test.shape[1] > 0:
            test_attn, _ = self.attn_samples(x_test, x_train, x_train)
            attn_out = torch.cat([train_attn, test_attn], dim=1)
        else:
            attn_out = train_attn
        
        x_flat = x_flat + attn_out
        x = x_flat.reshape(batch_size, n_cols, n_rows, emb_size)
        x = x.transpose(1, 2)  # Back to (batch, rows, cols, emb)
        x = self.norm2(x)
        
        # 3. MLP
        x = x + self.mlp(x)
        x = self.norm3(x)
        
        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""
    
    def __init__(
        self,
        embedding_size: int,
        n_heads: int,
        mlp_hidden_size: int,
        n_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embedding_size, n_heads, mlp_hidden_size, dropout)
            for _ in range(n_layers)
        ])
    
    def forward(self, x: torch.Tensor, train_size: int) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, train_size)
        return x


class Decoder(nn.Module):
    """Maps embeddings to output predictions."""
    
    def __init__(self, embedding_size: int, mlp_hidden_size: int, n_outputs: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, mlp_hidden_size),
            nn.GELU(),
            nn.Linear(mlp_hidden_size, n_outputs),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_test, embedding_size) tensor
            
        Returns:
            (batch, n_test, n_outputs) tensor of logits
        """
        return self.mlp(x)


class OpenTabModel(nn.Module):
    """
    OpenTab: A implementation of TabPFN for tabular classification.
    
    The model takes a table of (features, target) and predicts targets for
    test samples using in-context learning.
    
    Args:
        embedding_size: Dimension of embeddings
        n_heads: Number of attention heads
        mlp_hidden_size: Hidden size of MLP blocks
        n_layers: Number of transformer layers
        n_outputs: Number of output classes
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embedding_size: int = 96,
        n_heads: int = 4,
        mlp_hidden_size: int = 192,
        n_layers: int = 3,
        n_outputs: int = 10,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.n_outputs = n_outputs
        
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer = TransformerEncoder(
            embedding_size, n_heads, mlp_hidden_size, n_layers, dropout
        )
        self.decoder = Decoder(embedding_size, mlp_hidden_size, n_outputs)
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        train_size: int,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, n_rows, n_features) tensor of features
            y: (batch, train_size) tensor of training labels
            train_size: number of training samples
            
        Returns:
            (batch, n_test, n_outputs) tensor of logits for test samples
        """
        n_rows = x.shape[1]
        
        # Encode features: (batch, rows, features, emb)
        x_emb = self.feature_encoder(x, train_size)
        
        # Encode targets: (batch, rows, 1, emb)
        y_emb = self.target_encoder(y, n_rows)
        
        # Concatenate features and target: (batch, rows, features+1, emb)
        combined = torch.cat([x_emb, y_emb], dim=2)
        
        # Apply transformer
        transformed = self.transformer(combined, train_size)
        
        # Extract test embeddings from target column
        # Shape: (batch, n_test, emb)
        test_emb = transformed[:, train_size:, -1, :]
        
        # Decode to predictions
        logits = self.decoder(test_emb)
        
        return logits
    
    def forward_train_test(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience method matching sklearn interface.
        
        Args:
            X_train: (batch, n_train, n_features) training features
            y_train: (batch, n_train) training labels
            X_test: (batch, n_test, n_features) test features
            
        Returns:
            (batch, n_test, n_outputs) tensor of logits
        """
        train_size = X_train.shape[1]
        x = torch.cat([X_train, X_test], dim=1)
        return self.forward(x, y_train, train_size)


class OpenTabClassifier:
    """Sklearn-like interface for OpenTab classification.
    
    Usage:
        clf = OpenTabClassifier("checkpoint.pt")
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        probabilities = clf.predict_proba(X_test)
    """
    
    def __init__(
        self,
        model: Union[OpenTabModel, str, None] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            model: Either a OpenTabModel instance or path to checkpoint
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        if model is None:
            # Create default model
            self.model = OpenTabModel().to(self.device)
        elif isinstance(model, str):
            # Load from checkpoint (weights_only=False for numpy arrays in config)
            checkpoint = torch.load(model, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                # Full checkpoint from train.py
                config = checkpoint.get('config', {})
                self.model = OpenTabModel(
                    embedding_size=config.get('embedding_size', 96),
                    n_heads=config.get('n_heads', 4),
                    mlp_hidden_size=config.get('mlp_hidden', 192),
                    n_layers=config.get('n_layers', 3),
                    n_outputs=config.get('max_classes', 10),
                    dropout=config.get('dropout', 0.0),
                ).to(self.device)
                self.model.load_state_dict(checkpoint['model_state'])
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                # Alternative checkpoint format
                self.model = OpenTabModel(
                    **checkpoint.get('architecture', {})
                ).to(self.device)
                self.model.load_state_dict(checkpoint['model'])
            else:
                # Just state dict
                self.model = OpenTabModel().to(self.device)
                self.model.load_state_dict(checkpoint)
        else:
            self.model = model.to(self.device)
        
        self.model.eval()
        
        # Storage for fit data
        self.X_train_ = None
        self.y_train_ = None
        self.n_classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Store training data for later prediction.
        
        Note: TabPFN doesn't actually "train" during fit - it just stores
        the data for use during prediction (in-context learning).
        """
        self.X_train_ = X.astype(np.float32)
        self.y_train_ = y.astype(np.int64)
        self.n_classes_ = int(y.max()) + 1
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for test samples."""
        if self.X_train_ is None:
            raise ValueError("Must call fit() before predict()")
        
        X_test = X.astype(np.float32)
        
        with torch.no_grad():
            # Add batch dimension
            X_train_t = torch.from_numpy(self.X_train_).unsqueeze(0).to(self.device)
            y_train_t = torch.from_numpy(self.y_train_).float().unsqueeze(0).to(self.device)
            X_test_t = torch.from_numpy(X_test).unsqueeze(0).to(self.device)
            
            # Forward pass
            logits = self.model.forward_train_test(X_train_t, y_train_t, X_test_t)
            
            # Get probabilities for actual classes
            logits = logits[:, :, :self.n_classes_]
            probs = F.softmax(logits, dim=-1)
            
            return probs.squeeze(0).cpu().numpy()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for test samples."""
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    print("Testing OpenTab model...")
    
    model = OpenTabModel(
        embedding_size=96,
        n_heads=4,
        mlp_hidden_size=192,
        n_layers=3,
        n_outputs=10,
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create dummy data
    batch_size = 2
    n_train = 50
    n_test = 10
    n_features = 5
    
    X_train = torch.randn(batch_size, n_train, n_features)
    y_train = torch.randint(0, 3, (batch_size, n_train)).float()
    X_test = torch.randn(batch_size, n_test, n_features)
    
    # Forward pass
    logits = model.forward_train_test(X_train, y_train, X_test)
    
    print(f"Input shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}")
    print(f"Output shape: {logits.shape}")
    print("✓ Model test passed!")
