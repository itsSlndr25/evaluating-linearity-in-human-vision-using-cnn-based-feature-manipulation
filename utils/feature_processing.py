"""
VGG 特徵的標準化、修改、反標準化
Feature processing utilities for VGG features.

This module handles the transformation of VGG features including:
- Normalization using batch statistics (running mean & running std)
- Feature modification in PCA space
- Denormalization back to original scale
"""

import numpy as np
from typing import Tuple, Optional


def normalize_features(
    features: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray
) -> np.ndarray:
    """
    Normalize features using batch statistics.
    
    This is a critical step because different filters have different scales.
    (因為不同的filter有不同的scale
    
    Args:
        features: Raw features from VGG layer
                  Shape: (batch, n_filters, height, width)
        means: Running mean from batch normalization layer
               Shape: (1, n_filters, 1, 1)
        stds: Running std from batch normalization layer
              Shape: (1, n_filters, 1, 1)
    
    Returns:
        Normalized features with shape (batch*height*width, n_filters)
    """
    batch_size, n_filters, height, width = features.shape
    
    # Normalize: (x - mean) / std
    normalized = (features - means) / stds
    
    # Reshape for PCA: move channel to last dimension and flatten spatial dims
    # 為 PCA 重塑：將channel移到最後一維，並flatten
    # (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C)
    normalized = normalized.transpose(0, 2, 3, 1)  # (B, H, W, C)
    normalized = normalized.reshape(-1, n_filters)  # (B*H*W, C)
    
    return normalized


def denormalize_features(
    features: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    original_shape: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Denormalize features back to original scale.
    
    Args:
        features: Normalized features
                  Shape: (batch*height*width, n_filters)
        means: Running mean from batch normalization
        stds: Running std from batch normalization
        original_shape: Target shape (batch, n_filters, height, width)
    
    Returns:
        Denormalized features in original shape
    """
    batch_size, n_filters, height, width = original_shape
    
    # Reshape back to spatial format：(B*H*W, C) -> (B, H, W, C)
    features = features.reshape(batch_size, height, width, n_filters)
    
    # Move channel back to 2nd dimension：(B, H, W, C) -> (B, C, H, W)
    features = features.transpose(0, 3, 1, 2)
    
    # Denormalize: x * std + mean
    denormalized = features * stds + means
    
    return denormalized


def modify_feature_pc(
    pca_features: np.ndarray,
    pc_index: int,
    magnitude: float
) -> np.ndarray:
    """
    Modify a specific principal component by a magnitude factor.
    
    Args:
        pca_features: Features in PCA space
                      Shape: (n_samples, n_components)
        pc_index: Index of the PC to modify (0-indexed)
                  要修改的主成分index（從 0 開始）
        magnitude: Multiplication factor (e.g., 2.0, 0.5, -1.0)
    
    Returns:
        Modified PCA features
    """
    modified = pca_features.copy()
    modified[:, pc_index] *= magnitude
    return modified


def modify_feature_pcs_additive(
    pca_features: np.ndarray,
    pc_indices: Tuple[int, int],
    magnitudes: Tuple[float, float]
) -> np.ndarray:
    """
    Modify two principal components for additivity testing.
    同時修改兩個主成分
    
    Args:
        pca_features: Features in PCA space
        pc_indices: Tuple of two PC indices to modify
        magnitudes: Tuple of two magnitude factors
    
    Returns:
        Modified PCA features
    """
    modified = pca_features.copy()
    modified[:, pc_indices[0]] *= magnitudes[0]
    modified[:, pc_indices[1]] *= magnitudes[1]
    return modified


def extract_single_filter_feature(
    feature: np.ndarray,
    filter_index: int
) -> np.ndarray:
    """
    Extract a single filter's response from the feature map.
    
    Args:
        feature: Full feature map
                 Shape: (batch, n_filters, height, width)
        filter_index: Index of filter to extract
    
    Returns:
        Feature with only one filter active
    """
    # Create a copy with all zeros
    single_filter = np.zeros_like(feature)
    # Copy only the specified filter
    single_filter[:, filter_index, :, :] = feature[:, filter_index, :, :]
    return single_filter


def get_feature_statistics(features: np.ndarray) -> dict:
    """
    Compute statistics of features for analysis.
    
    Args:
        features: Feature array
    
    Returns:
        Dictionary containing mean, std, min, max, etc.
    """
    return {
        'mean': np.mean(features),
        'std': np.std(features),
        'min': np.min(features),
        'max': np.max(features),
        'shape': features.shape,
        'norm': np.linalg.norm(features),
    }