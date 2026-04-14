"""
從修改後的特徵重建圖像
處理優化過程，生成符合目標特徵的圖像

Image synthesis utilities for reconstructing images from modified features.

This module handles the optimization process to generate images that match
target features.
"""

import os
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm


def synthesize_image_from_feature(
    vgg_rec,
    target_feature: np.ndarray,
    layer_name: str,
    source_image: Optional[np.ndarray] = None,
    n_steps: int = 5000,
    early_layers_level: float = 0.0,
    seed: int = 100,
    verbose: bool = True
) -> np.ndarray:
    """
    Synthesize an image that produces the target feature.
    
    This uses gradient descent to optimize the image pixels to match
    the target feature response.
    使用梯度下降優化圖像像素來match目標特徵(ground truth)
    
    Args:
        vgg_rec: VGGRec object for feature extraction and reconstruction
        target_feature: Target feature to match
                       Shape: (1, n_filters, height, width)
        layer_name: Name of the VGG layer
        source_image: Initial image to start optimization from
                     If None, starts from random noise
        n_steps: Number of optimization steps
        early_layers_level: Regularization level (0 for early layers, 0.5 for later)
        seed: Random seed for initialization
        verbose: Whether to print progress
    
    Returns:
        Synthesized image as numpy array (224, 224, 3)
    """
    # Prepare target features dict
    target_feats = {layer_name: target_feature}
    
    # Reconstruct image
    img_reconstructed = vgg_rec.reconstruct_img(
        target_feats,
        img_0=source_image,
        n_steps=n_steps,
        early_layers_level=early_layers_level,
        seed=seed
    )
    
    return img_reconstructed


def generate_stimuli_batch(
    vgg_rec,
    source_image: np.ndarray,
    original_feature: np.ndarray,
    modified_features: Dict[str, np.ndarray],
    layer_name: str,
    output_dir: str,
    base_filename: str,
    n_steps: int = 5000,
    early_layers_level: float = 0.0,
    seed: int = 100
) -> Dict[str, str]:
    """
    Generate a batch of stimulus images with different feature modifications.
    
    Args:
        vgg_rec: VGGRec object
        source_image: Original source image
        original_feature: Original VGG feature before modification
        modified_features: Dictionary mapping condition names to modified features
                          e.g., {'PC1_x2.0': feature1, 'PC2_x0.5': feature2}
        layer_name: VGG layer name
        output_dir: Directory to save generated images
        base_filename: Base name for output files
        n_steps: Optimization steps
        early_layers_level: Regularization level
        seed: Random seed
    
    Returns:
        Dictionary mapping condition names to output file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    output_paths = {}
    
    for condition_name, modified_feature in tqdm(
        modified_features.items(),
        desc=f"Generating stimuli for {base_filename}"
    ):
        # Synthesize image
        img_synthesized = synthesize_image_from_feature(
            vgg_rec=vgg_rec,
            target_feature=modified_feature,
            layer_name=layer_name,
            source_image=source_image,
            n_steps=n_steps,
            early_layers_level=early_layers_level,
            seed=seed,
            verbose=False
        )
        
        # Create output filename
        output_filename = f"{base_filename}_{condition_name}.png"
        output_path = os.path.join(output_dir, layer_name, output_filename)
        
        # Save image
        from vggimg.vgg_img_1v1 import save_img
        save_img(output_path, img_synthesized)
        
        output_paths[condition_name] = output_path
    
    return output_paths


def create_homogeneity_stimuli(
    source_image: np.ndarray,
    original_feature: np.ndarray,
    pca_features: np.ndarray,
    pca_model,
    means: np.ndarray,
    stds: np.ndarray,
    pc_index: int,
    magnitude_factors: list,
    original_shape: tuple
) -> Dict[str, np.ndarray]:
    """
    Create stimuli for homogeneity exp.: L(mx) = mL(x)
    
    Tests whether multiplying a PC by different factors produces proportional
    perceptual changes.
    
    Args:
        source_image: Original image
        original_feature: Original VGG feature
        pca_features: Features in PCA space
        pca_model: Trained PCA model
        means: Batch norm means
        stds: Batch norm stds
        pc_index: Which PC to modify
        magnitude_factors: List of multiplication factors to test
        original_shape: Shape to denormalize back to
    
    Returns:
        Dictionary mapping condition names to modified features
    """
    from .feature_processing import (
        modify_feature_pc,
        denormalize_features
    )
    
    modified_features = {}
    
    for mag in magnitude_factors:
        # Modify PC
        modified_pca = modify_feature_pc(pca_features, pc_index, mag)
        
        # Inverse PCA transform
        modified_normalized = pca_model.inverse_transform(modified_pca)
        
        # Denormalize
        modified_feature = denormalize_features(
            modified_normalized,
            means,
            stds,
            original_shape
        )
        
        # Create condition name
        condition_name = f"PC{pc_index+1}_x{mag}"
        modified_features[condition_name] = modified_feature
    
    return modified_features


def create_additivity_stimuli(
    source_image: np.ndarray,
    original_feature: np.ndarray,
    pca_features: np.ndarray,
    pca_model,
    means: np.ndarray,
    stds: np.ndarray,
    pc_indices: tuple,
    magnitude_factors: tuple,
    original_shape: tuple
) -> Dict[str, np.ndarray]:
    """
    Create stimuli for additivity exp.: L(x+t) = L(x) + L(t)
    
    Tests whether combining two PC modifications produces additive perceptual effects.
    
    Args:
        source_image: Original image
        original_feature: Original VGG feature
        pca_features: Features in PCA space
        pca_model: Trained PCA model
        means: Batch norm means
        stds: Batch norm stds
        pc_indices: Tuple of two PC indices to modify
        magnitude_factors: Tuple of two magnitude factors
        original_shape: Shape to denormalize back to
    
    Returns:
        Dictionary with three conditions:
        - Individual PC modifications
        - Combined modification
    """
    from .feature_processing import (
        modify_feature_pc,
        modify_feature_pcs_additive,
        denormalize_features
    )
    
    modified_features = {}
    pc1_idx, pc2_idx = pc_indices
    mag1, mag2 = magnitude_factors
    
    # Individual modifications
    # PC1 only
    modified_pca_pc1 = modify_feature_pc(pca_features.copy(), pc1_idx, mag1)
    modified_normalized_pc1 = pca_model.inverse_transform(modified_pca_pc1)
    modified_feature_pc1 = denormalize_features(
        modified_normalized_pc1, means, stds, original_shape
    )
    modified_features[f"PC{pc1_idx+1}_x{mag1}"] = modified_feature_pc1
    
    # PC2 only
    modified_pca_pc2 = modify_feature_pc(pca_features.copy(), pc2_idx, mag2)
    modified_normalized_pc2 = pca_model.inverse_transform(modified_pca_pc2)
    modified_feature_pc2 = denormalize_features(
        modified_normalized_pc2, means, stds, original_shape
    )
    modified_features[f"PC{pc2_idx+1}_x{mag2}"] = modified_feature_pc2
    
    # Combined modification
    modified_pca_combined = modify_feature_pcs_additive(
        pca_features.copy(), pc_indices, magnitude_factors
    )
    modified_normalized_combined = pca_model.inverse_transform(modified_pca_combined)
    modified_feature_combined = denormalize_features(
        modified_normalized_combined, means, stds, original_shape
    )
    modified_features[f"PC{pc1_idx+1}_x{mag1}+PC{pc2_idx+1}_x{mag2}"] = modified_feature_combined
    
    return modified_features