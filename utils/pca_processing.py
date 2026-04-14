"""
特徵處理與轉換 PCA
PCA processing utilities for feature transformation.

This module handles:
- Training IncrementalPCA on large datasets
- Transforming features to PCA space
- Inverse transforming back to feature space
- Saving and loading PCA models
"""

import os
import pickle
import numpy as np
from sklearn.decomposition import IncrementalPCA
from typing import Optional, List
from tqdm import tqdm


class FeaturePCA:
    """
    Wrapper class for PCA operations on VGG features.
    
    This class uses IncrementalPCA to handle large datasets that don't fit in memory.
    使用 IncrementalPCA 處理無法一次性載入記憶體的大型資料集
    """
    
    def __init__(self, n_components: int = 128):
        """
        Initialize PCA model.
        
        Args:
            n_components: Number of principal components to keep
        """
        self.n_components = n_components
        self.pca = IncrementalPCA(n_components=n_components)
        self.is_fitted = False
        
    def fit(
        self,
        features_list: List[np.ndarray],
        batch_size: Optional[int] = None
    ):
        """
        Fit PCA model on a list of feature arrays.
        
        Args:
            features_list: List of feature arrays, each with shape (n_samples, n_features)
            batch_size: Number of samples per batch for incremental fitting
                       If None, process each array in features_list as one batch
        """
        print(f"Training PCA with {self.n_components} components...")
        
        # If batch_size is specified, concatenate all features and split
        if batch_size is not None:
            all_features = np.vstack(features_list)
            n_samples = all_features.shape[0]
            
            for start_idx in tqdm(range(0, n_samples, batch_size), desc="PCA fitting"):
                end_idx = min(start_idx + batch_size, n_samples)
                batch = all_features[start_idx:end_idx]
                self.pca.partial_fit(batch)
        else:
            # Process each feature array as one batch
            for features in tqdm(features_list, desc="PCA fitting"):
                self.pca.partial_fit(features)
        
        self.is_fitted = True
        print(f"PCA fitting complete. Explained variance ratio: "
              f"{np.sum(self.pca.explained_variance_ratio_):.4f}")
        
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features to PCA space.
        
        Args:
            features: Feature array with shape (n_samples, n_features)
        
        Returns:
            Transformed features in PCA space (n_samples, n_components)
        """
        if not self.is_fitted:
            raise RuntimeError("PCA model has not been fitted yet.")
        
        return self.pca.transform(features)
    
    def inverse_transform(self, pca_features: np.ndarray) -> np.ndarray:
        """
        Inverse transform from PCA space back to feature space.
        
        Args:
            pca_features: Features in PCA space (n_samples, n_components)
        
        Returns:
            Reconstructed features in original space (n_samples, n_features)
        """
        if not self.is_fitted:
            raise RuntimeError("PCA model has not been fitted yet.")
        
        return self.pca.inverse_transform(pca_features)
    
    def save(self, filepath: str):
        """
        Save PCA model
        
        Args:
            filepath: Path to save the model (should end with .pkl)
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'pca': self.pca,
                'n_components': self.n_components,
                'is_fitted': self.is_fitted
            }, f)
        
        print(f"PCA model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load PCA model from disk.
        
        Args:
            filepath: Path to the saved model 儲存的模型路徑
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PCA model not found at {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.pca = data['pca']
        self.n_components = data['n_components']
        self.is_fitted = data['is_fitted']
        
        print(f"PCA model loaded from {filepath}")
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        Get the explained variance ratio for each component.
        
        Returns:
            Array of explained variance ratios
        """
        if not self.is_fitted:
            raise RuntimeError("PCA model has not been fitted yet.")
        
        return self.pca.explained_variance_ratio_
    
    def get_principal_components(self) -> np.ndarray:
        """
        Get the principal component vectors.
        
        Returns:
            Principal components with shape (n_components, n_features)
        """
        if not self.is_fitted:
            raise RuntimeError("PCA model has not been fitted yet.")
        
        return self.pca.components_


def train_pca_from_images(
    image_paths: List[str],
    vgg_rec,
    layer_name: str,
    means: np.ndarray,
    stds: np.ndarray,
    n_components: int,
    batch_size: int = 32
) -> FeaturePCA:
    """
    Train PCA model from a list of images.
    
    This function:
    1. Loads images in batches
    2. Extracts VGG features
    3. Normalizes features
    4. Trains IncrementalPCA
    
    Args:
        image_paths: List of paths to source images
        vgg_rec: VGGRec object for feature extraction
        layer_name: Name of VGG layer to extract
        means: Batch norm means for normalization
        stds: Batch norm stds for normalization
        n_components: Number of PCs to compute
        batch_size: Batch size for processing
    
    Returns:
        Trained FeaturePCA object
    """
    from .feature_processing import normalize_features
    from vggimg.vgg_img_1v1 import load_img
    
    pca_model = FeaturePCA(n_components=n_components)
    normalized_features_list = []
    
    print(f"Extracting features from {len(image_paths)} images...")
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i:i+batch_size]
        batch_features = []
        
        for img_path in batch_paths:
            # Load image
            img = load_img(img_path)
            
            # Extract features
            features = vgg_rec.get_features(img)
            feature = features[layer_name]
            
            # Normalize
            normalized = normalize_features(feature, means, stds)
            batch_features.append(normalized)
        
        # Stack batch
        batch_features = np.vstack(batch_features)
        normalized_features_list.append(batch_features)
    
    # Fit PCA
    pca_model.fit(normalized_features_list)
    
    return pca_model