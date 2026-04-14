"""
Utilities package for VGG-based stimulus generation.
"""

from .feature_processing import (
    normalize_features,
    denormalize_features,
    modify_feature_pc,
    modify_feature_pcs_additive,
    get_feature_statistics,
)

from .pca_processing import (
    FeaturePCA,
    train_pca_from_images,
)

from .image_synthesis import (
    synthesize_image_from_feature,
    generate_stimuli_batch,
    create_homogeneity_stimuli,
    create_additivity_stimuli,
)

__all__ = [
    # Feature processing
    'normalize_features',
    'denormalize_features',
    'modify_feature_pc',
    'modify_feature_pcs_additive',
    'get_feature_statistics',
    
    # PCA processing
    'FeaturePCA',
    'train_pca_from_images',
    
    # Image synthesis
    'synthesize_image_from_feature',
    'generate_stimuli_batch',
    'create_homogeneity_stimuli',
    'create_additivity_stimuli',
]