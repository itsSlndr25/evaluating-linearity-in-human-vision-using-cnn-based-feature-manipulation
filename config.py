"""
Configuration file for VGG-based stimulus generation experiment.

This file contains all hyperparameters and path settings for the experiment.
"""

import torch
from torchvision import models

# ============================================================================
# Model Configuration
# ============================================================================
VGG_MODEL = models.vgg16_bn
VGG_WEIGHTS = models.VGG16_BN_Weights.IMAGENET1K_V1

# Target layers for feature extraction
LAYER_CONFIGS = {
    'conv1_1': {
        'layer_index': 0,      # Index in vgg.features
        'layer_name': 'features.0',
        'n_filters': 64,       # Number of filters in this layer
        'feature_map_size': 224,  # Spatial dimension of feature map
        'n_pcs': 5,            # Number of principal components to use
    },
    'conv2_1': {
        'layer_index': 7,
        'layer_name': 'features.7',
        'n_filters': 128,
        'feature_map_size': 112,
        'n_pcs': 10,
    }
}

# Default layer to use
DEFAULT_LAYER = 'conv2_1'

# ============================================================================
# PCA Configuration
# ============================================================================
# Number of components for PCA (can be overridden by layer config)
N_COMPONENTS = 128  # Maximum number of PCs to compute

# Batch size for incremental PCA fitting（用於節省記憶體）
IPCA_BATCH_SIZE = 32

# ============================================================================
# Image Synthesis Configuration
# ============================================================================
# Number of optimization steps for image reconstruction
N_RECONSTRUCTION_STEPS = 5000

# Early stopping level (0 for early layers, 0.5 for later layers)
EARLY_LAYERS_LEVEL = 0

# Control random seed for reproducibility
RANDOM_SEED = 100

# Image size (VGG input size)
IMG_SIZE = 224

# ============================================================================
# Modification Factors: change the factors if needed
# ============================================================================
# Exp1 - Homogeneity test: multiply single PC by these factors
MAG_FACTORS_HOMOGENEITY = [
    0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0,
    -0.5, -0.8, -1.0, -1.2, -1.5, -2.0
]

# Exp2 - Additivity test: combine two PCs with these factors
MAG_FACTORS_ADDITIVITY = [
    (1.0, 1.0),   # PC1 + PC2
    (1.5, 1.5),   # 1.5*PC1 + 1.5*PC2
    (2.0, 2.0),   # 2.0*PC1 + 2.0*PC2
]

# ============================================================================
# Path Configuration 
# ============================================================================
# Source images folder (e.g., images from NSD dataset)
SOURCE_IMAGES_DIR = "./source_images/"

# Output folder for generated stimuli
OUTPUT_BASE_DIR = "./stimuli_images/"

# Folder to save PCA models (for reuse)
PCA_MODELS_DIR = "./pca_models/"

# ============================================================================
# Device Configuration 
# ============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# Experiment Settings
# ============================================================================
# Which PCs to modify for homogeneity test
PCS_TO_TEST_HOMOGENEITY = {
    'conv1_1': [0, 1, 2, 3, 4],  # PC1-PC5
    'conv2_1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # PC1-PC10
}

# Which PC pairs to test for additivity
PCS_TO_TEST_ADDITIVITY = {
    'conv1_1': [(0, 1), (0, 2), (1, 2)],  # (PC1,PC2), (PC1,PC3), (PC2,PC3)
    'conv2_1': [(0, 1), (2, 3), (4, 5)],
}

# ============================================================================
# ImageNet Normalization Constants
# ============================================================================
IMGNET_MEAN = (0.485, 0.456, 0.406)
IMGNET_STD = (0.229, 0.224, 0.225)