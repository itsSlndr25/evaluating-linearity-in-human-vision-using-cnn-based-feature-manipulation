"""
主程式：使用 VGG 特徵生成心理物理實驗用刺激圖像
Main script for generating psychophysical experiment stimuli using VGG features and PCA.

This script implements the complete workflow:
1. Load source images from NSD dataset
2. Extract VGG features from target layer
3. Normalize features using batch statistics
4. Train PCA on normalized features
5. Modify PCs for homogeneity/additivity testing
6. Inverse transform and denormalize
7. Synthesize images from modified features

Usage:
    python main_generate_stimuli.py --layer conv2_1 --test homogeneity
"""

import os
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm

# Import configuration
import config

# Import VGG reconstruction module
from vggimg.vgg_img_1v1 import VGGRec, load_img

# Import utility functions
from utils import (
    normalize_features,
    denormalize_features,
    FeaturePCA,
    train_pca_from_images,
    create_homogeneity_stimuli,
    create_additivity_stimuli,
    generate_stimuli_batch,
)


def get_batch_norm_statistics(layer_config: dict) -> tuple:
    """
    Get batch normalization statistics (mean, std) from VGG model.
    
    Args:
        layer_config: Configuration dict for the target layer
    
    Returns:
        Tuple of (means, stds) as numpy arrays
    """
    # Load VGG model
    vgg = config.VGG_MODEL(weights=config.VGG_WEIGHTS).eval()
    
    # Get batch norm layer (always layer_index + 1)
    bn_layer_index = layer_config['layer_index'] + 1
    bn_layer = vgg.features[bn_layer_index]
    
    # Extract running mean and std
    means = bn_layer.running_mean.detach().cpu().numpy()
    var = bn_layer.running_var.detach().cpu().numpy()
    stds = np.sqrt(var + bn_layer.eps)
    
    # Reshape for broadcasting: (n_filters,) -> (1, n_filters, 1, 1)
    means = means.reshape((1, layer_config['n_filters'], 1, 1))
    stds = stds.reshape((1, layer_config['n_filters'], 1, 1))
    
    return means, stds


def load_source_images(source_dir: str) -> list:
    """
    Load all source image paths from directory.
    
    Args:
        source_dir: Directory containing source images
    
    Returns:
        List of image file paths
    """
    # Support common image formats
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(glob(os.path.join(source_dir, ext)))
    
    image_paths.sort()
    print(f"Found {len(image_paths)} source images")
    
    return image_paths


def main():
    """
    Main function to generate stimuli.
    """
    # Parse arguments 讓程式可以 從 terminal 接收參數
    parser = argparse.ArgumentParser(
        description='Generate psychophysical stimuli from VGG features'
    )
    parser.add_argument(
        '--layer',
        type=str,
        default=config.DEFAULT_LAYER,
        choices=list(config.LAYER_CONFIGS.keys()),
        help='Target VGG layer'
    )
    parser.add_argument(
        '--test',
        type=str,
        default='homogeneity',
        choices=['homogeneity', 'additivity', 'both'],
        help='Type of linearity test'
    )
    parser.add_argument(
        '--source_dir',
        type=str,
        default=config.SOURCE_IMAGES_DIR,
        help='Directory containing source images'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: auto-generated)'
    )
    parser.add_argument(
        '--retrain_pca',
        action='store_true',
        help='Retrain PCA even if saved model exists'
    )
    
    args = parser.parse_args()
    
    # Get layer configuration
    layer_config = config.LAYER_CONFIGS[args.layer]
    layer_name = layer_config['layer_name']
    n_filters = layer_config['n_filters']
    map_size = layer_config['feature_map_size']
    n_pcs = layer_config['n_pcs']
    
    print("="*80)
    print(f"Generating stimuli for {args.layer}")
    print(f"Layer: {layer_name}, Filters: {n_filters}, Map size: {map_size}")
    print(f"Test type: {args.test}")
    print("="*80)
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            config.OUTPUT_BASE_DIR,
            f"{args.layer}_{args.test}"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Initialize VGG feature extractor
    print("\n[1/7] Initializing VGG model...")
    vgg_rec = VGGRec(layer=layer_name, device=config.DEVICE)
    
    # Get batch normalization statistics
    print("\n[2/7] Extracting batch normalization statistics...")
    means, stds = get_batch_norm_statistics(layer_config)
    print(f"Means shape: {means.shape}, Stds shape: {stds.shape}")
    
    # Load source images
    print("\n[3/7] Loading source images...")
    image_paths = load_source_images(args.source_dir)
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {args.source_dir}")
    
    # Train or load PCA
    print("\n[4/7] Setting up PCA model...")
    pca_model_path = os.path.join(
        config.PCA_MODELS_DIR,
        f"pca_{args.layer}_ncomp{n_filters}.pkl"
    )
    
    pca_model = FeaturePCA(n_components=n_filters)
    
    if os.path.exists(pca_model_path) and not args.retrain_pca:
        print(f"Loading existing PCA model from {pca_model_path}")
        pca_model.load(pca_model_path)
    else:
        print(f"Training new PCA model...")
        pca_model = train_pca_from_images(
            image_paths=image_paths,
            vgg_rec=vgg_rec,
            layer_name=layer_name,
            means=means,
            stds=stds,
            n_components=n_filters,
            batch_size=config.IPCA_BATCH_SIZE
        )
        # Save PCA model
        pca_model.save(pca_model_path)
    
    # Print PCA statistics
    explained_var = pca_model.get_explained_variance_ratio()
    print(f"Explained variance (first {n_pcs} PCs): "
          f"{np.sum(explained_var[:n_pcs]):.4f}")
    
    # Generate stimuli for each source image
    print(f"\n[5/7] Generating stimuli for {len(image_paths)} images...")
    
    for img_idx, img_path in enumerate(tqdm(image_paths, desc="Processing images")):
        # Load image
        img = load_img(img_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Extract original features
        features = vgg_rec.get_features(img)
        original_feature = features[layer_name]
        original_shape = original_feature.shape  # (1, n_filters, H, W)
        
        # Normalize features
        normalized_feature = normalize_features(original_feature, means, stds)
        
        # Transform to PCA space
        pca_features = pca_model.transform(normalized_feature)
        
        # Generate homogeneity stimuli
        if args.test in ['homogeneity', 'both']:
            print(f"\n  Generating homogeneity stimuli for {base_name}...")
            
            pcs_to_test = config.PCS_TO_TEST_HOMOGENEITY[args.layer]
            
            for pc_idx in pcs_to_test:
                # Create modified features
                modified_features_dict = create_homogeneity_stimuli(
                    source_image=img,
                    original_feature=original_feature,
                    pca_features=pca_features.copy(),
                    pca_model=pca_model,
                    means=means,
                    stds=stds,
                    pc_index=pc_idx,
                    magnitude_factors=config.MAG_FACTORS_HOMOGENEITY,
                    original_shape=original_shape
                )
                
                # Generate images
                output_subdir = os.path.join(args.output_dir, "exp1_stimuli")
                generate_stimuli_batch(
                    vgg_rec=vgg_rec,
                    source_image=img,
                    original_feature=original_feature,
                    modified_features=modified_features_dict,
                    layer_name=layer_name,
                    output_dir=output_subdir,
                    base_filename=base_name,
                    n_steps=config.N_RECONSTRUCTION_STEPS,
                    early_layers_level=config.EARLY_LAYERS_LEVEL,
                    seed=config.RANDOM_SEED
                )
        
        # Generate additivity stimuli
        if args.test in ['additivity', 'both']:
            print(f"\n  Generating additivity stimuli for {base_name}...")
            
            pc_pairs_to_test = config.PCS_TO_TEST_ADDITIVITY[args.layer]
            
            for pair_idx, pc_pair in enumerate(pc_pairs_to_test):
                for mag_pair in config.MAG_FACTORS_ADDITIVITY:
                    # Create modified features
                    modified_features_dict = create_additivity_stimuli(
                        source_image=img,
                        original_feature=original_feature,
                        pca_features=pca_features.copy(),
                        pca_model=pca_model,
                        means=means,
                        stds=stds,
                        pc_indices=pc_pair,
                        magnitude_factors=mag_pair,
                        original_shape=original_shape
                    )
                    
                    # Generate images
                    output_subdir = os.path.join(
                        args.output_dir,
                        "exp2_stimuli"
                    )
                    generate_stimuli_batch(
                        vgg_rec=vgg_rec,
                        source_image=img,
                        original_feature=original_feature,
                        modified_features=modified_features_dict,
                        layer_name=layer_name,
                        output_dir=output_subdir,
                        base_filename=f"{base_name}_mag{mag_pair[0]}_{mag_pair[1]}",
                        n_steps=config.N_RECONSTRUCTION_STEPS,
                        early_layers_level=config.EARLY_LAYERS_LEVEL,
                        seed=config.RANDOM_SEED
                    )
    
    print("\n" + "="*80)
    print("✓ Stimulus generation complete!")
    print(f"Output saved to: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()