"""
Homogeneity Test Analysis for TvC Functions.

This script analyzes threshold vs contrast (TvC) functions to test
the homogeneity property of linearity: L(mx) = mL(x)

Usage:
    python analyze_homogeneity.py --data_file cih325.dat --ref_file cih325ref.con
"""

import os
import numpy as np
import argparse
from analysis_utils import (
    read_selected_columns,
    add_reference_column,
    organize_layer_data_homogeneity,
    plot_tvc_semilog,
    plot_tvc_weber_fraction
)


def main(data_filepath: str, ref_filepath: str, output_dir: str = './plot_hom'):
    """
    Main analysis function for homogeneity testing.
    
    Args:
        data_filepath: Path to .dat file with psychophysical data
        ref_filepath: Path to reference condition file (.con)
        output_dir: Output directory for plots
    """
    # Extract experiment ID from filename
    exp_id = os.path.basename(data_filepath)[:-4]
    output_subdir = os.path.join(output_dir, exp_id)
    os.makedirs(output_subdir, exist_ok=True)
    
    print("="*80)
    print(f"Homogeneity Analysis: {exp_id}")
    print("="*80)
    
    # Step 1: Load and preprocess data
    print("\n[1/4] Loading data...")
    selected_data = read_selected_columns(data_filepath)
    data = add_reference_column(selected_data, ref_filepath)
    
    # Sort by image
    data = data[data[:, 0].argsort()]
    print(f"Loaded {len(data)} trials")
    
    # Step 2: Organize data by layer for each image
    print("\n[2/4] Organizing data by image and layer...")
    
    # Store data for layer 1 across all images
    layer1_all_images = []
    
    # Store data for layer 2 across all images
    layer2_all_pcs = {}
    for pc_id in range(1, 11):
        layer2_all_pcs[f'PC{pc_id}'] = {
            'levels': None,
            'thresholds': []
        }
    
    # Process each image
    for img_idx in range(1, 5):
        print(f"  Processing image {img_idx}...")
        
        thresholds_dict = organize_layer_data_homogeneity(data, img_idx)
        
        # Collect layer 1 data
        layer1_data = thresholds_dict['layer1']['thresholds']
        layer1_all_images.append(layer1_data)
        
        # Collect layer 2 data
        for pc_id in range(1, 11):
            pc_key = f'PC{pc_id}'
            pc_data = thresholds_dict['layer2'][pc_key]
            
            # Store levels (same across images)
            if layer2_all_pcs[pc_key]['levels'] is None:
                layer2_all_pcs[pc_key]['levels'] = pc_data['levels']
            
            # Collect thresholds
            layer2_all_pcs[pc_key]['thresholds'].append(pc_data['thresholds'])
    
    # Stack layer 1 data: (n_pcs, n_levels, n_images)
    layer1_stacked = np.stack(layer1_all_images, axis=2)
    
    # Stack layer 2 data
    for pc_key in layer2_all_pcs.keys():
        thresholds_list = layer2_all_pcs[pc_key]['thresholds']
        layer2_all_pcs[pc_key]['thresholds'] = np.stack(thresholds_list, axis=1)
    
    print(f"  Layer 1 data shape: {layer1_stacked.shape}")
    print(f"  Layer 2: {len(layer2_all_pcs)} PCs organized")
    
    # Step 3: Calculate mean and SEM across images
    print("\n[3/4] Calculating statistics across images...")
    
    # Layer 1
    layer1_mean = np.mean(layer1_stacked, axis=2)
    layer1_std = np.std(layer1_stacked, axis=2, ddof=1)
    layer1_sem = layer1_std / np.sqrt(layer1_stacked.shape[2])
    
    # Get reference levels from first image data
    img1_data = organize_layer_data_homogeneity(data, 1)
    layer1_ref_levels = img1_data['layer1']['ref_levels']
    
    print(f"  Layer 1: {layer1_mean.shape[0]} PCs, {layer1_mean.shape[1]} levels")
    
    # Step 4: Generate plots
    print("\n[4/4] Generating plots...")
    
    # Plot Layer 1 TvC curves
    print("  Plotting Layer 1 TvC curves...")
    plot_tvc_semilog(
        thresholds=layer1_mean,
        sem=layer1_sem,
        ref_levels=layer1_ref_levels,
        layer_name='conv1_1',
        img_idx=0,  # Combined across images
        output_path=os.path.join(output_subdir, f'{exp_id}_TvC_conv1_1_CombinedImg.png'),
        condition_labels=[f'PC{i+1}' for i in range(layer1_mean.shape[0])],
        plot_title='',
        show_legend=False
    )
    
    # Plot Layer 2 Weber fractions
    print("  Plotting Layer 2 Weber fractions...")
    plot_tvc_weber_fraction(
        data_dict=layer2_all_pcs,
        layer_name='conv2_1',
        output_path=os.path.join(output_subdir, f'TvC_conv2_1_MeanAcrossImages.png'),
        plot_title='',
        show_legend=False
    )
    
    print("\n" + "="*80)
    print("✓ Homogeneity analysis complete!")
    print(f"  Output saved to: {output_subdir}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze TvC functions for homogeneity testing'
    )
    parser.add_argument(
        '--data_file',
        type=str,
        default='.exp_code/cih625.dat',
        help='Path to psychophysical data file (.dat)'
    )
    parser.add_argument(
        '--ref_file',
        type=str,
        default='.exp_code/cih625ref.con',
        help='Path to reference condition file (.con)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./plot_hom',
        help='Output directory for plots'
    )
    
    args = parser.parse_args()
    
    main(
        data_filepath=args.data_file,
        ref_filepath=args.ref_file,
        output_dir=args.output_dir
    )