"""
Additivity Test Analysis for TvC Functions.

This script analyzes threshold vs contrast (TvC) functions to test
the additivity property of linearity: L(x+t) = L(x) + L(t)


Usage:
    python analyze_additivity.py --data_file cih375.dat --ref_file cih375ref.con
"""

import os
import numpy as np
import argparse
from analysis_utils import (
    read_selected_columns,
    add_reference_column,
    organize_layer_data_additivity,
    plot_tvc_semilog
)


def main(data_filepath: str, ref_filepath: str, output_dir: str = './plot_add'):
    """
    Main analysis function for additivity testing.
    
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
    print(f"Additivity Analysis: {exp_id}")
    print("="*80)
    
    # Step 1: Load and preprocess data
    print("\n[1/4] Loading data...")
    selected_data = read_selected_columns(data_filepath)
    data = add_reference_column(selected_data, ref_filepath)
    
    # Sort by image
    data = data[data[:, 0].argsort()]
    print(f"Loaded {len(data)} trials")
    
    # Step 2: Organize data for each image and layer
    print("\n[2/4] Organizing data by image and layer...")
    
    thresholds_dataset = {}
    intensity_values_list = []
    
    for img_idx in range(1, 5):
        print(f"  Processing image {img_idx}...")
        
        results = organize_layer_data_additivity(data, img_idx)
        
        for layer_idx in range(1, 3):
            layer_key = f'layer{layer_idx}'
            
            if layer_key not in thresholds_dataset:
                thresholds_dataset[layer_key] = {}
            
            # Store thresholds and SEM
            thresholds_dataset[layer_key][f'thresholds_img{img_idx}'] = \
                results[layer_key]['thresholds']
            thresholds_dataset[layer_key][f'sem_img{img_idx}'] = \
                results[layer_key]['sem']
            
            # Store reference levels (only once per layer)
            if img_idx == 1:
                intensity_values_list.append(results[layer_key]['ref_levels'])
    
    # Stack reference levels for both layers
    intensity_values_stacked = np.vstack(intensity_values_list[:2])
    
    print(f"  Organized data for 2 layers, 4 images")
    print(f"  Layer 1: {thresholds_dataset['layer1']['thresholds_img1'].shape}")
    print(f"  Layer 2: {thresholds_dataset['layer2']['thresholds_img1'].shape}")
    
    # Step 3: Generate plots for each image and layer
    print("\n[3/4] Generating individual image plots...")
    
    for layer_idx in range(1, 3):
        layer_key = f'layer{layer_idx}'
        layer_name = f'conv{layer_idx}_1'
        ref_levels = intensity_values_stacked[layer_idx - 1, :]
        
        print(f"\n  Processing {layer_name}...")
        
        for img_idx in range(1, 5):
            print(f"    Image {img_idx}...")
            
            thresholds = thresholds_dataset[layer_key][f'thresholds_img{img_idx}']
            sem = thresholds_dataset[layer_key][f'sem_img{img_idx}']
            
            # Plot TvC curves
            plot_tvc_semilog(
                thresholds=thresholds,
                sem=sem,
                ref_levels=ref_levels,
                layer_name=layer_name,
                img_idx=img_idx,
                output_path=os.path.join(
                    output_subdir,
                    f'TvC_{exp_id}_{layer_name}_img{img_idx}.png'
                ),
                condition_labels=['PC1', 'PC2', 'Combined'],
                plot_title='',
                show_legend=False
            )
    
    # Step 4: Calculate and plot mean across images
    print("\n[4/4] Calculating means across images...")
    
    for layer_idx in range(1, 3):
        layer_key = f'layer{layer_idx}'
        layer_name = f'conv{layer_idx}_1'
        ref_levels = intensity_values_stacked[layer_idx - 1, :]
        
        print(f"\n  Processing {layer_name} mean...")
        
        # Collect data from all images
        data_all = []
        sem_all = []
        
        for img_idx in range(1, 5):
            thresholds = thresholds_dataset[layer_key][f'thresholds_img{img_idx}']
            sem = thresholds_dataset[layer_key][f'sem_img{img_idx}']
            
            data_all.append(thresholds[..., np.newaxis])
            sem_all.append(sem[..., np.newaxis])
        
        # Stack: (n_conditions, n_levels, n_images)
        data_all = np.concatenate(data_all, axis=2)
        sem_all = np.concatenate(sem_all, axis=2)
        
        # Calculate mean and SEM across images
        mean_thresholds = np.mean(data_all, axis=2)
        # For SEM across images, use pooled SEM
        pooled_sem = np.sqrt(np.mean(sem_all**2, axis=2))
        
        print(f"    Mean shape: {mean_thresholds.shape}")
        print(f"    Values: {mean_thresholds}")
        
        # Note: Uncomment if needed for analysis
        plot_tvc_semilog(
            thresholds=mean_thresholds,
            sem=pooled_sem,
            ref_levels=ref_levels,
            layer_name=layer_name,
            img_idx=0,
            output_path=os.path.join(
                output_subdir,
                f'TvC_{exp_id}_{layer_name}_CombinedImg.png'
            ),
            condition_labels=['PC1', 'PC2', 'Combined'],
            plot_title='',
            show_legend=False
        )
    
    print("\n" + "="*80)
    print("✓ Additivity analysis complete!")
    print(f"  Output saved to: {output_subdir}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze TvC functions for additivity testing'
    )
    parser.add_argument(
        '--data_file',
        type=str,
        default='.exp_code/cih575.dat',
        help='Path to psychophysical data file (.dat)'
    )
    parser.add_argument(
        '--ref_file',
        type=str,
        default='.exp_code/cih575ref.con',
        help='Path to reference condition file (.con)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./plot_add',
        help='Output directory for plots'
    )
    
    args = parser.parse_args()
    
    main(
        data_filepath=args.data_file,
        ref_filepath=args.ref_file,
        output_dir=args.output_dir
    )