"""
TvC(Threshold vs Contrast) 資料分析
Utility functions for psychophysical TvC(Threshold vs Contrast) data analysis.

This module provides functions for:
- Data loading and preprocessing
- Statistical calculations  
- Foley 1994 model fitting
- TvC curve plotting
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Tuple, Dict, Optional, List


# ============================================================================
# Data Loading and Preprocessing - 資料載入與前處理
# ============================================================================

def read_selected_columns(filepath: str) -> np.ndarray:
    """
    Read psychophysical data file and select relevant columns.
    讀取實驗資料檔案並選擇相關欄位
    
    Args:
        filepath: Path to .dat file
                  .dat 檔案路徑
    
    Returns:
        Selected columns: [pc_index, layer_index, threshold, response_accuracy]
    """
    data = np.loadtxt(filepath)
    # Select columns: pc_index(1), layer_index(3), threshold(-2), accuracy(-1)
    selected_data = data[:, [1, 3, -2, -1]]
    return selected_data


def add_reference_column(
    selected_data: np.ndarray,
    ref_filepath: str
) -> np.ndarray:
    """
    Add reference condition(參考圖像) column to data.
    
    Args:
        selected_data: Data from read_selected_columns
        ref_filepath: Path to reference condition file (.con or .txt)
    
    Returns:
        Data array with columns: [img, layer, pc, threshold, refcon]
    """
    # Load reference conditions
    refcon = np.loadtxt(ref_filepath)
    ref_values = refcon[:, 3].reshape(-1, 1)
    
    # Add reference column
    data_with_ref = np.hstack((selected_data, ref_values))
    
    # Reorder columns: [img, layer, pc, threshold, refcon]
    # Original order: [pc, layer, threshold, response, refcon]
    new_order = [3, 2, 0, 1, 4]
    data_reordered = data_with_ref[:, new_order]
    
    return data_reordered


def calculate_mean_sem_rawdata(arr: np.ndarray) -> np.ndarray:
    """
    Calculate mean and SEM for each unique condition pair.
    
    Args:
        arr: Array with shape (n_trials, 3)
             Columns: [condition1 (PC), value (threshold), condition2 (refcon)]
    
    Returns:
        Array with shape (n_unique_conditions, 4)
        Columns: [condition1, condition2, mean, sem]
    """
    # Identify unique condition pairs
    condition_pairs, inverse_indices = np.unique(
        arr[:, [0, 2]], axis=0, return_inverse=True
    )
    
    means = []
    sems = []
    
    # Calculate statistics for each condition pair
    for i in range(len(condition_pairs)):
        mask = (inverse_indices == i)
        data_values = arr[mask, 1]
        
        mean = np.mean(data_values)
        sem = np.std(data_values, ddof=1) / np.sqrt(len(data_values))
        
        means.append(mean)
        sems.append(sem)
    
    # Combine results
    means = np.array(means).reshape(-1, 1)
    sems = np.array(sems).reshape(-1, 1)
    output = np.hstack((condition_pairs, means, sems))
    
    return output


# ============================================================================
# Foley 1994 Model - Foley 1994 模型
# ============================================================================

def foley94_model(pedestal_contrasts: np.ndarray, k: float, p: float, 
                  q: float, Z: float) -> np.ndarray:
    """
    Foley 1994 contrast discrimination model.
    
    Simplified response model:
    R(E) = k * E^p / (E^q + Z)
    
    Args:
        pedestal_contrasts: Pedestal contrast levels (reference conditions)
        k: Scaling factor
        p: Numerator exponent (分子指數)
        q: Denominator exponent (分母指數)
        Z: Internal noise parameter
    
    Returns:
        Predicted threshold values
    """
    E = np.maximum(pedestal_contrasts, 1e-10)  # Prevent numerical issues
    denominator = np.maximum(E**q + Z, 1e-10)
    return k * (E**p) / denominator


# ============================================================================
# Data Organization
# ============================================================================

def organize_layer_data_homogeneity(
    data: np.ndarray,
    image_idx: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Organize data for homogeneity experiment (single PC modification).
    
    Args:
        data: Full dataset sorted by image and layer
        image_idx: Image index (1-4)
    
    Returns:
        Tuple of (thresholds_dict, sem_dict) for layer1 and layer2
    """
    # Extract data for this image
    img_data = data[data[:, 0] == image_idx]
    img_data = img_data[img_data[:, 1].argsort()]  # Sort by layer
    
    thresholds_dict = {}
    sem_dict = {}
    
    # Process layer 1 (5 PCs)
    layer1 = img_data[img_data[:, 1] == 1]
    layer1 = layer1[:, -3:]  # [pc, threshold, refcon]
    
    layer1_processed = calculate_mean_sem_rawdata(layer1)
    categories = np.unique(layer1_processed[:, 0])
    intensity_values = np.sort(np.unique(layer1_processed[:, 1]))
    
    thresholds_data = []
    sem_data = []
    
    for category in categories:
        subset = layer1_processed[layer1_processed[:, 0] == category]
        subset = subset[subset[:, 1].argsort()]
        thresholds_data.append(subset[:, 2])
        sem_data.append(subset[:, 3])
    
    thresholds_dict['layer1'] = {
        'thresholds': np.array(thresholds_data),
        'sem': np.array(sem_data),
        'ref_levels': intensity_values
    }
    
    # Process layer 2 (10 PCs)
    layer2 = img_data[img_data[:, 1] == 2]
    layer2 = layer2[:, -3:]
    
    image_dict = {}
    for pc_id in range(1, 11):
        mask = layer2[:, 0] == pc_id
        levels = layer2[mask, 2]
        thresholds = layer2[mask, 1]
        
        sort_idx = np.argsort(levels)
        levels_sorted = levels[sort_idx]
        thresholds_sorted = thresholds[sort_idx]
        
        image_dict[f'PC{pc_id}'] = {
            'levels': levels_sorted,
            'thresholds': thresholds_sorted
        }
    
    thresholds_dict['layer2'] = image_dict
    
    return thresholds_dict


def organize_layer_data_additivity(
    data: np.ndarray,
    image_idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Organize data for additivity experiment (combined PC modification).
    
    Args:
        data: Full dataset sorted by image and layer
        image_idx: Image index (1-4)
    
    Returns:
        Tuple of (thresholds, sem, ref_levels) for the specified image
    """
    img_data = data[data[:, 0] == image_idx]
    img_data = img_data[img_data[:, 1].argsort()]
    
    results = {}
    
    for layer_idx in range(1, 3):  # layer 1 and 2
        layer = img_data[img_data[:, 1] == layer_idx]
        layer = layer[:, -3:]  # [pc, threshold, refcon]
        
        layer_processed = calculate_mean_sem_rawdata(layer)
        
        categories = np.unique(layer_processed[:, 0])
        intensity_values = np.sort(np.unique(layer_processed[:, 1]))
        
        thresholds_data = []
        sem_data = []
        
        for category in categories:
            subset = layer_processed[layer_processed[:, 0] == category]
            subset = subset[subset[:, 1].argsort()]
            thresholds_data.append(subset[:, 2])
            sem_data.append(subset[:, 3])
        
        results[f'layer{layer_idx}'] = {
            'thresholds': np.array(thresholds_data),
            'sem': np.array(sem_data),
            'ref_levels': intensity_values
        }
    
    return results


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_tvc_semilog(
    thresholds: np.ndarray,
    sem: np.ndarray,
    ref_levels: np.ndarray,
    layer_name: str,
    img_idx: int,
    output_path: str,
    condition_labels: Optional[List[str]] = None,
    plot_title: str = '',
    show_legend: bool = False
):
    """
    Plot TvC functions in semi-log scale with Foley 1994 model fitting.
    
    Args:
        thresholds: Threshold data, shape (n_conditions, n_levels)
        sem: Standard error of mean, same shape as thresholds
        ref_levels: Reference contrast levels (x-axis)
        layer_name: Layer name for labeling (e.g., 'conv1_1')
        img_idx: Image index
        output_path: Path to save the figure
        condition_labels: Labels for each condition (default: PC1, PC2, ...)
        plot_title: Plot title (empty for publication)
        show_legend: Whether to show legend
    """
    num_conditions = thresholds.shape[0]
    
    # Default labels
    if condition_labels is None:
        condition_labels = [f'PC{i+1}' for i in range(num_conditions)]
    
    # Colors and markers
    colors = plt.cm.cool(np.linspace(0, 1, num_conditions))
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', '+', 'x']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each condition
    for i in range(num_conditions):
        condition_mean = thresholds[i, :]
        condition_sem = sem[i, :]
        
        # Plot data points with error bars
        ax.errorbar(
            ref_levels, condition_mean, yerr=condition_sem,
            marker=markers[i % len(markers)],
            color=colors[i],
            linestyle='',
            markersize=8,
            capsize=5,
            capthick=2,
            linewidth=4,
            label=f'{condition_labels[i]} Data'
        )
        
        # Fit Foley model
        try:
            popt, _ = curve_fit(
                foley94_model, ref_levels, condition_mean,
                p0=[1.0, 2.0, 2.0, 0.01],
                bounds=([0.01, 0.1, 0.1, 0.0001], [100, 10, 10, 1]),
                maxfev=10000
            )
            
            # Plot fitted curve
            x_smooth = np.logspace(
                np.log10(min(ref_levels) * 0.9),
                np.log10(max(ref_levels) * 1.1),
                100
            )
            y_fitted = foley94_model(x_smooth, *popt)
            
            ax.semilogx(
                x_smooth, y_fitted,
                color=colors[i],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=4,
                label=f'{condition_labels[i]} Fit'
            )
            
        except Exception as e:
            print(f"Model fitting failed for {condition_labels[i]}: {e}")
    
    # Format plot
    ax.set_xlabel(plot_title if plot_title else '', fontsize=12)
    ax.set_ylabel('', fontsize=12)
    ax.set_title('', fontsize=14)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    # Format x-axis
    ax.set_xticks(ref_levels)
    ax.get_xaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"{x:.3f}")
    )
    ax.get_xaxis().set_minor_locator(plt.NullLocator())
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # Format y-axis
    ax.set_yticks([])
    ax.tick_params(axis='both', labelsize=24)
    
    if show_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_tvc_weber_fraction(
    data_dict: Dict[str, Dict],
    layer_name: str,
    output_path: str,
    plot_title: str = '',
    show_legend: bool = False
):
    """
    Plot Weber fractions (threshold/level) for layer 2.
    
    Args:
        data_dict: Dictionary with PC data
        layer_name: Layer name
        output_path: Output file path
        plot_title: Plot title
        show_legend: Whether to show legend
    """
    import pandas as pd
    
    pc_names = []
    weber_mean = []
    weber_std = []
    level_values = []
    
    for pc_name, pc_data in data_dict.items():
        levels = pc_data['levels']
        thresholds = pc_data['thresholds']
        
        # Calculate Weber fraction
        weber = thresholds / np.array(levels)[:, np.newaxis]
        weber_m = np.mean(weber, axis=1)
        weber_s = np.std(weber, axis=1)
        
        # Store data
        pc_names.extend([pc_name] * len(levels))
        weber_mean.extend(weber_m)
        weber_std.extend(weber_s)
        level_values.extend(levels)
    
    # Create DataFrame
    df = pd.DataFrame({
        'PC': pc_names,
        'Level': level_values,
        'Weber Fraction': weber_mean,
        'Weber Std': weber_std
    })
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    unique_pcs = df['PC'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_pcs)))
    
    for i, pc in enumerate(unique_pcs):
        pc_data = df[df['PC'] == pc]
        ax.errorbar(
            pc_data['Level'], pc_data['Weber Fraction'],
            yerr=pc_data['Weber Std'],
            marker='o', label=pc, color=colors[i],
            capsize=3, capthick=1, linewidth=2
        )
    
    ax.set_xscale('log')
    ax.set_title(plot_title if plot_title else '')
    ax.set_xlabel('', fontsize=12)
    ax.set_ylabel('', fontsize=12)
    ax.set_yticks([])
    ax.tick_params(axis='both', labelsize=24)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    if show_legend:
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()