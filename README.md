# dCNN Feature Linearity & Psychophysics Pipeline

## 📋 Project Overview

This project investigates the correspondence of the linearity of deep Convolutional Neural Networks (dCNN) and the human visual system between each other. Specifically, we test our hypothesis via two fundamental properties of linear systems: homogeneity & additivity

### Key Contributions
1. Built PCA-based feature manipulation framework
2. Implemented gradient-based feature inversion pipeline
3. Integrated computational modeling with 2-AFC psychophysics

### Research Question
Does human perception exhibit linear behavior corresponding to deep CNN models?

### Method Summary
1. Extract VGG16 features from natural scene images
2. Apply PCA to decorrelate feature dimensions
3. Systematically modify principal components
4. Synthesize stimuli images via gradient descent
5. Measure human perception using 2-AFC psychophysics

---

## 🏗️ Architecture

```
project/
├── config.py                    # All hyperparameters and paths
├── main_generate_stimuli.py     # Main script for stimulus generation
├── utils/
│   ├── __init__.py
│   ├── feature_processing.py   # Feature normalization/modification
│   ├── pca_processing.py       # PCA training and transformation
│   └── image_synthesis.py      # Image reconstruction from features
├── vggimg/
│   └── vgg_img_1v1.py          # VGG feature extraction (base module)
└── README.md

- Project 2- CNN-Based Feature Manipulation & Human Vision Experiment.pdf ← summary of the project
- Project 2-dCNN 特徵操弄&人類視覺實驗.pdf ← summary of the project *chinese version
```

---

## 🚀 Quick Start

### Installation

```bash
# Create conda environment
conda create -n vgg_linearity python=3.8
conda activate vgg_linearity

# Install dependencies
pip install torch torchvision numpy scikit-learn Pillow tqdm matplotlib
```

### Generate Stimuli

```bash
# For homogeneity testing on conv2_1
python main_generate_stimuli.py --layer conv2_1 --test homogeneity

# For additivity testing on conv1_1
python main_generate_stimuli.py --layer conv1_1 --test additivity

# For both tests
python main_generate_stimuli.py --layer conv2_1 --test both

# Retrain PCA model
python main_generate_stimuli.py --layer conv2_1 --test homogeneity --retrain_pca
```

### Command Line Arguments

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--layer` | `conv1_1`, `conv2_1` | `conv2_1` | Target VGG layer |
| `--test` | `homogeneity`, `additivity`, `both` | `homogeneity` | Test type |
| `--source_dir` | path | `./source_images/` | Source images directory |
| `--output_dir` | path | auto-generated | Output directory |
| `--retrain_pca` | flag | False | Force PCA retraining |

---

## 🔬 Workflow Details

### 1. Feature Extraction
- Use pretrained VGG16-BN from torchvision
- Extract features from `conv1_1` (early layer) or `conv2_1` (mid layer)
- Features shape: `(batch, n_filters, height, width)`

### 2. Normalization
- Apply batch normalization statistics (mean, std) from VGG
- Solves the problem of different filter scales
- Reshape to `(n_samples, n_filters)` for PCA

### 3. PCA Transformation
- Train IncrementalPCA on all source images
- Transform features to PCA space
- Modify specific principal components:
  - **Homogeneity**: Multiply single PC by factors (0.5, 2.0, -1.0, etc.)
  - **Additivity**: Combine two PCs (PC1 + PC2)

### 4. Inverse Transformation
- Inverse PCA transform back to feature space
- Denormalize using batch statistics
- Result: Modified VGG features

### 5. Image Synthesis
- Use gradient descent to optimize image pixels
- Minimize MSE between synthesized image features and target features
- Generate stimulus images for psychophysical experiments

---


## 📁 Output Structure

```
exp1_stimuli/
├── conv1_1/
│   ├── image001_pc1_x1.1.png
│   ├── image001_pc1_x5.0.png
│   └── ...
├── conv2_1/
    └── ...
    
exp2_stimuli/
├── conv1_1/
│   ├── image001_pc1_x1.1.png
│   ├── image001_pc1_x5.0.png
│   └── ...
├── conv2_1/
    └── ...
```

---

## 🔧 Configuration

Edit `config.py` to customize:

- **Model**: VGG architecture and weights
- **Layers**: Add new layers with their parameters
- **PCA**: Number of components, batch size
- **Synthesis**: Optimization steps, learning rate (in vgg_img_1v1.py)
- **Modification factors**: Different magnitudes for modifying features
- **Paths**: Input/output directories

---

## 📖 Key Concepts

### Why PCA?
1. **Decorrelation**: Makes feature dimensions linearly independent
2. **Scale normalization**: Different filters have different response magnitudes
3. **Dimensionality reduction**: Focus on principal variation directions

### Why Batch Normalization Statistics?
VGG16-BN learns running statistics during training. Using these ensures:
- Features are properly scaled
- Consistent with how the network was trained
- Easier optimization during image synthesis

### Image Synthesis Details
- **Loss**: MSE between target features and synthesized image features
- **Optimizer**: Adam with learning rate decay
- **Initialization**: Start from source image (not random noise)
- **Regularization**: `early_layers_level` parameter (0 for early layers)

---

## 🧪 Psychophysics Experiment

### 2-AFC Paradigm
- **Method**: Ψ-method (Kontsevich & Tyler, 1999)
- **Display**: 500ms presentation, fixation point
- **Trials**: 40 per run + 2 practice
- **Participants**: Measure perceptual thresholds

### Analysis
- Extract just-noticeable-difference (JND) thresholds
- Fit TvC curves to data
- Test linearity hypothesis 

---

## 💡 Tips for Use

### For Psychophysics Experiments
1. Start with a small set of source images to test pipeline
2. Check synthesized images visually before running experiments
3. Save PCA models to ensure consistency across sessions
4. Use `--retrain_pca` if source images change

### For Further Analysis
- PCA models are saved in `pca_models/` for reuse
- Explained variance ratios help interpret PC importance

### Debugging
- Set `n_steps=100` in config for faster testing
- Check feature statistics with `get_feature_statistics()`
- Verify normalization: normalized features should have mean≈0, std≈1

---

## 📊 Data Analysis

After conducting psychophysical experiments with the generated stimuli, analyze discrimination thresholds using the included analysis scripts.

### Analysis Scripts

```
analysis
├── analysis_utils.py          # Shared utilities
├── analyze_homogeneity.py     # Homogeneity test analysis (TvC fit & plot)
└── analyze_additivity.py      # Additivity test analysis (TvC fit & plot)
```

### Analysis Workflow

1. **Load psychophysical data** - .dat file with threshold measurements
2. **Organize by condition** - Group by image, layer, and PC
3. **Calculate statistics** - Mean and SEM across trials/images
4. **Fit Foley 1994 model** - Contrast discrimination model
5. **Generate TvC plots** - Threshold vs Contrast curves

### Output

```
plot_hom/                      # Homogeneity test results
  ├── example_cih_TvC_conv1_1.png
  └── ...

plot_add/                      # Additivity test results
  ├── example_cih.png
  └── ...
```

### Foley 1994 Model

The analysis fits a simplified contrast discrimination model:

```
R(E) = k * E^p / (E^q + Z)
```

Where `E` is the pedestal contrast, and parameters `k`, `p`, `q`, `Z` are estimated via curve fitting.

---

## 👤 Author

**I-Hang Chen**  
Master's Thesis Project  
Testing the Linearity in the Correspondence between dCNN and Human Visual System

---

## 📝 License

This code is for research and educational purposes.
