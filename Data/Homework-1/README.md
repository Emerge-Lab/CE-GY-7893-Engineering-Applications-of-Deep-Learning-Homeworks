# Homework 1 - 1D Regression Data

This directory contains a data generation script for creating noisy 1D regression datasets.

## Files

- `generate_regression_data.py`: Main script for generating regression data
- `regression_data_iid.npz`: Generated dataset with IID Gaussian noise
- `regression_data_non_iid.npz`: Generated dataset with non-IID noise
- `regression_datasets_comparison.png`: Visualization comparing both datasets

## Usage

```bash
# Generate the datasets
uv run python generate_regression_data.py
```

## Dataset Characteristics

### True Function
The underlying function is: `f(x) = 2.5*x + 1.0 + 0.3*x²`

This is primarily linear with a small quadratic component, making it suitable for testing linear regression methods while still providing some interesting nonlinearity.

### IID Gaussian Noise Dataset
- **Noise type**: Independent and identically distributed Gaussian noise
- **Characteristics**: 
  - Constant variance across all input values
  - No correlation between noise at different points
  - Clean baseline for regression algorithms

### Non-IID Noise Dataset
- **Noise type**: Complex, realistic noise with multiple components
- **Characteristics**:
  - **Heteroscedastic**: Noise variance increases with |x|
  - **Autocorrelated**: Neighboring points have correlated noise (AR(1) process)
  - **Outliers**: 5% of points have additional outlier noise
  - More challenging and realistic for testing robust regression methods

## Loading Data

```python
import numpy as np

# Load IID data
data_iid = np.load('regression_data_iid.npz')
x_iid, y_iid, y_true_iid = data_iid['x'], data_iid['y'], data_iid['y_true']

# Load non-IID data  
data_non_iid = np.load('regression_data_non_iid.npz')
x_non_iid, y_non_iid, y_true_non_iid = data_non_iid['x'], data_non_iid['y'], data_non_iid['y_true']
```

## Purpose

These datasets are designed to:
1. Test regression algorithms under different noise conditions
2. Understand the impact of noise assumptions on model performance
3. Practice robust regression techniques
4. Explore the difference between IID and non-IID noise scenarios