"""
1D Regression Data Generator

This script generates noisy 1D regression datasets for homework exercises.
Two types of noise are implemented:
1. IID Gaussian noise - independent and identically distributed
2. Non-IID noise - heteroscedastic (variance depends on input) + autocorrelated

Usage:
    python generate_regression_data.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def true_function(x):
    """
    The underlying true function for regression.
    A primarily linear function with a small nonlinear component.
    Good for testing linear regression methods.
    """
    # Primary linear component
    linear_part = 2.5 * x + 1.0
    
    # Small nonlinear component (can be approximated by linear methods)
    nonlinear_part = 0.3 * x**2
    
    return linear_part + nonlinear_part


def generate_iid_data(n_samples=200, noise_std=0.8, x_range=(-3, 3), seed=42):
    """
    Generate 1D regression data with IID Gaussian noise.
    
    Args:
        n_samples: Number of data points
        noise_std: Standard deviation of Gaussian noise
        x_range: Tuple of (min, max) for x values
        seed: Random seed for reproducibility
    
    Returns:
        x: Input features (n_samples,)
        y: Noisy targets (n_samples,)
        y_true: True function values without noise (n_samples,)
    """
    np.random.seed(seed)
    
    # Generate input points
    x = np.random.uniform(x_range[0], x_range[1], n_samples)
    x = np.sort(x)  # Sort for better visualization
    
    # Generate true function values
    y_true = true_function(x)
    
    # Add IID Gaussian noise
    noise = np.random.normal(0, noise_std, n_samples)
    y = y_true + noise
    
    return x, y, y_true


def generate_non_iid_data(n_samples=200, base_noise_std=0.3, x_range=(-3, 3), seed=42):
    """
    Generate 1D regression data with non-IID noise.
    The noise has two components:
    1. Heteroscedastic: noise variance depends on input magnitude
    2. Autocorrelated: neighboring points have correlated noise
    
    Args:
        n_samples: Number of data points
        base_noise_std: Base standard deviation for noise
        x_range: Tuple of (min, max) for x values
        seed: Random seed for reproducibility
    
    Returns:
        x: Input features (n_samples,)
        y: Noisy targets (n_samples,)
        y_true: True function values without noise (n_samples,)
    """
    np.random.seed(seed)
    
    # Generate input points
    x = np.random.uniform(x_range[0], x_range[1], n_samples)
    x = np.sort(x)  # Sort for autocorrelation to make sense
    
    # Generate true function values
    y_true = true_function(x)
    
    # Heteroscedastic noise: variance increases with |x|
    noise_std = base_noise_std * (1 + 0.8 * np.abs(x))
    
    # Generate base noise
    base_noise = np.random.normal(0, 1, n_samples)
    
    # Add autocorrelation using AR(1) process
    autocorr_coef = 0.7
    autocorr_noise = np.zeros(n_samples)
    autocorr_noise[0] = base_noise[0]
    
    for i in range(1, n_samples):
        autocorr_noise[i] = (autocorr_coef * autocorr_noise[i-1] + 
                           np.sqrt(1 - autocorr_coef**2) * base_noise[i])
    
    # Scale by heteroscedastic standard deviation
    noise = noise_std * autocorr_noise
    
    # Add occasional outliers (5% of points)
    outlier_mask = np.random.random(n_samples) < 0.05
    outlier_noise = np.random.normal(0, 3 * base_noise_std, n_samples)
    noise[outlier_mask] += outlier_noise[outlier_mask]
    
    y = y_true + noise
    
    return x, y, y_true


def plot_data(x_iid, y_iid, y_true_iid, x_non_iid, y_non_iid, y_true_non_iid, save_path=None):
    """
    Plot both datasets for comparison.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # IID data plot
    ax1.scatter(x_iid, y_iid, alpha=0.6, color='blue', s=20, label='Noisy data')
    ax1.plot(x_iid, y_true_iid, 'r-', linewidth=2, label='True function')
    ax1.set_title('IID Gaussian Noise', fontsize=14)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Non-IID data plot
    ax2.scatter(x_non_iid, y_non_iid, alpha=0.6, color='green', s=20, label='Noisy data')
    ax2.plot(x_non_iid, y_true_non_iid, 'r-', linewidth=2, label='True function')
    ax2.set_title('Non-IID Noise (Heteroscedastic + Autocorrelated)', fontsize=14)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def save_data(x, y, y_true, filename):
    """
    Save data to numpy format.
    """
    data = {
        'x': x,
        'y': y,
        'y_true': y_true
    }
    np.savez(filename, **data)
    print(f"Data saved to: {filename}")


def main():
    """
    Generate and save both types of regression data.
    """
    print("Generating 1D regression datasets...")
    
    # Create output directory
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    
    # Generate IID data
    print("Generating IID Gaussian noise data...")
    x_iid, y_iid, y_true_iid = generate_iid_data(
        n_samples=200, 
        noise_std=0.8, 
        seed=42
    )
    
    # Generate non-IID data
    print("Generating non-IID noise data...")
    x_non_iid, y_non_iid, y_true_non_iid = generate_non_iid_data(
        n_samples=200, 
        base_noise_std=0.4, 
        seed=42
    )
    
    # Save datasets
    save_data(x_iid, y_iid, y_true_iid, output_dir / "regression_data_iid.npz")
    save_data(x_non_iid, y_non_iid, y_true_non_iid, output_dir / "regression_data_non_iid.npz")
    
    # Create visualization
    print("Creating visualization...")
    plot_data(
        x_iid, y_iid, y_true_iid,
        x_non_iid, y_non_iid, y_true_non_iid,
        save_path=output_dir / "regression_datasets_comparison.png"
    )
    
    print("\nDataset characteristics:")
    print(f"IID data - Mean noise std: {np.std(y_iid - y_true_iid):.3f}")
    print(f"Non-IID data - Mean noise std: {np.std(y_non_iid - y_true_non_iid):.3f}")
    print(f"Non-IID data - Noise std range: {np.std(y_non_iid - y_true_non_iid):.3f}")
    
    # Show sample autocorrelation for non-IID data
    noise_non_iid = y_non_iid - y_true_non_iid
    autocorr = np.corrcoef(noise_non_iid[:-1], noise_non_iid[1:])[0, 1]
    print(f"Non-IID data - Noise autocorrelation (lag-1): {autocorr:.3f}")


if __name__ == "__main__":
    main()