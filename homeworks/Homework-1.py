# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Solution 1: Linear Regression
#
# Complete implementation of linear regression using three different approaches.

# %% [markdown]
# ## Setup and Data Loading

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Load the IID regression data
data_path = Path("../Data/Homework-1/regression_data_iid.npz")
data = np.load(data_path)

x = data['x']
y = data['y'] 
y_true = data['y_true']

print(f"Data loaded: {len(x)} points")
print(f"X range: [{x.min():.2f}, {x.max():.2f}]")
print(f"Y range: [{y.min():.2f}, {y.max():.2f}]")

# %% [markdown]
# Let's visualize the data first:

# %%
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, color='blue', s=20, label='Noisy data')
plt.plot(x, y_true, 'r-', linewidth=2, label='True function')
plt.xlabel('x')
plt.ylabel('y')
plt.title('IID Regression Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## Problem 1.1: Explicit Linear Regression Solution

# %%
def explicit_linear_regression(x, y):
    """
    Solve linear regression using the explicit normal equation.
    
    Args:
        x: Input features (n_samples,)
        y: Target values (n_samples,)
    
    Returns:
        beta: Parameters [intercept, slope]
    """
    # TODO FILL THIS IN
    pass
    
# Fit the model
beta_explicit = explicit_linear_regression(x, y)
print(f"Explicit solution: intercept = {beta_explicit[0]:.4f}, slope = {beta_explicit[1]:.4f}")

# Make predictions
y_pred_explicit = beta_explicit[0] + beta_explicit[1] * x

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, color='blue', s=20, label='Noisy data')
plt.plot(x, y_pred_explicit, 'r-', linewidth=2, label='Predictions')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Explicit Linear Regression Predictions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## Problem 1.2: Gradient Descent Implementation
# This problem practices using gradient descent to solve minimization problems. In practice, this might not be how you actually do linear regression, but it's warmup for other problems in the future.

# %%
def gradient_descent_linear_regression(x, y, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
    """
    Solve linear regression using gradient descent.
    
    Args:
        x: Input features (n_samples,)
        y: Target values (n_samples,)
        learning_rate: Step size for gradient descent
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
    
    Returns:
        beta: Parameters [intercept, slope]
        costs: Cost function values over iterations
    """
    # TODO FILL THIS IN
    pass
    
    return beta, costs

def predictor(beta_values, x):
    """
    Make predictions using linear regression parameters.
    
    Args:
        beta_values: Array of regression parameters [intercept, slope]
        x: Input features (n_samples,)
    
    Returns:
        y_pred: Predicted values (n_samples,)
    """
    return beta_values[0] + beta_values[1] * x

# Fit the model
beta_gd, costs = gradient_descent_linear_regression(x, y, learning_rate=0.1, max_iterations=1000)
print(f"Gradient descent solution: intercept = {beta_gd[0]:.4f}, slope = {beta_gd[1]:.4f}")

# Make predictions
y_pred_gd = predictor(beta_gd, x)

# Plot convergence
plt.figure(figsize=(8, 5))
plt.plot(costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Gradient Descent Convergence')
plt.grid(True, alpha=0.3)
plt.show()

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, color='blue', s=20, label='Noisy data')
plt.plot(x, y_pred_gd, 'r-', linewidth=2, label='Predictions')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Predictions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ### Problem 1.2 Bonus
# Can you make the gradient descent solution run faster (in wall-clock time) than the explicit solution?

# %% [markdown]
# ## Problem 1.3: Scipy Optimization
# For this problem, I want you to be aware of existing tools for doing linear regression. Take a look at the scipy documentation to find how to use its linear regression functions. In particular, take a look at the minimize function from scipy.optimize!

# %%
def scipy_linear_regression(x, y):
    """
    Solve linear regression using scipy optimization.
    
    Args:
        x: Input features (n_samples,)
        y: Target values (n_samples,)
    
    Returns:
        beta: Parameters [intercept, slope]
    """
    # TODO FILL THIS IN
    pass 
    
# Fit the model
beta_scipy = scipy_linear_regression(x, y)
print(f"Scipy solution: intercept = {beta_scipy[0]:.4f}, slope = {beta_scipy[1]:.4f}")

# Make predictions
y_pred_scipy = beta_scipy[0] + beta_scipy[1] * x

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, color='blue', s=20, label='Noisy data')
plt.plot(x, y_pred_scipy, 'r-', linewidth=2, label='Predictions')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Explicit Linear Regression Predictions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## Problem 1.4: Comparison and Analysis

# %%
# Compare solutions
print("=== Solution Comparison ===")
print(f"Explicit:         intercept = {beta_explicit[0]:.6f}, slope = {beta_explicit[1]:.6f}")
print(f"Gradient Descent: intercept = {beta_gd[0]:.6f}, slope = {beta_gd[1]:.6f}")
print(f"Scipy:            intercept = {beta_scipy[0]:.6f}, slope = {beta_scipy[1]:.6f}")

# Compute MSE for each method
mse_explicit = np.mean((y_pred_explicit - y)**2)
mse_gd = np.mean((y_pred_gd - y)**2)
mse_scipy = np.mean((y_pred_scipy - y)**2)

print("\n=== Mean Squared Error ===")
print(f"Explicit:         MSE = {mse_explicit:.6f}")
print(f"Gradient Descent: MSE = {mse_gd:.6f}")
print(f"Scipy:            MSE = {mse_scipy:.6f}")

# %% [markdown]
# ### Visualization of Results

# %%
plt.figure(figsize=(12, 8))
plt.scatter(x, y, alpha=0.6, color='lightblue', s=20, label='Noisy data')
plt.plot(x, y_true, 'k--', linewidth=2, label='True function')

# Sort x for smooth lines
sort_idx = np.argsort(x)
x_sorted = x[sort_idx]

plt.plot(x_sorted, (beta_explicit[0] + beta_explicit[1] * x_sorted), 'r-', 
         linewidth=2, label=f'Explicit (MSE: {mse_explicit:.4f})')
plt.plot(x_sorted, (beta_gd[0] + beta_gd[1] * x_sorted), 'g-', 
         linewidth=2, label=f'Gradient Descent (MSE: {mse_gd:.4f})')
plt.plot(x_sorted, (beta_scipy[0] + beta_scipy[1] * x_sorted), 'b-', 
         linewidth=2, label=f'Scipy (MSE: {mse_scipy:.4f})')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of Linear Regression Methods')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## What should we learn from this and expect to see?
#
# 1. **Accuracy**: All three methods should give essentially identical results (within numerical precision) 
#    since they're all solving the same optimization problem. The explicit solution is mathematically exact,
#    while gradient descent and scipy are iterative approximations that converge to the same solution.
#
# 2. **Computational Efficiency**: 
#    - **Explicit**: Fastest for small problems, involves matrix operations
#    - **Gradient Descent**: Slowest, often requires many iterations
#    - **Scipy**: Fast, uses optimizations to avoid taking matrix inverses. 
#
# 3. **Convergence**: Gradient descent should converge in ~50-200 iterations depending on learning rate.
#
# 4. **Parameter Differences**: Minimal differences due to numerical precision. Gradient descent might 
#    have slightly different results due to random initialization and early stopping.
#
# 5. **When to Use Each Method**:
#    - **Explicit**: Small datasets, when you need exact solution, mostly used for educational purposes.
#    - **Gradient Descent**: Primarily used here for educational purposes.
#    - **Scipy**: Production code, complex cost functions, when you need robust optimization

# %% [markdown]
# ## Problem 1.5 Extension to Polynomial Features
# If you visualized the data above, you might note that it's not actually linear. As discussed in class, you can construct your own features within which the problem looks linear. Here, we'll construct some polynomial features and use those for lienar regression.

# %%
def polynomial_regression(x, y, degree=2):
    """
    Solve polynomial regression using explicit solution. Use a 2d polynomial. 
    
    Args:
        x: Input features (n_samples,)
        y: Target values (n_samples,)
        degree: Degree of polynomial
    
    Returns:
        beta: Parameters [intercept, x^1 coeff, x^2 coeff, ...]
    """
    pass

    # TODO FILL THIS IN
    return beta

# Fit quadratic model (matches our true function better)
beta_poly = polynomial_regression(x, y, degree=2)
print(f"Polynomial solution: intercept = {beta_poly[0]:.4f}, x = {beta_poly[1]:.4f}, x^2 = {beta_poly[2]:.4f}")

# Compare with true function coefficients: f(x) = 1.0 + 2.5*x + 0.3*x^2
print(f"True function:       intercept = 1.0000, x = 2.5000, x^2 = 0.3000")

# Make predictions
y_pred_poly = beta_poly[0] + beta_poly[1] * x + beta_poly[2] * x**2
mse_poly = np.mean((y_pred_poly - y)**2)

print(f"Polynomial MSE: {mse_poly:.6f} (should be lower than linear!)")

# Visualize polynomial fit
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, color='lightblue', s=20, label='Noisy data')
plt.plot(x_sorted, y_true[sort_idx], 'k--', linewidth=2, label='True function')
plt.plot(x_sorted, (beta_explicit[0] + beta_explicit[1] * x_sorted), 'r-', 
         linewidth=2, label=f'Linear (MSE: {mse_explicit:.4f})')

y_pred_poly_sorted = beta_poly[0] + beta_poly[1] * x_sorted + beta_poly[2] * x_sorted**2
plt.plot(x_sorted, y_pred_poly_sorted, 'purple', 
         linewidth=2, label=f'Quadratic (MSE: {mse_poly:.4f})')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear vs Polynomial Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# # Problem 2: Non-linear Data and Cross-Validation
# In the previous problem, we focused on a dataset that was pretty explicitly close to linear. Because we were always fitting it with a linear function and the fitting process had no hyperparameters, there was no possibility of overfitting. Now, we explore a setting where we have some control over the function that we're fitting and so there's the possibility of overfitting. We explore fitting with polynomials of different degree and see how cross-k validation can be used to help us avoid some amount of overfitting. We'll get some practice implementing it ourselves and then make sure we also know how to do it in scikit-learn. 
#
# Now let's work with a more complex, non-linear dataset and explore:
# - Train-test splits and model complexity
# - K-fold cross-validation (both manual and scikit-learn implementations)
# - Overfitting detection and prevention

# %% [markdown]
# ## Generate Non-linear Dataset

# %%
def generate_nonlinear_data(n_samples=100, noise_std=0.3, seed=42):
    """
    Generate a complex non-linear dataset for testing polynomial regression.
    
    Args:
        n_samples: Number of data points to generate
        noise_std: Standard deviation of Gaussian noise to add
        seed: Random seed for reproducibility
    
    Returns:
        x: Input features (n_samples,)
        y: Noisy target values (n_samples,)
        y_true: True function values without noise (n_samples,)
    """
    np.random.seed(seed)
    
    # Generate input points
    x = np.linspace(-2, 2, n_samples)
    
    # Complex non-linear function
    y_true = 0.5 * x**3 - 2 * x**2 + x + 1 + 0.5 * np.sin(5 * x)
    
    # Add noise
    y = y_true + np.random.normal(0, noise_std, n_samples)
    
    return x, y, y_true

# Generate the dataset
x_nl, y_nl, y_true_nl = generate_nonlinear_data(n_samples=80, noise_std=0.4)

# Visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(x_nl, y_nl, alpha=0.7, color='blue', s=30, label='Noisy data')
plt.plot(x_nl, y_true_nl, 'r-', linewidth=2, label='True function')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Non-linear Dataset')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# # Problem 2A: Manual Implementation (No Scikit-Learn)
#
# First, let's implement polynomial regression and cross-validation from scratch to understand the concepts. Take your data, split it into 70% train and 30% test and then perform 5-fold cross-validation to create estimates of the std. deviation of the error for each of the polynomial degrees. 

# %% [markdown]
# ## Manual Train-Test Split and Polynomial Features

# %%
def manual_train_test_split(X, y, test_size=0.3, random_state=42):
    """
    Manually implement train-test split without using sklearn.
    
    Args:
        X: Input features (n_samples, n_features)
        y: Target values (n_samples,)
        test_size: Fraction of data to use for testing (0.0 to 1.0)
        random_state: Random seed for reproducible splits
    
    Returns:
        X_train: Training features
        X_test: Test features  
        y_train: Training labels
        y_test: Test labels
    """
    # TODO FILL THIS IN
    pass

def create_polynomial_features(x, degree):
    """
    Create polynomial features manually (without sklearn).
    
    Args:
        x: Input features (n_samples, 1) or (n_samples,)
        degree: Maximum degree of polynomial features
    
    Returns:
        X: Design matrix with polynomial features (n_samples, degree+1)
           Columns are [1, x, x^2, x^3, ..., x^degree]
    """
    # TODO FILL THIS IN
    pass

def fit_polynomial_manual(X, y):
    """
    Fit polynomial regression using the normal equation.
    
    Args:
        X: Design matrix with polynomial features (n_samples, n_features)
        y: Target values (n_samples,)
    
    Returns:
        beta: Polynomial coefficients (n_features,)
    """
    # Add small regularization for numerical stability
    # TODO FILL THIS IN
    pass

def predict_polynomial_manual(X, beta):
    """
    Make predictions using fitted polynomial coefficients.
    
    Args:
        X: Design matrix with polynomial features (n_samples, n_features)
        beta: Polynomial coefficients (n_features,)
    
    Returns:
        y_pred: Predicted values (n_samples,)
    """
    # TODO FILL THIS IN
    pass

def mean_squared_error_manual(y_true, y_pred):
    """
    Calculate Mean Squared Error manually (without sklearn).
    
    Args:
        y_true: True target values (n_samples,)
        y_pred: Predicted values (n_samples,)
    
    Returns:
        mse: Mean squared error (scalar)
    """
    # TODO FILL THIS IN
    pass

# Manual train-test split
x_nl_reshaped = x_nl.reshape(-1, 1)
X_train_manual, X_test_manual, y_train_manual, y_test_manual = manual_train_test_split(
    x_nl_reshaped, y_nl, test_size=0.3, random_state=42
)

# Test different polynomial degrees manually
degrees = range(1, 16)
train_errors_manual = []
test_errors_manual = []

for degree in degrees:
    # Create polynomial features
    X_train_poly = create_polynomial_features(X_train_manual, degree)
    X_test_poly = create_polynomial_features(X_test_manual, degree)
    
    # Fit model
    beta = fit_polynomial_manual(X_train_poly, y_train_manual)
    
    # Make predictions
    y_train_pred = predict_polynomial_manual(X_train_poly, beta)
    y_test_pred = predict_polynomial_manual(X_test_poly, beta)
    
    # Calculate errors
    train_mse = mean_squared_error_manual(y_train_manual, y_train_pred)
    test_mse = mean_squared_error_manual(y_test_manual, y_test_pred)
    
    train_errors_manual.append(train_mse)
    test_errors_manual.append(test_mse)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors_manual, 'o-', label='Training Error', color='blue')
plt.plot(degrees, test_errors_manual, 'o-', label='Test Error', color='red')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Manual Implementation: Training vs Test Error')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.show()

optimal_degree_manual = degrees[np.argmin(test_errors_manual)]
print(f"Manual implementation - Optimal degree: {optimal_degree_manual}")

# %% [markdown]
# ## Manual K-Fold Cross-Validation
# In K-fold cross validation we construct $k$ train test splits. We then train a model for each of these splits and compute the standard deviation of our fits around to the splits. 

# %%
def manual_kfold_split(n_samples, k=5, shuffle=True, random_state=42):
    """
    Manually implement k-fold cross-validation splits.
    
    Args:
        n_samples: Total number of samples in dataset
        k: Number of folds for cross-validation
        shuffle: Whether to shuffle data before splitting
        random_state: Random seed for reproducible splits
    
    Returns:
        folds: List of (train_indices, test_indices) tuples for each fold
    """
    # TODO FILL THIS IN
    pass

def manual_cross_validate(X, y, degree, k=5):
    """
    Perform k-fold cross-validation manually for polynomial regression.
    
    Args:
        X: Input features (n_samples, n_input_features)
        y: Target values (n_samples,)
        degree: Polynomial degree to test
        k: Number of folds for cross-validation
    
    Returns:
        fold_errors: Array of MSE values for each fold (k,)
    """
    n_samples = len(X)
    folds = manual_kfold_split(n_samples, k=k, random_state=42)

    # TODO fill the rest of this in. Basically, you will need to do the fits
    # for each fold and track the final MSE of each fit. 
    pass

# Manual cross-validation
cv_scores_manual = []
cv_stds_manual = []

print("Performing manual 5-fold cross-validation...")
for degree in degrees:
    fold_errors = manual_cross_validate(x_nl_reshaped, y_nl, degree, k=5)
    
    # Filter out infinite values
    valid_errors = fold_errors[np.isfinite(fold_errors)]
    if len(valid_errors) > 0:
        cv_scores_manual.append(valid_errors.mean())
        cv_stds_manual.append(valid_errors.std())
    else:
        cv_scores_manual.append(np.inf)
        cv_stds_manual.append(0)

# Plot manual cross-validation results
plt.figure(figsize=(12, 6))
plt.errorbar(degrees, cv_scores_manual, yerr=cv_stds_manual, 
             marker='o', capsize=5, capthick=2, label='Manual 5-Fold CV')
plt.plot(degrees, test_errors_manual, 'r--', alpha=0.7, label='Manual Single Split')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Manual Implementation: Cross-Validation vs Single Split')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.show()

optimal_degree_cv_manual = degrees[np.argmin(cv_scores_manual)]
print(f"Manual CV - Optimal degree: {optimal_degree_cv_manual}")

# %% [markdown]
# # Problem 2B: Scikit-Learn Implementation
#
# Now let's use scikit-learn to do the same analysis and compare results. You don't have to do anything here, I just wanted to give you some code for how you might implement it. 

# %%
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Scikit-learn train-test split
X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(
    x_nl_reshaped, y_nl, test_size=0.3, random_state=42
)

# Test different polynomial degrees with scikit-learn
train_errors_sk = []
test_errors_sk = []

for degree in degrees:
    # Create polynomial pipeline
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    # Fit the model
    poly_model.fit(X_train_sk, y_train_sk)
    
    # Make predictions
    y_train_pred = poly_model.predict(X_train_sk)
    y_test_pred = poly_model.predict(X_test_sk)
    
    # Calculate errors
    train_mse = mean_squared_error(y_train_sk, y_train_pred)
    test_mse = mean_squared_error(y_test_sk, y_test_pred)
    
    train_errors_sk.append(train_mse)
    test_errors_sk.append(test_mse)

optimal_degree_sk = degrees[np.argmin(test_errors_sk)]
print(f"Scikit-learn - Optimal degree: {optimal_degree_sk}")

# %% [markdown]
# ## Scikit-Learn Cross-Validation

# %%
def evaluate_with_sklearn_kfold(X, y, degrees, k=5):
    """
    Evaluate using scikit-learn k-fold cross-validation.
    
    Args:
        X: Input features (n_samples, n_input_features)
        y: Target values (n_samples,)
        degrees: List of polynomial degrees to evaluate
        k: Number of folds for cross-validation
    
    Returns:
        cv_scores: List of mean cross-validation scores for each degree
        cv_stds: List of standard deviations of cross-validation scores for each degree
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    cv_scores = []
    cv_stds = []
    
    for degree in degrees:
        # Create polynomial pipeline
        poly_model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])
        
        # Perform k-fold cross-validation
        scores = cross_val_score(poly_model, X, y, cv=kf, 
                               scoring='neg_mean_squared_error')
        
        # Convert to positive MSE
        mse_scores = -scores
        
        cv_scores.append(mse_scores.mean())
        cv_stds.append(mse_scores.std())
    
    return cv_scores, cv_stds

# Scikit-learn cross-validation
cv_scores_sk, cv_stds_sk = evaluate_with_sklearn_kfold(x_nl_reshaped, y_nl, degrees, k=5)

optimal_degree_cv_sk = degrees[np.argmin(cv_scores_sk)]
print(f"Scikit-learn CV - Optimal degree: {optimal_degree_cv_sk}")

# %% [markdown]
# ## Comparison: Manual vs Scikit-Learn

# %%
# Compare all implementations
plt.figure(figsize=(15, 10))

# Plot 1: Train-Test Split Comparison
plt.subplot(2, 2, 1)
plt.plot(degrees, train_errors_manual, 'o-', label='Manual Train', color='blue', alpha=0.7)
plt.plot(degrees, test_errors_manual, 'o-', label='Manual Test', color='red', alpha=0.7)
plt.plot(degrees, train_errors_sk, 's--', label='Sklearn Train', color='blue')
plt.plot(degrees, test_errors_sk, 's--', label='Sklearn Test', color='red')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Train-Test Split: Manual vs Scikit-Learn')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Plot 2: Cross-Validation Comparison
plt.subplot(2, 2, 2)
plt.errorbar(degrees, cv_scores_manual, yerr=cv_stds_manual, 
             marker='o', capsize=3, alpha=0.7, label='Manual CV')
plt.errorbar(degrees, cv_scores_sk, yerr=cv_stds_sk, 
             marker='s', capsize=3, label='Sklearn CV')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Cross-Validation: Manual vs Scikit-Learn')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Plot 3: Model Visualization (using optimal degree)
plt.subplot(2, 2, 3)
x_plot = np.linspace(-2, 2, 200)
X_plot = x_plot.reshape(-1, 1)

# Manual implementation
X_plot_poly_manual = create_polynomial_features(X_plot, optimal_degree_cv_manual)
X_train_poly_manual = create_polynomial_features(X_train_manual, optimal_degree_cv_manual)
beta_manual = fit_polynomial_manual(X_train_poly_manual, y_train_manual)
y_plot_manual = predict_polynomial_manual(X_plot_poly_manual, beta_manual)

plt.scatter(x_nl, y_nl, alpha=0.6, color='lightblue', s=20, label='Data')
plt.plot(x_plot, y_plot_manual, 'b-', linewidth=2, label=f'Manual (deg {optimal_degree_cv_manual})')
plt.plot(x_nl, y_true_nl, 'k--', alpha=0.7, label='True function')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Manual Implementation Result')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Scikit-learn visualization
plt.subplot(2, 2, 4)
poly_model_sk = Pipeline([
    ('poly', PolynomialFeatures(degree=optimal_degree_cv_sk)),
    ('linear', LinearRegression())
])
poly_model_sk.fit(X_train_sk, y_train_sk)
y_plot_sk = poly_model_sk.predict(X_plot)

plt.scatter(x_nl, y_nl, alpha=0.6, color='lightblue', s=20, label='Data')
plt.plot(x_plot, y_plot_sk, 'g-', linewidth=2, label=f'Sklearn (deg {optimal_degree_cv_sk})')
plt.plot(x_nl, y_true_nl, 'k--', alpha=0.7, label='True function')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scikit-Learn Implementation Result')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Analysis Summary
#
# **Key Things you might learn / expect:**
#
# 1. **Implementation Consistency**: Manual and scikit-learn implementations should give very similar results,
#    validating our understanding of the underlying algorithms.
#
# 2. **Cross-Validation**: Cross-validation should indicate that there's some noise in which value of beta looks the best. If we didn't cross-validate, we might not see it. 
#
# 3. **Overfitting Detection**: Both approaches clearly show overfitting around degree 8-10, where
#    training error continues decreasing but test/validation error increases.
#
#
# Question: which polynomial degree looks best? Why?
#
# **Your answer here:**

# %% [markdown]
# # Problem 3: Sentiment Analysis - Bag of Words vs OpenAI Embeddings
#
# In this final problem, we'll explore text classification using two very different approaches:
# 1. **Bag of Words (BoW)**: Traditional sparse feature representation
# 2. **OpenAI Embeddings**: Modern dense vector representations
#
# We'll use the [multiclass sentiment analysis dataset](https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset) 
# which contains ~41k text samples with 3 sentiment classes: negative, neutral, positive.
#
# ## Learning Objectives
# - Practice with traditional NLP feature engineering (bag of words)
# - Learn to use modern embedding APIs (OpenAI)
# - Apply linear models to text classification
# - Understand how to do multi-class classification

# %% [markdown]
# ## Load Sentiment Analysis Dataset
#
# **Note**: This will download the dataset from Hugging Face (~4.7MB)

# %%
from datasets import load_dataset
import pandas as pd

def load_sentiment_data(subset_size=1000):
    """
    Load and prepare the sentiment analysis dataset.
    
    Args:
        subset_size: Number of samples to use (for computational efficiency)
    
    Returns:
        train_texts, test_texts, train_labels, test_labels
    """
    print("Loading sentiment analysis dataset from Hugging Face...")
    
    # Load the dataset
    dataset = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset")
    
    # Convert to pandas for easier manipulation
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    
    print(f"Full dataset size: Train={len(train_df)}, Test={len(test_df)}")
    print(f"Classes: {train_df['sentiment'].unique()}")
    print(f"Class distribution:\n{train_df['sentiment'].value_counts()}")
    
    # Take a balanced subset for efficiency
    if subset_size < len(train_df):
        samples_per_class = subset_size // 3
        train_subset = []
        
        for sentiment in ['negative', 'neutral', 'positive']:
            class_data = train_df[train_df['sentiment'] == sentiment].sample(
                n=samples_per_class, random_state=42
            )
            train_subset.append(class_data)
        
        train_df = pd.concat(train_subset).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Also subset test data
    test_subset_size = min(300, len(test_df))
    test_df = test_df.sample(n=test_subset_size, random_state=42).reset_index(drop=True)
    
    print(f"Using subset: Train={len(train_df)}, Test={len(test_df)}")
    
    # Extract texts and labels
    train_texts = train_df['text'].tolist()
    test_texts = test_df['text'].tolist()
    
    # Convert sentiment labels to integers
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    train_labels = train_df['sentiment'].map(label_map).values
    test_labels = test_df['sentiment'].map(label_map).values
    
    return train_texts, test_texts, train_labels, test_labels

# Load the data
train_texts, test_texts, train_labels, test_labels = load_sentiment_data(subset_size=900)

# Show some examples
print("\n=== Sample Texts ===")
for i in range(3):
    sentiment_names = ['negative', 'neutral', 'positive']
    print(f"{sentiment_names[train_labels[i]]}: {train_texts[i][:100]}...")

# %% [markdown]
# ## Problem 3A: Bag of Words Approach
#
# In this section, you'll implement text classification using the traditional "bag of words" approach.
# This creates sparse feature vectors based on word counts in the text.
#
# **Your task**: Complete the `bag_of_words_classification` function below by:
# 1. Constructing a BOW feature representation (you can use sci-kit learn here if you want or implement it yourself)
# 2. Train the model and return results in the defined format

# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def bag_of_words_classification(train_texts, test_texts, train_labels, test_labels):
    """
    Implement sentiment classification using bag of words.
    
    Args:
        train_texts: List of training text samples
        test_texts: List of test text samples
        train_labels: Array of training labels (n_train_samples,)
        test_labels: Array of test labels (n_test_samples,)
    
    Returns:
        dict: Dictionary containing results for the count method with
              train_acc, test_acc, and predictions
    """
    print("=== Bag of Words Approach ===")
    
    # TODO fill in the fitting
    
    # TODO: You need to return a dictionary with the results
    # The format should be:
    # return {
    #     'count': {'train_acc': train_acc_count, 'test_acc': test_acc_count, 'predictions': test_pred_count}
    # }
    pass

# Run bag of words experiments
bow_results = bag_of_words_classification(train_texts, test_texts, train_labels, test_labels)

# %% [markdown]
# ## Problem 3B: OpenAI Embeddings Approach
#
# Modern approach using dense vector representations from OpenAI's embedding models. 
# These are pre-trained models that convert text into high-dimensional vectors that capture semantic meaning.
#
# **Your tasks**:
# 1. Set up your OpenAI API key (see instructions below)
# 2. Complete the `get_openai_embeddings` function to handle API calls
# 3. Complete the `openai_embeddings_classification` function
# 4. Compare results with the bag-of-words approach
#
# ### API Key Setup
# **Note**: You'll need an OpenAI API key. Create one at https://platform.openai.com
#
# **IMPORTANT SECURITY**: Never put API keys directly in your code! 
#
# **Recommended approach**: Use a `.env` file
# 1. Create a `.env` file in your project root (already in .gitignore)
# 2. Add: `OPENAI_API_KEY=your-api-key-here`
# 3. The code below will automatically load it
#
# **WARNING**: Your API key should NEVER be in your commit history or public repos!

# %%
import openai
import os
from openai import OpenAI
import time

# %%
# Choose your API key method (uncomment ONE of the following):

# Option 1: Load from .env file (Recommended)
from dotenv import load_dotenv
load_dotenv()
# IT WILL PRINT TRUE IF THIS SUCCEEDED

# Option 2: Interactive input (most secure for shared environments). 
# I am not making it the default because it's more annoying but it can be the more secure one. 
# the sense in which it is the less secure one is it can induce you to write down your private key
# and store it somewhere where people can easily access it. 

# import getpass
# os.environ['OPENAI_API_KEY'] = getpass.getpass("Enter OpenAI API key: ")

# %% [markdown]
# ### Step 1: Implement OpenAI Embeddings Function
#
# The function below handles API calls to OpenAI. It's implemented for you but all you need to know is that it takes the total text and converts it into a feature vector.
#

# %%
def get_openai_embeddings(texts, model="text-embedding-3-small", batch_size=100):
    """
    Get OpenAI embeddings for a list of texts.
    
    Args:
        texts: List of text strings
        model: OpenAI embedding model to use
        batch_size: Number of texts to process at once
    
    Returns:
        embeddings: numpy array of shape (n_texts, embedding_dim)
    """
    # Get API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("WARNING: No OpenAI API key found!")
        print("Please create a .env file with OPENAI_API_KEY=your-key")
        print("For this demo, we'll use random embeddings instead")
        
        # Return random embeddings for demo purposes
        np.random.seed(42)
        return np.random.randn(len(texts), 1536)  # text-embedding-3-small dimension
    
    print("API key loaded successfully")
    
    client = OpenAI(api_key=api_key)
    
    print(f"Getting OpenAI embeddings for {len(texts)} texts...")
    print(f"Model: {model}")
    
    all_embeddings = []
    
    # Process in batches to avoid rate limits
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        try:
            # Get embeddings for this batch
            response = client.embeddings.create(
                input=batch_texts,
                model=model
            )
            
            # Extract embeddings
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            print(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            print("Using random embeddings for this batch")
            # Fallback to random embeddings
            batch_size_actual = len(batch_texts)
            random_embeddings = np.random.randn(batch_size_actual, 1536).tolist()
            all_embeddings.extend(random_embeddings)
    
    return np.array(all_embeddings)

# %% [markdown]
# ### Step 2: Implement Classification with OpenAI Embeddings
#
# **Your task**: Complete this function by:
# 1. Understanding how embeddings are obtained (already implemented)
# 2. Adding print statements to examine embedding properties
# 3. Training a logistic regression classifier on the embeddings
# 4. Returning results in the correct format

# %%
def openai_embeddings_classification(train_texts, test_texts, train_labels, test_labels):
    """
    Implement sentiment classification using OpenAI embeddings.
    
    Args:
        train_texts: List of training text samples
        test_texts: List of test text samples
        train_labels: Array of training labels (n_train_samples,)
        test_labels: Array of test labels (n_test_samples,)
    
    Returns:
        dict: Dictionary containing train_acc, test_acc, predictions, 
              embeddings_train, and embeddings_test
    """
    print("=== OpenAI Embeddings Approach ===")
    
    # Get embeddings (this may take a few minutes and cost ~$0.01-0.05)
    print("Getting training embeddings...")
    X_train_embeddings = get_openai_embeddings(train_texts)
    
    print("Getting test embeddings...")
    X_test_embeddings = get_openai_embeddings(test_texts)
    
    # TODO: Add print statements to examine embedding properties
    # You should print:
    # - Embedding dimensionality: X_train_embeddings.shape[1] 
    # - Embedding range: [X_train_embeddings.min():.3f}, {X_train_embeddings.max():.3f}]
    
    # TODO: Train a logistic regression classifier on embeddings
    # Use LogisticRegression(max_iter=1000, random_state=42)
    
    # TODO: Make predictions and calculate accuracies
    # Use accuracy_score to calculate train_acc_embeddings and test_acc_embeddings
        
    # TODO: Return results in the specified format:
    # return {
    #     'train_acc': train_acc_embeddings, 
    #     'test_acc': test_acc_embeddings, 
    #     'predictions': test_pred_embeddings,
    #     'embeddings_train': X_train_embeddings,
    #     'embeddings_test': X_test_embeddings
    # }
    pass

# Run OpenAI embeddings experiment
openai_results = openai_embeddings_classification(train_texts, test_texts, train_labels, test_labels)

# %% [markdown]
# ## Problem 3C: Compare Approaches
#
# **Your task**: Analyze the results from both approaches by:
# 1. Running the comparison code below
# 2. Examining the accuracy differences
# 3. Understanding the trade-offs between sparse (BoW) vs dense (embeddings) representations
# 4. Answering the reflection questions at the end

# %%
# Summary comparison
print("\n" + "="*60)
print("SENTIMENT ANALYSIS COMPARISON")
print("="*60)

approaches = [
    ("Count Vectorizer (BoW)", bow_results['count']),
    ("OpenAI Embeddings", openai_results)
]

print(f"{'Approach':<25} {'Train Acc':<10} {'Test Acc':<10} {'Overfitting':<12}")
print("-" * 65)

for name, results in approaches:
    train_acc = results['train_acc']
    test_acc = results['test_acc']
    overfitting = train_acc - test_acc
    print(f"{name:<25} {train_acc:<10.3f} {test_acc:<10.3f} {overfitting:<12.3f}")

# %% [markdown]
# ## Detailed Analysis

# %%
# Classification reports
sentiment_names = ['negative', 'neutral', 'positive']

print("\n1. OpenAI Embeddings Results:")
print(classification_report(test_labels, openai_results['predictions'], 
                          target_names=sentiment_names))

# %% [markdown]
# ## Visualize Results

# %%
# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Accuracy comparison
methods = ['Count BoW', 'OpenAI']
train_accs = [bow_results['count']['train_acc'], openai_results['train_acc']]
test_accs = [bow_results['count']['test_acc'],  openai_results['test_acc']]

axes[0].bar(methods, train_accs, alpha=0.7, label='Train Accuracy', color='blue')
axes[0].bar(methods, test_accs, alpha=0.7, label='Test Accuracy', color='red')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Method Comparison')
axes[0].legend()
axes[0].tick_params(axis='x', rotation=45)

# Confusion matrix for best method
best_method = max(approaches, key=lambda x: x[1]['test_acc'])
best_predictions = best_method[1]['predictions']

cm = confusion_matrix(test_labels, best_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=sentiment_names, yticklabels=sentiment_names, ax=axes[1])
axes[1].set_title(f'Confusion Matrix - {best_method[0]}')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

# Overfitting analysis
overfitting_scores = [results['train_acc'] - results['test_acc'] for _, results in approaches]
axes[2].bar(methods, overfitting_scores, color='green', alpha=0.7)
axes[2].set_ylabel('Overfitting (Train - Test)')
axes[2].set_title('Overfitting Comparison')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Error Analysis

# %%
# Find examples where methods disagree
bow_pred = bow_results['count']['predictions']
openai_pred = openai_results['predictions']

# Find disagreements
disagreements = np.where(bow_pred != openai_pred)[0]

print(f"\n=== ERROR ANALYSIS ===")
print(f"Methods disagree on {len(disagreements)}/{len(test_labels)} samples ({len(disagreements)/len(test_labels)*100:.1f}%)")

if len(disagreements) > 0:
    print(f"\nSample disagreements:")
    for i in disagreements[:5]:  # Show first 5 disagreements
        true_label = sentiment_names[test_labels[i]]
        bow_label = sentiment_names[bow_pred[i]]
        openai_label = sentiment_names[openai_pred[i]]
        text = test_texts[i][:100] + "..." if len(test_texts[i]) > 100 else test_texts[i]
        
        print(f"\nText: {text}")
        print(f"True: {true_label}, BoW: {bow_label}, OpenAI: {openai_label}")
