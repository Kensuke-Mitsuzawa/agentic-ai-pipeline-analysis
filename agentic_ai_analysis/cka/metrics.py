import numpy as np

def center_kernel_matrix(K: np.ndarray) -> np.ndarray:
    """Centers the kernel matrix K."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def compute_dimension_wise_median_heuristic(X: np.ndarray) -> np.ndarray:
    """
    Computes the dimension-wise median heuristic for the Gaussian kernel bandwidth.
    
    See "E.1 Variable-wise Median Heuristic" in https://arxiv.org/pdf/2311.01537#page=15.85
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        
    Returns:
        np.ndarray: Bandwidth vector of shape (n_features,)
    """
    n_samples, n_features = X.shape
    bandwidths = np.zeros(n_features)
    
    if n_samples <= 1:
        return np.ones(n_features)
        
    for d in range(n_features):
        feature_col = X[:, d:d+1]
        # Compute pairwise absolute differences for the d-th feature
        diffs = np.abs(feature_col - feature_col.T)
        
        # We only want the upper triangle (excluding diagonal) to compute the median
        upper_tri_indices = np.triu_indices(n_samples, k=1)
        pairwise_distances = diffs[upper_tri_indices]
        
        # Median heuristic: length scale = median pairwise distance
        median_dist = np.median(pairwise_distances)
        
        # Avoid zero bandwidth if all points have the same feature value
        if median_dist == 0:
            median_dist = 1e-6
            
        bandwidths[d] = median_dist
            
    return bandwidths

def compute_gaussian_kernel_variable_bandwidth(X: np.ndarray, bandwidths: np.ndarray) -> np.ndarray:
    """
    Computes the Gaussian kernel matrix using a variable-wise bandwidth vector.
    
    K(x_i, x_j) = exp( - sum_{d=1}^D (x_i,d - x_j,d)^2 / (2 * bandwidth_d^2) )
    """
    n_samples, n_features = X.shape
    K = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(n_samples):
            # Squared differences scaled by bandwidths
            diff_sq = ((X[i] - X[j]) ** 2) / (2 * (bandwidths ** 2))
            K[i, j] = np.exp(-np.sum(diff_sq))
            
    return K

def center_kernel_matrix_unbiased(K: np.ndarray) -> np.ndarray:
    """
    Centers the kernel matrix K and zeros the diagonal to compute unbiased HSIC.
    Based on Song et al. (2012) - Feature Selection via Dependence Maximization.
    """
    n = K.shape[0]
    np.fill_diagonal(K, 0.0)
    
    # We will just use the standard centering! CKA does not usually require unbiased HSIC.
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def compute_unbiased_hsic(K_X: np.ndarray, K_Y: np.ndarray) -> float:
    """
    Computes the unbiased estimator of HSIC.
    See Song et al. (2012) or the unbiased HSIC formulation.
    """
    n = K_X.shape[0]
    if n < 4:
        return 0.0
        
    np.fill_diagonal(K_X, 0.0)
    np.fill_diagonal(K_Y, 0.0)
    
    # K_X @ ones
    K_X_sum_row = np.sum(K_X, axis=1)
    K_Y_sum_row = np.sum(K_Y, axis=1)
    
    term1 = np.sum(K_X * K_Y)
    term2 = np.sum(K_X_sum_row) * np.sum(K_Y_sum_row) / ((n - 1) * (n - 2))
    term3 = 2 * np.sum(K_X_sum_row * K_Y_sum_row) / (n - 2)
    
    hsic_val = (term1 + term2 - term3) / (n * (n - 3))
    return hsic_val

def compute_hsic(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Computes the empirical Hilbert-Schmidt Independence Criterion (HSIC) 
    between two feature matrices X and Y using a Gaussian kernel with 
    the dimension-wise median heuristic.
    
    Args:
        X: Feature matrix for Agent A, shape (N, d)
        Y: Feature matrix for Agent B, shape (N, d)
        
    Returns:
        float: The HSIC value
    """
    n_samples = X.shape[0]
    
    if n_samples <= 1:
        return 0.0
        
    # 1. Compute bandwidth vectors via dimension-wise median heuristic
    sigma_X = compute_dimension_wise_median_heuristic(X)
    sigma_Y = compute_dimension_wise_median_heuristic(Y)
    
    # 2. Compute Kernel Matrices
    K_X = compute_gaussian_kernel_variable_bandwidth(X, sigma_X)
    K_Y = compute_gaussian_kernel_variable_bandwidth(Y, sigma_Y)
    
    # 3. Compute Unbiased HSIC score
    return compute_unbiased_hsic(K_X, K_Y)

def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Computes the Centered Kernel Alignment (CKA) between two matrices.
    
    CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
    
    Returns a score between 0 (independent) and 1 (identical).
    """
    hsic_xy = compute_hsic(X, Y)
    hsic_xx = max(compute_hsic(X, X), 0.0)
    hsic_yy = max(compute_hsic(Y, Y), 0.0)
    
    denominator = np.sqrt(hsic_xx * hsic_yy)
    if denominator == 0.0:
        return 0.0
    # end if
        
    cka_val = hsic_xy / denominator
    
    # Clip CKA to [-1, 1] for sanity, though standard CKA is [0, 1]
    return float(np.clip(cka_val, -1.0, 1.0))

