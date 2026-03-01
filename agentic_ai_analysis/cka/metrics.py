import numpy as np
import torch
from mmd_tst_variable_detector import QuadraticKernelGaussianKernel

def get_median_scale(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the median heuristic for the RBF kernel bandwidth (sigma).
    sigma = median(||x_i - x_j||)
    """
    if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
        x = x.to(torch.float32)
    # end

    if torch.cuda.is_available():
        x = x.to(device='cuda')
    # end

    # Efficient pairwise distance calculation
    # x shape: (N, D)
    # pdist returns the upper triangle of the distance matrix flattened
    # dists = torch.nn.functional.pdist(x, p=2)
    # median_dist = torch.median(dists)
    
    # # Return float, ensuring it's not zero to avoid division by zero
    # return max(float(median_dist.item()), 1e-6)

    kernel_obj = QuadraticKernelGaussianKernel(ard_weights=torch.ones(x.shape[-1]))
    kernel_obj.to(x.device)

    tensor_length_scale = kernel_obj._get_median_dim(x, x, is_safe_guard_same_xy=False)
    assert tensor_length_scale is not None

    return tensor_length_scale

def hsic_centered(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Computes the Hilbert-Schmidt Independence Criterion (HSIC) 
    using centered Gram matrices.
    HSIC(K, L) = tr(K_c * L_c) / (n-1)^2
    """
    n = K.shape[0]
    # Centering matrix H = I - 1/n * J
    H = torch.eye(n, device=K.device) - (1.0 / n)
    
    # Center the kernel matrices
    Kc = H @ K @ H
    Lc = H @ L @ H
    
    # HSIC calculation
    return torch.trace(Kc @ Lc) / ((n - 1) ** 2)


def compute_rbf_kernel(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Computes the ARD Gaussian RBF kernel matrix using a dimension-wise length scale vector.
    K_ij = exp(-0.5 * || (x_i - x_j) / sigma ||^2)
    """
    # 1. Cast to float32 for CPU compatibility and numerical stability
    if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
        x = x.to(torch.float32)
    # end
    
    if sigma.dtype in [torch.float16, torch.bfloat16]:
        sigma = sigma.to(torch.float32)
    # end

    if torch.cuda.is_available():
        x = x.to(device='cuda')
        sigma = sigma.to(device='cuda')
    # end

    # 2. Safety guard: Prevent division by zero if any dimension's variance collapsed
    # This replaces the old max(val, 1e-6) logic for the vector
    sigma_safe = torch.clamp(sigma, min=1e-8)

    # 3. Geometric Scaling
    # Broadcast division: x is (N, D), sigma_safe is (D,) -> x_scaled is (N, D)
    x_scaled = x / sigma_safe

    # 4. Squared Pairwise distances on the scaled feature space
    # cdist computes full matrix (N, N)
    dist_sq = torch.cdist(x_scaled, x_scaled, p=2) ** 2
    
    # 5. Compute Kernel
    # Since we already divided by sigma inside the squared distance, 
    # we now only divide by 2.0.
    return torch.exp(-dist_sq / 2.0)

def get_cka_value(
    kernel_x_length_scale: torch.Tensor,
    kernel_y_length_scale: torch.Tensor,
    x: torch.Tensor, 
    y: torch.Tensor
) -> float:
    """
    Computes CKA similarity: HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
    """
    # 1. Compute Kernel Matrices (N x N)
    K = compute_rbf_kernel(x, kernel_x_length_scale)
    L = compute_rbf_kernel(y, kernel_y_length_scale)

    K = K.to(torch.float32)
    L = L.to(torch.float32)
    
    # 2. Compute HSIC values
    hsic_kl = hsic_centered(K, L)
    hsic_kk = hsic_centered(K, K)
    hsic_ll = hsic_centered(L, L)
    
    # 3. Normalize
    cka = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-8)

    val_cka = cka.cpu().item()
    return val_cka


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Computes the Centered Kernel Alignment (CKA) between two matrices.
    
    CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
    
    Returns a score between 0 (independent) and 1 (identical).
    """ 
    # cka_val = hsic_xy / denominator
    sigma_X = get_median_scale(torch.from_numpy(X))
    sigma_Y = get_median_scale(torch.from_numpy(Y))

    cka_val = get_cka_value(
        sigma_X,
        sigma_Y,
        torch.from_numpy(X),
        torch.from_numpy(Y)
    )

    # Clip CKA to [-1, 1] for sanity, though standard CKA is [0, 1]
    return float(np.clip(cka_val, -1.0, 1.0))

