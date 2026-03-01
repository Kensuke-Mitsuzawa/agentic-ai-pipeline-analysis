import abc
import numpy as np
import torch
import pydantic
from mmd_tst_variable_detector import QuadraticKernelGaussianKernel


class CKAResultContainer(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    cka: float
    hsic_xy: float
    hsic_xx: float
    hsic_yy: float
    kernel_x_length_scale: np.ndarray
    kernel_y_length_scale: np.ndarray
    kernel_L: np.ndarray
    kernel_K: np.ndarray

    def get_cka_value(self) -> float:
        return self.cka


class BaseCKAInterface(abc.ABC):
    @abc.abstractmethod
    def compute_cka(        
        self,
        x: torch.Tensor, 
        y: torch.Tensor
    ) -> CKAResultContainer:
        raise NotImplementedError


class CKAwithHsicMultiLengthScaleDimKernel(BaseCKAInterface):
    @staticmethod
    def get_median_scale(x: torch.Tensor) -> QuadraticKernelGaussianKernel:
        """
        Computes the median heuristic for the RBF kernel bandwidth (sigma).
        sigma = median(||x_i - x_j||)
        """
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.to(torch.float32)
        # end

        # if torch.cuda.is_available():
        #     x = x.to(device='cuda')
        # # end

        # Efficient pairwise distance calculation
        # x shape: (N, D)
        # pdist returns the upper triangle of the distance matrix flattened
        # dists = torch.nn.functional.pdist(x, p=2)
        # median_dist = torch.median(dists)
        
        # # Return float, ensuring it's not zero to avoid division by zero
        # return max(float(median_dist.item()), 1e-6)

        ard = torch.ones(x.shape[-1]).to(x.device)
        kernel_obj = QuadraticKernelGaussianKernel(ard_weights=ard)
        # kernel_obj.to(x.device)

        tensor_length_scale = kernel_obj._get_median_dim(x, x, is_safe_guard_same_xy=False)
        tensor_length_scale = tensor_length_scale.to(dtype=torch.float32, device=x.device)
        kernel_obj.bandwidth = torch.nn.Parameter(tensor_length_scale, requires_grad=False)

        return kernel_obj

    @staticmethod
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

    @staticmethod
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
        self,
        kernel_func_x: QuadraticKernelGaussianKernel,
        kernel_func_y: QuadraticKernelGaussianKernel,
        x: torch.Tensor, 
        y: torch.Tensor,
    ) -> CKAResultContainer:
        """
        Computes CKA similarity: HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
        """
        # 1. Compute Kernel Matrices (N x N)

        container_K = kernel_func_x._compute_kernel_matrix_dim(x, x)
        container_L = kernel_func_y._compute_kernel_matrix_dim(y, y)

        K = container_K.kernel_matrix_container.k_xy
        L = container_L.kernel_matrix_container.k_xy

        K = K.to(torch.float32)
        L = L.to(torch.float32)
        
        # 2. Compute HSIC values
        hsic_kl = self.hsic_centered(K, L)
        hsic_kk = self.hsic_centered(K, K)
        hsic_ll = self.hsic_centered(L, L)
        
        # 3. Normalize
        cka = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-8)

        val_cka = cka.cpu().item()

        # Clip CKA to [-1, 1] for sanity, though standard CKA is [0, 1]
        cka_final = float(np.clip(val_cka, -1.0, 1.0))

        return CKAResultContainer(
            cka=cka_final,
            hsic_xy=hsic_kl.cpu().item(),
            hsic_xx=hsic_kk.cpu().item(),
            hsic_yy=hsic_ll.cpu().item(),
            kernel_x_length_scale=kernel_func_x.bandwidth.detach().cpu().numpy(),
            kernel_y_length_scale=kernel_func_y.bandwidth.detach().cpu().numpy(),
            kernel_L=L.detach().cpu().numpy(),
            kernel_K=K.detach().cpu().numpy()
        )

    def compute_cka(self, x: torch.Tensor, y: torch.Tensor) -> CKAResultContainer:
        """
        Computes the Centered Kernel Alignment (CKA) between two matrices.
        
        CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
        
        Returns a score between 0 (independent) and 1 (identical).
        """ 

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        y = y.to(device)

        kernel_func_x = self.get_median_scale(x)
        kernel_func_y = self.get_median_scale(y)

        cka_container = self.get_cka_value(
            kernel_func_x,
            kernel_func_y,
            x,
            y,
        )

        return cka_container



class CKAwithHsicSingleLengthScale(BaseCKAInterface):
    @staticmethod
    def _compute_median_heuristic(tensor_data: torch.Tensor, 
                                max_samples: int = 5000) -> float:
        """
        Computes median pairwise distance on a tensor (N, Features).
        """
        # 1. Subsample if data is too large for O(N^2) distance matrix
        n_points = tensor_data.shape[0]
        if n_points > max_samples:
            indices = torch.randperm(n_points)[:max_samples]
            data_subset = tensor_data[indices]
        else:
            data_subset = tensor_data

        # 2. Compute Pairwise Distance (Condensed 1D vector)
        # p=2 is Euclidean distance
        dists = torch.nn.functional.pdist(data_subset, p=2)
        
        # 3. Median
        median_dist = torch.median(dists)
        
        # Safety: Return 1.0 if median is 0 (e.g. all points identical)
        return float(median_dist.item()) if median_dist.item() > 1e-6 else 1.0

    # ----- Core Logic: HSIC with Dual Sigma -----

    @staticmethod
    def kernel_x(sigma_x: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Standard Gaussian: exp(- ||x-y||^2 / (2 * sigma_x^2))
        dist_sq = torch.cdist(x, y, p=2) ** 2
        gamma = 1.0 / (2.0 * (sigma_x ** 2))
        return torch.exp(-gamma * dist_sq)

    @staticmethod
    def kernel_y(sigma_y: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Standard Gaussian: exp(- ||x-y||^2 / (2 * sigma_y^2))
        dist_sq = torch.cdist(x, y, p=2) ** 2
        gamma = 1.0 / (2.0 * (sigma_y ** 2))
        return torch.exp(-gamma * dist_sq)

    def compute_cka(
        self,
        x: torch.Tensor, 
        y: torch.Tensor) -> CKAResultContainer:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        y = y.to(device)

        n_samples = x.shape[0]

        sigma_x = self._compute_median_heuristic(tensor_data=x)
        sigma_y = self._compute_median_heuristic(tensor_data=y)
        
        K = self.kernel_x(sigma_x, x, x)
        L = self.kernel_y(sigma_y, y, y)
        
        # 2. Centering Matrix H
        H = torch.eye(n_samples, device=device) - (1.0 / n_samples) * torch.ones((n_samples, n_samples), device=device)
        
        K = K.to(torch.float32)
        L = L.to(torch.float32)
        H = H.to(torch.float32)

        # 3. Centered Kernels
        Kc = torch.mm(torch.mm(H, K), H)
        Lc = torch.mm(torch.mm(H, L), H)
        
        # 4. HSIC Value
        # Trace(Kc @ Lc) = Sum(Kc * Lc)
        n_sq = (n_samples - 1) ** 2
        hsic_xy = torch.sum(Kc * Lc) / n_sq
        hsic_xx = torch.sum(Kc * Kc) / n_sq
        hsic_yy = torch.sum(Lc * Lc) / n_sq
        
        # 5. CKA Normalization (The missing step!)
        cka_val = hsic_xy / torch.sqrt(hsic_xx * hsic_yy)        
        return CKAResultContainer(
            cka=cka_val.cpu().item(),
            hsic_xy=hsic_xy.cpu().item(),
            hsic_xx=hsic_xx.cpu().item(),
            hsic_yy=hsic_yy.cpu().item(),
            kernel_x_length_scale=np.array([sigma_x]),
            kernel_y_length_scale=np.array([sigma_y]),
            kernel_L=L.detach().cpu().numpy(),
            kernel_K=K.detach().cpu().numpy()
        )


class CKALinear(BaseCKAInterface):
    def compute_cka(self, x: torch.Tensor, y: torch.Tensor) -> CKAResultContainer:
        """
        Computes Linear CKA. 
        x and y should be shape (N, d_x) and (N, d_y).
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        y = y.to(device)
        
        n_samples = x.shape[0]

        # 1. Linear Kernels (Dot Product)
        K = torch.mm(x, x.t())
        L = torch.mm(y, y.t())
        
        # 2. Centering Matrix H
        H = torch.eye(n_samples, device=device) - (1.0 / n_samples) * torch.ones((n_samples, n_samples), device=device)
        
        K = K.to(torch.float32)
        L = L.to(torch.float32)
        H = H.to(torch.float32)

        # 3. Centered Kernels
        Kc = torch.mm(torch.mm(H, K), H)
        Lc = torch.mm(torch.mm(H, L), H)
        
        # 4. HSIC Values
        n_sq = (n_samples - 1) ** 2
        hsic_xy = torch.sum(Kc * Lc) / n_sq
        hsic_xx = torch.sum(Kc * Kc) / n_sq
        hsic_yy = torch.sum(Lc * Lc) / n_sq
        
        # 5. CKA Normalization
        cka_val = hsic_xy / torch.sqrt(hsic_xx * hsic_yy)
        
        res_obj = CKAResultContainer(
            cka=cka_val.cpu().item(),
            hsic_xy=hsic_xy.cpu().item(),
            hsic_xx=hsic_xx.cpu().item(),
            hsic_yy=hsic_yy.cpu().item(),
            kernel_x_length_scale=np.array([1.0]),
            kernel_y_length_scale=np.array([1.0]),
            kernel_L=L.detach().cpu().numpy(),
            kernel_K=K.detach().cpu().numpy()
        )

        return res_obj

        

def main(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Computes the Centered Kernel Alignment (CKA) between two matrices.
    
    CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
    
    Returns a score between 0 (independent) and 1 (identical).
    """ 
    # cka_obj = CKAwithHsicMultiLengthScaleDimKernel()
    # cka_obj = CKAwithHsicSingleLengthScale()
    cka_obj = CKALinear()
    cka_container = cka_obj.compute_cka(torch.from_numpy(X), torch.from_numpy(Y))
    return cka_container.cka
