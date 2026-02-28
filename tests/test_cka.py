import numpy as np
from agentic_ai_analysis.cka.metrics import compute_cka, compute_hsic, compute_dimension_wise_median_heuristic

def test_cka_identical():
    # Two identical matrices should have CKA = 1.0
    N, d = 20, 10
    X = np.random.randn(N, d)
    cka_score = compute_cka(X, X)
    print(f"Identical matrices CKA: {cka_score}")
    assert np.isclose(cka_score, 1.0, atol=1e-5), f"Expected 1.0, got {cka_score}"

def test_cka_orthogonal():
    # Two independent/orthogonal (random) matrices should have near-zero CKA
    N, d = 200, 10
    X = np.random.randn(N, d)
    Y = np.random.randn(N, d)
    cka_score = compute_cka(X, Y)
    print(f"Independent random matrices CKA: {cka_score}")
    assert cka_score < 0.1, f"Expected near 0, got {cka_score}"

def test_cka_correlated():
    # Highly correlated but not identical
    N, d = 50, 10
    X = np.random.randn(N, d)
    Y = X + np.random.randn(N, d) * 0.1
    cka_score = compute_cka(X, Y)
    print(f"Correlated matrices CKA: {cka_score}")
    assert cka_score > 0.8, f"Expected high correlation, got {cka_score}"

def test_median_heuristic():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    # Diff pairs: (1,2)->1, (1,3)->2, (1,4)->3, (2,3)->1, (2,4)->2, (3,4)->1
    # Diffs: [1, 2, 3, 1, 2, 1] -> Sorted: [1, 1, 1, 2, 2, 3]
    # Median is 1.5
    bw = compute_dimension_wise_median_heuristic(X)
    print(f"Median heuristic bandwidth: {bw}")
    assert np.isclose(bw[0], 1.5)

if __name__ == "__main__":
    test_cka_identical()
    test_cka_orthogonal()
    test_cka_correlated()
    test_median_heuristic()
    print("All CKA tests passed!")
