import random
import numpy as np
from agentic_ai_analysis.cka.metrics import compute_cka, get_median_scale

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

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


def test_cka_linear_dependency():
    rand_gen = np.random.default_rng(42)
    # test hsic
    # I want to make the dependency X -> Y.
    X = rand_gen.normal(1, 1, (100, 10))
    noise_term = rand_gen.normal(0, 1, (100, 10))
    Y = X * 2 + 10 + noise_term
    cka_score = compute_cka(X, Y)
    logger.info(f"HSIC score: {cka_score}")
    assert cka_score > 0.5, f"Expected high correlation, got {cka_score}"


def test_cka_non_linear_dependency():
    rand_gen = np.random.default_rng(42)
    # test hsic
    # I want to make the dependency X -> Y.
    X = rand_gen.normal(1, 1, (100, 10))
    Y = np.exp(X * 2)
    cka_score = compute_cka(X, Y)
    logger.info(f"Non-linear dependency CKA score: {cka_score}")
    # assert cka_score > 0.8, f"Expected high correlation, got {cka_score}"


if __name__ == "__main__":
    test_cka_identical()
    test_cka_orthogonal()
    test_cka_correlated()
    test_cka_linear_dependency()
    test_cka_non_linear_dependency()
    logger.info("All CKA tests passed!")
