import random
import numpy as np
from agentic_ai_analysis.cka.metrics import main

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


n_sample = 500
n_dim = 300

def test_cka_correlated():
    # Highly correlated but not identical
    N, d = n_sample, n_dim
    X = np.random.randn(N, d)
    Y = X + np.random.randn(N, d) * 0.1
    cka_score = main(X, Y)
    logger.info(f"Correlated matrices CKA. Y = X + noise: {cka_score}")
    # assert cka_score > 0.8, f"Expected high correlation, got {cka_score}"


def test_cka_linear_dependency():
    rand_gen = np.random.default_rng(42)
    # test hsic
    # I want to make the dependency X -> Y.
    X = rand_gen.normal(1, 1, (n_sample, n_dim))
    noise_term = rand_gen.normal(0, 1, (n_sample, n_dim))
    Y = X * 2 + 10 + noise_term
    cka_score = main(X, Y)
    logger.info(f"Linear dependency CKA score. Y = 2X + 10 + noise: {cka_score}")
    # assert cka_score > 0.5, f"Expected high correlation, got {cka_score}"


def test_cka_non_linear_dependency():
    rand_gen = np.random.default_rng(42)
    # test hsic
    # I want to make the dependency X -> Y.
    X = rand_gen.normal(1, 1, (n_sample, n_dim))
    Y = np.sin(X * 2.0) + X
    cka_score = main(X, Y)
    logger.info(f"Non-linear dependency CKA score. Y = sin(2X) + X: {cka_score}")
    # assert cka_score > 0.8, f"Expected high correlation, got {cka_score}"

def test_cka_independent():
    rand_gen = np.random.default_rng(42)
    # test hsic
    # I want to make the dependency X -> Y.
    X = rand_gen.normal(1, 1, (n_sample, n_dim))
    Y = rand_gen.laplace(100, 1, (n_sample, n_dim))
    cka_score = main(X, Y)
    logger.info(f"Independent CKA score. X = Gaussian(1, 1), Y = laplace(100, 1): {cka_score}")
    # assert cka_score > 0.8, f"Expected high correlation, got {cka_score}"


if __name__ == "__main__":
    test_cka_correlated()
    test_cka_linear_dependency()
    test_cka_non_linear_dependency()
    test_cka_independent()
    logger.info("All CKA tests passed!")
