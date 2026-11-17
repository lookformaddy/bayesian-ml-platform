import math

import pytest

from bayesian.linalg import (
    cholesky,
    inverse,
    kl_gaussian,
    logdet,
    matmul,
    multivariate_normal_sample,
    solve,
    transpose,
)


def test_solve_small_system():
    A = [[2.0, 1.0], [1.0, 3.0]]
    b = [5.0, 6.0]
    x = solve(A, b)
    assert abs(x[0] - 1.8) < 1e-9
    assert abs(x[1] - 1.4) < 1e-9


def test_solve_singular_raises():
    with pytest.raises(ValueError):
        solve([[1.0, 1.0], [1.0, 1.0]], [1.0, 2.0])


def test_inverse_roundtrip():
    A = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]
    I = matmul(A, inverse(A))
    for i in range(3):
        for j in range(3):
            assert abs(I[i][j] - (1.0 if i == j else 0.0)) < 1e-9


def test_cholesky_reconstructs():
    A = [[4.0, 2.0], [2.0, 3.0]]
    L = cholesky(A)
    assert abs(matmul(L, transpose(L))[0][1] - 2.0) < 1e-9
    assert abs(logdet(A) - math.log(8.0)) < 1e-9


def test_cholesky_non_pd_raises():
    with pytest.raises(ValueError):
        cholesky([[1.0, 2.0], [2.0, 1.0]])


def test_multivariate_normal_sample_mean():
    import random

    rng = random.Random(0)
    samples = multivariate_normal_sample(
        [1.0, -1.0], [[1.0, 0.0], [0.0, 4.0]], rng, 2000
    )
    assert len(samples) == 2000
    m0 = sum(s[0] for s in samples) / 2000
    m1 = sum(s[1] for s in samples) / 2000
    assert abs(m0 - 1.0) < 0.1
    assert abs(m1 + 1.0) < 0.2


def test_kl_gaussian_identical_is_zero():
    mean = [0.5, -0.5]
    cov = [[1.0, 0.2], [0.2, 1.0]]
    assert abs(kl_gaussian(mean, cov, mean, cov)) < 1e-9


def test_kl_gaussian_shifted_positive():
    mean0 = [0.0, 0.0]
    mean1 = [2.0, 0.0]
    cov = [[1.0, 0.0], [0.0, 1.0]]
    kl = kl_gaussian(mean0, cov, mean1, cov)
    assert abs(kl - 2.0) < 1e-9  # 0.5 * ||diff||^2
