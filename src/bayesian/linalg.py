"""Small pure-stdlib linear algebra helpers.

Matrices are ``list[list[float]]`` and vectors ``list[float]``. The platform
only ever needs tiny dense systems (the feature count is single digits), so
straightforward O(n^3) routines are the honest choice — no hidden dependency,
every operation auditable in source.
"""
from __future__ import annotations

import math


def matvec(A: list[list[float]], x: list[float]) -> list[float]:
    return [sum(a * xi for a, xi in zip(row, x)) for row in A]


def transpose(A: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*A)]


def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    Bt = transpose(B)
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def add_diag(A: list[list[float]], value: float) -> list[list[float]]:
    out = [row[:] for row in A]
    for i in range(len(out)):
        out[i][i] += value
    return out


def solve(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve ``A x = b`` by Gauss-Jordan with partial pivoting."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            raise ValueError("singular matrix")
        M[col], M[pivot] = M[pivot], M[col]
        piv = M[col][col]
        M[col] = [v / piv for v in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                factor = M[r][col]
                M[r] = [v - factor * w for v, w in zip(M[r], M[col])]
    return [M[i][n] for i in range(n)]


def inverse(A: list[list[float]]) -> list[list[float]]:
    n = len(A)
    cols = []
    for j in range(n):
        e = [0.0] * n
        e[j] = 1.0
        cols.append(solve(A, e))
    return transpose(cols)


def cholesky(A: list[list[float]]) -> list[list[float]]:
    """Lower-triangular Cholesky factor L with A = L L^T."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                v = A[i][i] - s
                if v <= 0.0:
                    raise ValueError("matrix is not positive definite")
                L[i][j] = math.sqrt(v)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def logdet(A: list[list[float]]) -> float:
    L = cholesky(A)
    return 2.0 * sum(math.log(L[i][i]) for i in range(len(L)))


def multivariate_normal_sample(
    mean: list[float], cov: list[list[float]], rng, n: int
) -> list[list[float]]:
    """Draw ``n`` samples from N(mean, cov) using the Cholesky factor."""
    L = cholesky(cov)
    d = len(mean)
    out = []
    for _ in range(n):
        z = [rng.gauss(0.0, 1.0) for _ in range(d)]
        out.append([mean[i] + sum(L[i][k] * z[k] for k in range(i + 1)) for i in range(d)])
    return out


def kl_gaussian(
    mean0: list[float],
    cov0: list[list[float]],
    mean1: list[float],
    cov1: list[list[float]],
) -> float:
    """KL(N0 || N1) for two multivariate Gaussians, closed form."""
    d = len(mean0)
    inv1 = inverse(cov1)
    diff = [a - b for a, b in zip(mean0, mean1)]
    trace_term = sum(inv1[i][j] * cov0[j][i] for i in range(d) for j in range(d))
    quad = sum(diff[i] * sum(inv1[i][j] * diff[j] for j in range(d)) for i in range(d))
    return 0.5 * (trace_term + quad - d + logdet(cov1) - logdet(cov0))
