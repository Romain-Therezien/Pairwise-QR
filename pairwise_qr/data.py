"""Synthetic data-generating process and pair construction.

DGP (Section 4.1 of the paper):
    Z   ~ U(-z_range, z_range)
    X   = Z + mu_Z + sigma_Z * eps,   eps ~ N(0, 1)
    mu_Z    = 0.1 * Z
    sigma_Z = 0.3 * |Z|
    s(X, X') = sin(X + X')

The pairwise sample is built from all n(n-1)/2 pairs (i, j), i < j, so that
the empirical pinball risk is the complete U-statistic of Eq. (12). Use
``subsample_pairs`` for the incomplete U-statistic variant (Eq. (8)).

Note on ``z_range``: the paper's text states Z ~ U(-1, 1); the published
figures were generated with z_range = 1.5. The default here matches the
figures. Set z_range = 1.0 to match the text exactly.
"""

from __future__ import annotations

import numpy as np

MU_COEF = 0.1
SIGMA_COEF = 0.3


def _dgp_params(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Conditional mean shift and standard deviation of X given Z."""
    return MU_COEF * Z, SIGMA_COEF * np.abs(Z)


def generate_pairs(
    n_samples: int,
    z_range: float = 1.5,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw n i.i.d. samples and return all n(n-1)/2 pairs.

    Returns
    -------
    pair_Z : (n_pairs, 2) float32 array of covariate pairs (Z_i, Z_j)
    pair_s : (n_pairs,) float32 array of scores s(X_i, X_j) = sin(X_i + X_j)
    """
    rng = np.random.default_rng(rng)
    Z = rng.uniform(-z_range, z_range, n_samples)
    mu, sigma = _dgp_params(Z)
    X = Z + mu + sigma * rng.standard_normal(n_samples)

    i, j = np.triu_indices(n_samples, k=1)
    pair_Z = np.stack([Z[i], Z[j]], axis=1).astype(np.float32)
    pair_s = np.sin(X[i] + X[j]).astype(np.float32)
    return pair_Z, pair_s


def subsample_pairs(
    pair_Z: np.ndarray,
    pair_s: np.ndarray,
    n_pairs: int | None = None,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Incomplete U-statistic: draw B pairs with replacement (Blom, 1976).

    Defaults to B = n log n pairs, where n is recovered from the complete
    pair count n(n-1)/2, matching the experiment of Table 2 in the paper.
    """
    rng = np.random.default_rng(rng)
    total = len(pair_s)
    if n_pairs is None:
        # invert total = n(n-1)/2 to recover n, then B = n log n
        n = int(round((1 + np.sqrt(1 + 8 * total)) / 2))
        n_pairs = int(n * np.log(n))
    idx = rng.choice(total, size=n_pairs, replace=True)
    return pair_Z[idx], pair_s[idx]
