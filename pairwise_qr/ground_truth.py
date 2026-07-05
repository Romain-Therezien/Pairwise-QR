"""Monte-Carlo estimation of the true conditional quantile Q_s(tau | Z, Z').

The analytical form of Q_s(tau | Z, Z') is intractable for the synthetic DGP,
so it is estimated by drawing M realizations of (X, X') given each covariate
pair and taking the empirical tau-quantile (Eq. (25) of the paper). Model
accuracy is then measured as MAE against these estimates.
"""

from __future__ import annotations

import numpy as np

from .data import _dgp_params


def mc_true_quantiles(
    pair_Z: np.ndarray,
    taus,
    n_mc: int = 5000,
    chunk_size: int = 250,
    rng: np.random.Generator | int | None = 42,
) -> np.ndarray:
    """Estimate Q_s(tau | Z, Z') by Monte Carlo, chunked over draws.

    Parameters
    ----------
    pair_Z : (n_pairs, 2) array of covariate pairs
    taus : scalar or sequence of quantile levels
    n_mc : number of MC draws per pair
    chunk_size : MC draws processed at a time (bounds memory at
        n_pairs * chunk_size floats)

    Returns
    -------
    (len(taus), n_pairs) array of estimated conditional quantiles
    """
    rng = np.random.default_rng(rng)
    taus = np.atleast_1d(taus)
    n_pairs = pair_Z.shape[0]

    Z_i = pair_Z[:, 0][:, None]
    Z_j = pair_Z[:, 1][:, None]
    mu_i, sigma_i = _dgp_params(Z_i)
    mu_j, sigma_j = _dgp_params(Z_j)

    samples = np.empty((n_pairs, n_mc), dtype=np.float32)
    for start in range(0, n_mc, chunk_size):
        end = min(start + chunk_size, n_mc)
        m = end - start
        eps_i = rng.standard_normal((n_pairs, m)).astype(np.float32)
        eps_j = rng.standard_normal((n_pairs, m)).astype(np.float32)
        X_i = Z_i + mu_i + sigma_i * eps_i
        X_j = Z_j + mu_j + sigma_j * eps_j
        samples[:, start:end] = np.sin(X_i + X_j)

    return np.stack([np.quantile(samples, t, axis=1) for t in taus])
