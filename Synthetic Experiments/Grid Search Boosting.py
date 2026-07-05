# %%
import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import ParameterGrid

# -----------------------------
# Data generation
# -----------------------------
def generate_data(n_samples):
    Z = np.random.uniform(-1.5, 1.5, n_samples)

    mu = 0.1 * Z
    sigma = 0.3 * np.abs(Z)

    X = Z + mu + sigma * np.random.normal(0, 1, len(Z))
    X = torch.tensor(X, dtype=torch.float32)

    score = torch.sin(X[:, None] + X[None, :])

    triu = torch.triu_indices(n_samples, n_samples, offset=1)

    pair_Z = np.stack([Z[triu[0]], Z[triu[1]]], axis=1)
    pair_scores = score[triu[0], triu[1]]

    return pair_Z.astype(np.float32), pair_scores.numpy()


# -----------------------------
# Shared noise (PAIRWISE FIX)
# -----------------------------
def build_shared_noise(n_pairs, n_mc=1000, seed=42):
    rng = np.random.default_rng(seed)
    eps_i = rng.normal(size=(n_pairs, n_mc)).astype(np.float32)
    eps_j = rng.normal(size=(n_pairs, n_mc)).astype(np.float32)
    return eps_i, eps_j


# -----------------------------
# True quantile computation
# -----------------------------
def compute_true_quantile_chunked(Z, quantiles, n_mc=5000, chunk_size=250):
    rng = np.random.default_rng(42)

    quantiles = np.atleast_1d(quantiles)
    n_pairs = Z.shape[0]

    samples = np.empty((n_pairs, n_mc), dtype=np.float32)

    Z_i = Z[:, 0][:, None]
    Z_j = Z[:, 1][:, None]

    mu_i = 0.1 * Z_i
    mu_j = 0.1 * Z_j

    sigma_i = 0.3 * np.abs(Z_i)
    sigma_j = 0.3 * np.abs(Z_j)

    for start in range(0, n_mc, chunk_size):
        end = min(start + chunk_size, n_mc)
        m = end - start

        eps_i = rng.normal(size=(n_pairs, m)).astype(np.float32)
        eps_j = rng.normal(size=(n_pairs, m)).astype(np.float32)

        X_i = Z_i + mu_i + sigma_i * eps_i
        X_j = Z_j + mu_j + sigma_j * eps_j

        samples[:, start:end] = np.sin(X_i + X_j)

    return np.stack([
        np.quantile(samples, q, axis=1)
        for q in quantiles
    ])

# %%
# -----------------------------
# Setup
# -----------------------------
quantiles = [0.2, 0.5, 0.8]  # np.linspace(0.05, 0.95, 19)

param_grid = {
    "n_estimators": [300, 600],
    "learning_rate": [0.01, 0.05],
    "max_depth": [3, 5, 6],
    "min_samples_leaf": [5, 10, 20],
    "subsample": [0.7, 0.9]
}

n_repeats = 5
n_train = 1000
n_val = 200

best_score = np.inf
best_params = None

# -----------------------------
# Fixed validation set
# -----------------------------
X_val, y_val = generate_data(n_val)
X_val_np = X_val

n_pairs_val = X_val_np.shape[0]

# %%
eps_i, eps_j = build_shared_noise(n_pairs_val, n_mc=5000, seed=42)

true_q_val = compute_true_quantile_chunked(
    X_val_np,
    quantiles,
    n_mc=5000,
    chunk_size=250
)

# %%

# -----------------------------
# Training loop
# -----------------------------
for params in ParameterGrid(param_grid):

    repeat_scores = []

    for seed in range(n_repeats):

        np.random.seed(seed)
        torch.manual_seed(seed)

        X_train, y_train = generate_data(n_train)

        quantile_errors = []

        for q_idx, q in enumerate(quantiles):

            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                random_state=seed,
                **params
            )

            model.fit(X_train, y_train)
            pred = model.predict(X_val_np)

            err = np.mean(np.abs(true_q_val[q_idx] - pred))
            quantile_errors.append(err)

        repeat_scores.append(np.mean(quantile_errors))
        print(f"Params: {params}, Seed: {seed}, Score: {repeat_scores[-1]:.4f}")
    score = np.mean(repeat_scores)

    if score < best_score:
        best_score = score
        best_params = params


print("Best params:", best_params)
print("Best score:", best_score)
# %%
