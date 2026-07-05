"""Pinball loss across quantile levels (Figure 1 of the paper).

Trains one NN per quantile level on the pairwise synthetic data, repeats over
several seeds, and plots mean +/- std of train and test pinball losses.

Usage:
    python experiments/synthetic_pinball_curve.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from pairwise_qr import generate_pairs, pinball_loss_np, train_nn

# ---- Configuration ----
N_TRAIN = 500          # i.i.d. samples -> n(n-1)/2 training pairs
N_TEST = 100
N_EPOCHS = 100
N_REPEATS = 10
QUANTILES = np.round(np.linspace(0, 1, 21)[1:-1], 2)  # 0.05 ... 0.95
SEEDS = list(range(42, 42 + N_REPEATS))
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

# ---- Experiment ----
train_err = {q: [] for q in QUANTILES}
test_err = {q: [] for q in QUANTILES}

for rep, seed in enumerate(SEEDS, 1):
    print(f"Repeat {rep}/{N_REPEATS} (seed {seed})")
    X_tr, y_tr = generate_pairs(N_TRAIN, rng=seed)
    X_te, y_te = generate_pairs(N_TEST, rng=seed + 1000)
    for tau in QUANTILES:
        model, _ = train_nn(X_tr, y_tr, tau, n_epochs=N_EPOCHS, seed=seed)
        train_err[tau].append(pinball_loss_np(y_tr, model.predict(X_tr), tau))
        test_err[tau].append(pinball_loss_np(y_te, model.predict(X_te), tau))

# ---- Plot ----
def mean_std(d):
    return (np.array([np.mean(d[q]) for q in QUANTILES]),
            np.array([np.std(d[q]) for q in QUANTILES]))

m_tr, s_tr = mean_std(train_err)
m_te, s_te = mean_std(test_err)

plt.figure(figsize=(12, 6))
plt.plot(QUANTILES, m_tr, marker="o", label="Train Error")
plt.fill_between(QUANTILES, m_tr - s_tr, m_tr + s_tr, color="blue", alpha=0.2)
plt.plot(QUANTILES, m_te, marker="o", label="Test Error")
plt.fill_between(QUANTILES, m_te - s_te, m_te + s_te, color="orange", alpha=0.2)
plt.xlabel("Quantiles")
plt.ylabel("Pinball Loss")
plt.title("Pinball Loss Across Quantiles")
plt.legend()
plt.savefig(OUT / "fig1_pinball_loss_across_quantiles.png", dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved {OUT / 'fig1_pinball_loss_across_quantiles.png'}")
