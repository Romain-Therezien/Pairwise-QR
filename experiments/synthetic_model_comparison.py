"""Model comparison: NN vs LightGBM vs Gradient Boosting (Figure 6).

Accuracy is the MAE against the Monte-Carlo ground-truth conditional quantile
(Eq. (25)), over 10 seeds, with 95% CIs.

Usage:
    python experiments/synthetic_model_comparison.py
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from pairwise_qr import generate_pairs, make_gbm, make_lightgbm, mc_true_quantiles, train_nn
from pairwise_qr.metrics import mae_vs_truth

# ---- Configuration ----
N_TRAIN = 1000
N_VAL = 200
N_EPOCHS = 100
QUANTILES = np.linspace(0.05, 0.95, 19)
SEEDS = list(range(42, 52))
N_MC = 5000
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

# ---- Fixed validation set and MC ground truth ----
X_val, y_val = generate_pairs(N_VAL, rng=42)
print(f"Computing MC ground truth on {len(y_val)} validation pairs...")
true_q = mc_true_quantiles(X_val, QUANTILES, n_mc=N_MC, rng=42)

# ---- Training loop ----
results = {}  # (model_name, tau, seed) -> MAE
for seed in SEEDS:
    X_tr, y_tr = generate_pairs(N_TRAIN, rng=seed)
    for qi, tau in enumerate(QUANTILES):
        for name in ["GradientBoosting", "LightGBM", "NeuralNet"]:
            t0 = time.time()
            if name == "NeuralNet":
                model, _ = train_nn(X_tr, y_tr, tau, n_epochs=N_EPOCHS, seed=seed)
                preds = model.predict(X_val)
            elif name == "LightGBM":
                model = make_lightgbm(tau, seed=seed)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
            else:
                model = make_gbm(tau, seed=seed)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
            err = mae_vs_truth(true_q[qi], preds)
            results[(name, round(tau, 2), seed)] = err
            print(f"seed {seed} | {name:16s} tau={tau:.2f} MAE={err:.4f} ({time.time() - t0:.1f}s)")

# ---- Aggregate and plot ----
by_model = defaultdict(lambda: defaultdict(list))
for (name, tau, _), err in results.items():
    by_model[name][tau].append(err)

plt.figure(figsize=(8, 5))
for name, per_tau in by_model.items():
    taus = sorted(per_tau)
    mean = [np.mean(per_tau[t]) for t in taus]
    lo = [np.percentile(per_tau[t], 2.5) for t in taus]
    hi = [np.percentile(per_tau[t], 97.5) for t in taus]
    plt.plot(taus, mean, marker="o", label=name)
    plt.fill_between(taus, lo, hi, alpha=0.2)
plt.xlabel("Quantile")
plt.ylabel("Mean Absolute Error")
plt.title("Mean Absolute Error by Model and Quantile with 95% CI")
plt.legend()
plt.grid(True)
plt.savefig(OUT / "fig6_model_comparison_mae.png", dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved {OUT / 'fig6_model_comparison_mae.png'}")
