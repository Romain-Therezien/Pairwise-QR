"""Conditional quantile surfaces at tau = 0.1, 0.5, 0.9 (Figure 2).

Usage:
    python experiments/synthetic_surfaces.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from pairwise_qr import generate_pairs, train_nn

# ---- Configuration ----
N_TRAIN = 1000
N_TEST = 500
N_PLOT = 5000
N_EPOCHS = 100
SEED = 42
TAUS = [0.1, 0.5, 0.9]
COLORS = ["blue", "orange", "green"]
ALPHAS = [0.3, 0.4, 0.5]
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

# ---- Train one model per quantile ----
X_tr, y_tr = generate_pairs(N_TRAIN, rng=SEED)
X_te, y_te = generate_pairs(N_TEST, rng=SEED + 1000)

models = {}
for tau in TAUS:
    print(f"Training tau = {tau}")
    models[tau], _ = train_nn(X_tr, y_tr, tau, n_epochs=N_EPOCHS, seed=SEED)

# ---- Surface grid ----
z = np.linspace(X_tr[:, 0].min(), X_tr[:, 0].max(), 50)
zp = np.linspace(X_tr[:, 1].min(), X_tr[:, 1].max(), 50)
Zg, Zpg = np.meshgrid(z, zp)
grid = np.c_[Zg.ravel(), Zpg.ravel()].astype(np.float32)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
id_xs = np.random.choice(len(X_te), size=N_PLOT, replace=False)
ax.scatter(X_te[id_xs, 0], X_te[id_xs, 1], y_te[id_xs], alpha=0.4, s=10, color="yellow", edgecolor="k")
for tau, c, a in zip(TAUS, COLORS, ALPHAS):
    S = models[tau].predict(grid).reshape(Zg.shape)
    ax.plot_surface(Zg, Zpg, S, alpha=a, color=c)
ax.set_xlabel("Z")
ax.set_ylabel("Z'")
ax.set_zlabel("S")
ax.view_init(elev=30, azim=-50)
plt.title("Neural Network Quantile Regression Surfaces")
handles = [Patch(facecolor="yellow", alpha=0.4, edgecolor="k", label="Scores")] + [
    Patch(facecolor=c, alpha=a, label=f"Quantile {t}") for t, c, a in zip(TAUS, COLORS, ALPHAS)
]
ax.legend(handles=handles)
plt.savefig(OUT / "fig2_quantile_surfaces.pdf", dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved {OUT / 'fig2_quantile_surfaces.pdf'}")
