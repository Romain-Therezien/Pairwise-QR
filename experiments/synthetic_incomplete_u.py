"""Complete vs incomplete U-statistics: runtime / accuracy trade-off (Table 2).

The complete case trains on all n(n-1)/2 pairs; the incomplete case on
B = n log n pairs sampled with replacement (Blom, 1976), preserving the
O(1/sqrt(n)) rate for minimizers (Clemencon et al., 2016).

Usage:
    python experiments/synthetic_incomplete_u.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from pairwise_qr import generate_pairs, pinball_loss_np, subsample_pairs, train_nn

# ---- Configuration ----
N_TRAIN = 1000
N_TEST = 100
TAU = 0.5
N_EPOCHS = 100
SEED = 42

# ---- Data ----
X_tr, y_tr = generate_pairs(N_TRAIN, rng=SEED)
X_te, y_te = generate_pairs(N_TEST, rng=SEED + 1000)
X_inc, y_inc = subsample_pairs(X_tr, y_tr, rng=SEED)  # B = n log n
print(f"Complete: {len(y_tr)} pairs | Incomplete: {len(y_inc)} pairs")

# ---- Train both ----
rows = []
for label, (X, y) in {
    "Complete (n^2)": (X_tr, y_tr),
    "Incomplete (n log n)": (X_inc, y_inc),
}.items():
    t0 = time.time()
    model, _ = train_nn(X, y, TAU, n_epochs=N_EPOCHS, seed=SEED)
    runtime = time.time() - t0
    loss = pinball_loss_np(y_te, model.predict(X_te), TAU)
    rows.append({"Method": label, "Training Time (s)": runtime, "Pinball Loss": loss})

df = pd.DataFrame(rows)
print(df.to_string(index=False))
