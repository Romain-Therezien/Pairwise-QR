"""Facial recognition application (Section 4.2, Table 1, Figs. 3-5 and 12-17).

Pipeline: load pair scores + per-image covariates, build symmetric features
(Z + Z', |Z - Z'|), train one NN quantile regressor per level tau, evaluate
D^2 and coverage on the held-out split, then run the SHAP analyses.

Expects data/Scores.parquet and data/Qualities.parquet (see data/README.md).

Usage:
    python experiments/facial_recognition.py --pair-type genuine
    python experiments/facial_recognition.py --pair-type impostor
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pairwise_qr import make_symmetric_features, train_nn
from pairwise_qr.metrics import d2_score, empirical_coverage

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SEED = 42
QUANTILES = [0.01] + [round(x, 2) for x in np.linspace(0, 1, 21)[1:-1]] + [0.99]
SHAP_BACKGROUND = 500   # background sample size for the SHAP explainer
SHAP_EVAL = 2000        # validation rows explained (full set is very slow)


# ---------------------------------------------------------------- data
def load_data(pair_type: str):
    scores = pd.read_parquet(DATA / "Scores.parquet", engine="fastparquet")
    qualities = pd.read_parquet(DATA / "Qualities.parquet", engine="fastparquet")

    diag = 1 if pair_type == "genuine" else 0
    df = scores[scores["diag"] == diag]
    df = df.merge(qualities.add_suffix("_1"), on="image_1", how="inner")
    df = df.merge(qualities.add_suffix("_2"), on="image_2", how="inner")

    base_cols = [c for c in qualities.columns if c != "image"]
    X = make_symmetric_features(df, base_cols)
    y = df["score"].values.astype(np.float32)
    return X, y


# ---------------------------------------------------------------- shap
def run_shap(models, X_train, X_val, feature_names, taus_to_plot, pair_type):
    import shap

    rng = np.random.default_rng(SEED)
    background = X_train[rng.choice(len(X_train), min(SHAP_BACKGROUND, len(X_train)), replace=False)]
    X_explain = X_val[rng.choice(len(X_val), min(SHAP_EVAL, len(X_val)), replace=False)]

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            return self.model(x)

    shap_values = {}
    for tau in taus_to_plot:
        print(f"SHAP for tau = {tau}")
        explainer = shap.Explainer(Wrapper(models[tau].eval()), background)
        sv = explainer(X_explain)
        sv.feature_names = feature_names
        shap_values[tau] = sv

        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(sv, max_display=6, show=False, group_remaining_features=False)
        plt.title(f"Most Important Variables (tau = {tau}, {pair_type} pairs)")
        plt.savefig(OUT / f"shap_beeswarm_{pair_type}_tau{tau}.pdf", bbox_inches="tight")
        plt.close()

    # Feature-importance heatmap across quantiles (Fig. 5)
    importance = pd.DataFrame(
        {tau: np.abs(sv.values).mean(axis=0) for tau, sv in shap_values.items()},
        index=feature_names,
    )
    top = importance.mean(axis=1).nlargest(11).index
    plt.figure(figsize=(8, 6))
    sns.heatmap(importance.loc[top], cmap="YlGnBu")
    plt.title(f"Feature importance across quantiles ({pair_type} pairs)")
    plt.xlabel("Quantiles")
    plt.ylabel("Features")
    plt.savefig(OUT / f"shap_heatmap_{pair_type}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP figures to {OUT}")


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-type", choices=["genuine", "impostor"], default="genuine")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args()

    print(f"Loading {args.pair_type} pairs...")
    X, y = load_data(args.pair_type)
    feature_names = X.columns.tolist()

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    # ---- Per-quantile training ----
    models, d2, coverage = {}, {}, {}
    for tau in QUANTILES:
        print(f"Training tau = {tau}")
        models[tau], _ = train_nn(
            X_tr, y_tr, tau,
            hidden_units=(64, 32), n_epochs=args.epochs, lr=1e-3, seed=SEED,
        )
        preds = models[tau].predict(X_val)
        d2[tau] = d2_score(y_val, preds, tau, y_train=y_tr)
        coverage[tau] = empirical_coverage(y_val, preds)
        print(f"  D2 = {d2[tau]:.3f} | coverage = {coverage[tau]:.3f}")

    # ---- Table 1: D2 at the operating quantiles ----
    operating = [0.01, 0.05] if args.pair_type == "genuine" else [0.95, 0.99]
    print("\nD2-score at operating quantiles (Table 1):")
    for tau in operating:
        print(f"  tau = {tau}: {100 * d2[tau]:.1f}%")

    # ---- Relative improvement across quantiles (Fig. 13/15) ----
    plt.figure(figsize=(10, 6))
    plt.plot(QUANTILES, [d2[t] for t in QUANTILES], marker="o")
    plt.title(f"Relative Improvement vs Quantiles ({args.pair_type} pairs)")
    plt.xlabel("Quantile (tau)")
    plt.ylabel("Relative Improvement (D2)")
    plt.grid(True)
    plt.savefig(OUT / f"d2_across_quantiles_{args.pair_type}.pdf", bbox_inches="tight")
    plt.close()

    # ---- Calibration on the held-out split (Fig. 12) ----
    plt.figure(figsize=(10, 6))
    plt.plot(QUANTILES, [coverage[t] for t in QUANTILES], marker="o", label="Empirical Coverage")
    plt.plot(QUANTILES, QUANTILES, linestyle="--", label="Ideal (y = x)")
    plt.title(f"Quantile Coverage on held-out data ({args.pair_type} pairs)")
    plt.xlabel("Quantile (tau)")
    plt.ylabel("Empirical Coverage")
    plt.grid(True)
    plt.legend()
    plt.savefig(OUT / f"coverage_{args.pair_type}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved D2 and coverage figures to {OUT}")

    # ---- SHAP analyses (Figs. 3-5) ----
    if not args.skip_shap:
        taus_to_plot = operating + [0.5]
        run_shap(models, X_tr, X_val, feature_names, taus_to_plot, args.pair_type)


if __name__ == "__main__":
    main()
