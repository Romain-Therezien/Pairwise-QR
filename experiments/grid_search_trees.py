"""Hyperparameter grid search for tree-based baselines (Appendix C).

Selection criterion: MAE against the MC ground-truth conditional quantile,
averaged over quantile levels and repeated seeds.

Usage:
    python experiments/grid_search_trees.py --model lightgbm
    python experiments/grid_search_trees.py --model gbm
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.model_selection import ParameterGrid

from pairwise_qr import generate_pairs, make_gbm, make_lightgbm, mc_true_quantiles
from pairwise_qr.metrics import mae_vs_truth

GRIDS = {
    "gbm": {
        "n_estimators": [300, 600],
        "learning_rate": [0.01, 0.05],
        "max_depth": [3, 5, 6],
        "min_samples_leaf": [5, 10, 20],
        "subsample": [0.7, 0.9],
    },
    "lightgbm": {
        "num_leaves": [31, 63],
        "max_depth": [6, 10],
        "learning_rate": [0.01, 0.05],
        "min_child_samples": [10, 20],
        "subsample": [0.7, 0.9],
        "reg_lambda": [0, 1],
    },
}
FACTORIES = {"gbm": make_gbm, "lightgbm": make_lightgbm}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gbm", "lightgbm"], required=True)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=list(np.linspace(0.05, 0.95, 19)),
        help="Include the operating quantiles you care about "
        "(e.g. 0.01 0.05 0.95 0.99 for the FR use case).",
    )
    args = parser.parse_args()

    X_val, _ = generate_pairs(args.n_val, rng=42)
    print(f"Computing MC ground truth on {X_val.shape[0]} validation pairs...")
    true_q = mc_true_quantiles(X_val, args.quantiles, n_mc=5000, rng=42)

    factory = FACTORIES[args.model]
    best_score, best_params = np.inf, None
    for params in ParameterGrid(GRIDS[args.model]):
        repeat_scores = []
        for seed in range(args.n_repeats):
            X_tr, y_tr = generate_pairs(args.n_train, rng=seed)
            errs = []
            for qi, tau in enumerate(args.quantiles):
                model = factory(tau, seed=seed, **params)
                model.fit(X_tr, y_tr)
                errs.append(mae_vs_truth(true_q[qi], model.predict(X_val)))
            repeat_scores.append(np.mean(errs))
            print(f"{params} | seed {seed} | score {repeat_scores[-1]:.4f}")
        score = float(np.mean(repeat_scores))
        if score < best_score:
            best_score, best_params = score, params

    print("\nBest params:", best_params)
    print("Best score:", best_score)


if __name__ == "__main__":
    main()
