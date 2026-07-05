"""Pairwise Quantile Regression.

Code for "On Pairwise Quantile Regression - Statistical Guarantees and
Applications" (UAI 2026).

The empirical risk is a U-statistic of degree 2: the pinball loss averaged
over all pairs (i, j), i < j, of training observations. See Section 3 of the
paper for the statistical framework and Theorem 1 for the fast-rate analysis.
"""

from .losses import pinball_loss_np, pinball_loss_torch
from .data import generate_pairs, subsample_pairs
from .ground_truth import mc_true_quantiles
from .features import make_symmetric_features
from .models import NNQuantileRegressor, train_nn, make_gbm, make_lightgbm
from .metrics import d2_score, empirical_coverage

__all__ = [
    "pinball_loss_np",
    "pinball_loss_torch",
    "generate_pairs",
    "subsample_pairs",
    "mc_true_quantiles",
    "make_symmetric_features",
    "NNQuantileRegressor",
    "train_nn",
    "make_gbm",
    "make_lightgbm",
    "d2_score",
    "empirical_coverage",
]
