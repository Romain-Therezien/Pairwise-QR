# Pairwise Quantile Regression

Code for the paper **"On Pairwise Quantile Regression — Statistical Guarantees
and Applications"** (UAI 2026).

Quantile regression extended to the pairwise setting: the response is a
similarity score `s(X, X′)` between two independent observations and the
covariates are the pair `(Z, Z′)`. The conditional quantile is learned by
minimizing the **pairwise pinball loss in the form of a U-statistic**, for
which the paper establishes fast learning rates of order `O(log(n)/n)` —
faster than the `1/√n` rate of pointwise quantile regression — under a mild
lower bound on the conditional density near the quantile.


## Repository structure

```
pairwise_qr/                    # Reusable library
├── losses.py                   #   pinball loss (torch + numpy)
├── data.py                     #   synthetic DGP, complete & incomplete pair sets
├── ground_truth.py             #   Monte-Carlo true conditional quantiles
├── features.py                 #   symmetric pair features (Z+Z', |Z−Z'|)
├── models.py                   #   NN quantile regressor, GBM/LightGBM factories
└── metrics.py                  #   D², empirical coverage, MAE vs ground truth
experiments/                    # One script per paper result (see table below)
tests/                          # Smoke tests of the statistical building blocks
data/                           # FR parquet files (added separately — see data/README.md)
outputs/                        # Generated figures (gitignored)
```

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/Romain-Therezien/Pairwise-QR.git
cd Pairwise-QR
pip install -r requirements.txt
```

## Quickstart

```python
from pairwise_qr import generate_pairs, train_nn
from pairwise_qr.metrics import empirical_coverage

# n i.i.d. samples -> all n(n-1)/2 pairs (the complete U-statistic sample)
X_train, y_train = generate_pairs(500, rng=42)
X_test,  y_test  = generate_pairs(100, rng=43)

# one model per quantile level tau, trained with the pinball loss
model, history = train_nn(X_train, y_train, tau=0.9, n_epochs=100, seed=42)

print("coverage:", empirical_coverage(y_test, model.predict(X_test)))  
```

For large n, replace the complete pair set with an incomplete U-statistic of
`n log n` sampled pairs at a negligible cost in accuracy (Table 2):

```python
from pairwise_qr import subsample_pairs
X_inc, y_inc = subsample_pairs(X_train, y_train, rng=42)   # B = n log n pairs
```

## Reproducing the paper

| Paper result | Command |
|---|---|
| Fig. 1 - Pinball loss across quantiles | `python experiments/synthetic_pinball_curve.py` |
| Fig. 2 - Conditional quantile surfaces (τ = 0.1, 0.5, 0.9) | `python experiments/synthetic_surfaces.py` |
| Fig. 6 - NN vs LightGBM vs Gradient Boosting (MAE vs MC ground truth) | `python experiments/synthetic_model_comparison.py` |
| Table 2 - Complete vs incomplete U-statistics (runtime / accuracy) | `python experiments/synthetic_incomplete_u.py` |
| Appendix C - Hyperparameter grid searches | `python experiments/grid_search_trees.py --model lightgbm` (or `--model gbm`) |
| Table 1, Figs. 3-5, 12-17 - FR results, coverage, SHAP | `python experiments/facial_recognition.py --pair-type genuine` (and `--pair-type impostor`) |

All experiments use fixed seeds (42, and 42–51 for repeated runs). Figures are
written to `outputs/`.


## Facial recognition data

The FR experiments use **125,052 genuine pairs** and **1,149,498 impostor
pairs**, each image annotated with quality and demographic covariates (Table 6
of the paper). The dataset contains only similarity scores and per-image covariates, no
images and no identity information. All synthetic experiments run without it.

For each pair type, the operating quantiles are the ones that drive
recognition errors: **low quantiles (τ = 0.01, 0.05) for genuine pairs**
(false rejections) and **high quantiles (τ = 0.95, 0.99) for impostor pairs**
(false acceptances).

## Method in one paragraph

Given training samples `(X₁, Z₁), …, (Xₙ, Zₙ)`, the empirical risk is the
U-statistic of degree 2 averaging the pinball loss
`ρ_τ(s(Xᵢ, Xⱼ) − q(Zᵢ, Zⱼ))` over all pairs `i < j` (Eq. 12). Its
reduced-variance property yields excess-risk bounds of order `log(n)/n`
(Theorem 1) under a lower bound on the conditional density near the quantile
(Assumption 3); and unlike pointwise quantile regression, the strongest
variance–risk control (θ = 1) always holds in the pairwise case
(Proposition 1). Inputs are made symmetric via `(Z + Z′, |Z − Z′|)`, ensuring
invariance to pair ordering. Interpretability is obtained post hoc with
Shapley values, decomposing each predicted conditional quantile into
covariate-specific contributions.

## Citation

```bibtex
@inproceedings{therezien2026pairwise,
  title     = {On Pairwise Quantile Regression: Statistical Guarantees and Applications},
  author    = {Therezien, Romain and Cl{\'e}men{\c{c}}on, Stephan and Fantin, Girard and El-Abdouni, Hamza},
  booktitle = {Proceedings of the Conference on Uncertainty in Artificial Intelligence (UAI)},
  year      = {2026}
}
```

## Acknowledgments

This research was fully funded by the French National Research Agency (ANR) in the framework of the FAR-SEE project (ANR-24-CE23-0921).

