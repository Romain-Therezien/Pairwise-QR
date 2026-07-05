"""Symmetric pair features.

Each base covariate c yields two order-invariant features (Section 4.2):
    c_sum  = c_1 + c_2          (aggregate effect across the pair)
    c_diff = |c_1 - c_2|        (discrepancy between the two images)

This guarantees invariance to pair ordering, q(z, z') = q(z', z), as required
of the hypothesis class Q in the paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_symmetric_features(
    df: pd.DataFrame,
    base_cols,
    suffixes: tuple[str, str] = ("_1", "_2"),
) -> pd.DataFrame:
    """Build (sum, |diff|) features from per-image columns ``c{suffix}``."""
    s1, s2 = suffixes
    out = {}
    for col in base_cols:
        c1, c2 = df[f"{col}{s1}"], df[f"{col}{s2}"]
        out[f"{col}_sum"] = c1 + c2
        out[f"{col}_diff"] = np.abs(c1 - c2)
    return pd.DataFrame(out, index=df.index)
