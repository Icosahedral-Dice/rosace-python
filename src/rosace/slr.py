"""Simple Linear Regression scoring for rosace.

Mirrors the R `runSLR` function: fits a linear regression of normalized
counts vs time for each variant and returns the slope as the score.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.stats

from rosace.assay import AssayGrowth, AssaySetGrowth
from rosace.score import Score


def run_slr(
    assay: Union[AssayGrowth, AssaySetGrowth],
    t: Optional[list[float]] = None,
) -> Score:
    """Run Simple Linear Regression (SLR) scoring.

    For each variant, fits an OLS regression of normalized count vs time and
    uses the slope as the functional score. This is the "naive" scoring
    method used as an alternative to the full Bayesian ROSACE model.

    Parameters
    ----------
    assay:
        AssayGrowth or AssaySetGrowth with ``norm_counts`` populated (for
        AssayGrowth) or ``combined_counts`` populated (for AssaySetGrowth).
    t:
        Time vector. If None, uses ``[0, 1, ..., T-1]`` for each replicate.

    Returns
    -------
    Score
        Score object with slope (``mean``), standard error (``sd``), and
        local false sign rate (``lfsr``) per variant.

        .. note::
            Calling :func:`~rosace.utils.output_score` on the returned
            ``Score`` will recompute and overwrite the ``lfsr`` column using
            the same normal-approximation formula.  Both computations are
            equivalent; the ``lfsr`` stored here is provided for convenience
            so the raw ``Score`` object is already usable without an extra
            ``output_score`` call.
    """
    if isinstance(assay, AssayGrowth):
        return _slr_assay_growth(assay, t)
    if isinstance(assay, AssaySetGrowth):
        return _slr_assay_set_growth(assay, t)
    raise TypeError(
        f"assay must be AssayGrowth or AssaySetGrowth, got {type(assay)}"
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_lfsr(mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Compute local false sign rate from mean and standard deviation."""
    safe_sd = np.where(np.isnan(sd) | (sd <= 0), 1e-6, sd)
    lfsr_neg = scipy.stats.norm.sf(0, loc=mu, scale=safe_sd)
    lfsr_pos = scipy.stats.norm.cdf(0, loc=mu, scale=safe_sd)
    return np.minimum(lfsr_neg, lfsr_pos)


def _slr_assay_growth(assay: AssayGrowth, t: Optional[list[float]]) -> Score:
    """SLR for a single AssayGrowth replicate."""
    norm_counts = assay.norm_counts
    if norm_counts is None:
        raise ValueError("assay.norm_counts must be populated before running SLR.")

    var_names = assay.norm_var_names or assay.var_names
    n_vars = len(var_names)
    T = norm_counts.shape[1]
    rounds = assay.rounds  # T - 1
    if t is None:
        # Match R: seq(0, rounds)/rounds → [0, 1/rounds, ..., 1]
        t_arr = np.arange(T, dtype=float) / rounds if rounds > 0 else np.zeros(T)
    else:
        t_arr = np.array(t[:T], dtype=float)

    slopes = np.full(n_vars, np.nan)
    stderrs = np.full(n_vars, np.nan)

    for i in range(n_vars):
        y = norm_counts[i, :]
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            continue
        slope, _, _, _, stderr = scipy.stats.linregress(t_arr[mask], y[mask])
        slopes[i] = slope
        stderrs[i] = stderr

    lfsr = _compute_lfsr(np.nan_to_num(slopes, nan=0.0), stderrs)

    score_df = pd.DataFrame(
        {"variant": var_names, "mean": slopes, "sd": stderrs, "lfsr": lfsr}
    )

    return Score(
        method="SLR",
        type="AssayGrowth",
        assay_name=assay.key,
        score=score_df,
    )


def _slr_assay_set_growth(
    assay: AssaySetGrowth, t: Optional[list[float]]
) -> Score:
    """SLR for an AssaySetGrowth (pooled regression across all replicates).

    For each variant, concatenates (time, normalized_count) pairs from every
    replicate and fits a single pooled OLS regression.

    Time is normalised by ``max(rounds)`` so that each replicate spans
    ``[0, 1/denom, ..., rounds_i/denom]``, matching the R implementation:
    ``xt <- unlist(lapply(rounds, function(x) seq(0, x) / max(rounds)))``.
    """
    var_names = assay.var_names
    n_vars = len(var_names)

    rounds_list = assay.rounds
    if rounds_list is None:
        raise ValueError(
            "AssaySetGrowth.rounds must be populated before running SLR."
        )

    # Match R: denom <- max(rounds); xt <- seq(0, x)/denom for each replicate
    max_rounds = max(rounds_list)

    # Build (t_block, col_slice) pairs for each replicate
    blocks: list[tuple[np.ndarray, list[int]]] = []
    col_offset = 0
    for n_rounds in rounds_list:
        n_tp = n_rounds + 1
        if t is None:
            t_block = np.arange(n_tp, dtype=float) / max_rounds
        else:
            t_block = np.array(t[:n_tp], dtype=float)
        blocks.append((t_block, list(range(col_offset, col_offset + n_tp))))
        col_offset += n_tp

    slopes = np.full(n_vars, np.nan)
    stderrs = np.full(n_vars, np.nan)

    for i in range(n_vars):
        t_pooled: list[float] = []
        y_pooled: list[float] = []
        for t_block, cols in blocks:
            y_block = assay.combined_counts[i, cols]
            mask = ~np.isnan(y_block)
            if mask.sum() >= 2:
                t_pooled.extend(t_block[mask].tolist())
                y_pooled.extend(y_block[mask].tolist())

        if len(t_pooled) < 2:
            continue
        slope, _, _, _, stderr = scipy.stats.linregress(t_pooled, y_pooled)
        slopes[i] = slope
        stderrs[i] = stderr

    lfsr = _compute_lfsr(np.nan_to_num(slopes, nan=0.0), stderrs)

    score_df = pd.DataFrame(
        {"variant": var_names, "mean": slopes, "sd": stderrs, "lfsr": lfsr}
    )

    return Score(
        method="SLR",
        type="AssaySetGrowth",
        assay_name=assay.key,
        score=score_df,
    )
