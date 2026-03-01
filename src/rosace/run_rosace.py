"""Functions to prepare input and run ROSACE Stan models."""

from __future__ import annotations
import os
import tempfile
from typing import Optional, Union
import numpy as np
import pandas as pd
import scipy.stats

from rosace.assay import AssayGrowth, AssaySetGrowth
from rosace.score import Score
from rosace.stan_models import STAN_MODELS
from rosace.utils import map_blosum_score


def gen_rosace_input(
    assay: Union[AssayGrowth, AssaySetGrowth],
    method: str = "ROSACE1",
    t: Optional[list[float]] = None,
    n_mean_groups: int = 5,
    var_info: Optional[pd.DataFrame] = None,
) -> dict:
    """Generate the Stan data dictionary for a ROSACE growth model.

    Parameters
    ----------
    assay:
        AssayGrowth or AssaySetGrowth with norm_counts populated.
    method:
        One of ROSACE0, ROSACE1, ROSACE2, ROSACE3.
    t:
        Time vector. If None, uses [0, 1, ..., T-1].
    n_mean_groups:
        Number of mean count groups for vMAPm (quantile-based).
    var_info:
        DataFrame with columns ``variant``, ``pos``, ``wt``, ``mut``.
        Required for ROSACE1/2/3 (position-level grouping).
        For ROSACE2/3, also needs ``wt`` and ``mut`` for BLOSUM mapping.

    Returns
    -------
    dict
        Stan data dictionary ready to pass to CmdStanPy.
    """
    if isinstance(assay, AssayGrowth):
        norm_counts = assay.norm_counts
        var_names = assay.norm_var_names or assay.var_names
        T = assay.counts.shape[1]
    else:
        norm_counts = assay.combined_counts
        var_names = assay.var_names
        T = assay.combined_counts.shape[1]

    if norm_counts is None:
        raise ValueError("assay must have norm_counts populated.")

    V = len(var_names)
    if t is None:
        t = list(range(T))

    # Replace any remaining NaN with 0 in norm_counts
    m = np.nan_to_num(norm_counts, nan=0.0)

    # Mean count groups (vMAPm): based on mean of absolute norm counts
    mean_abs = np.abs(m).mean(axis=1)
    quantiles = np.percentile(mean_abs, np.linspace(0, 100, n_mean_groups + 1))
    vMAPm = np.searchsorted(quantiles[1:-1], mean_abs, side="right") + 1  # 1-indexed

    data: dict = {
        "T": T,
        "V": V,
        "M": n_mean_groups,
        "vMAPm": vMAPm.tolist(),
        "t": t,
        "m": m.tolist(),
    }

    if method == "ROSACE0":
        return data

    # For ROSACE1/2/3: need position mapping
    if var_info is None:
        raise ValueError(f"var_info is required for method={method!r}")

    vi = var_info.set_index("variant")

    # Build position list in order of appearance
    pos_order: list = []
    pos_map: dict = {}
    for v in var_names:
        if v not in vi.index:
            raise KeyError(f"Variant {v!r} not found in var_info")
        pos = vi.loc[v, "pos"]
        if pos not in pos_map:
            pos_map[pos] = len(pos_order) + 1  # 1-indexed
            pos_order.append(pos)

    P = len(pos_order)
    vMAPp = [pos_map[vi.loc[v, "pos"]] for v in var_names]

    data["P"] = P
    data["vMAPp"] = vMAPp

    if method == "ROSACE1":
        return data

    # ROSACE2/3: BLOSUM grouping
    blosum_groups: list[int] = []
    for v in var_names:
        row = vi.loc[v]
        try:
            group = map_blosum_score(str(row["wt"]), str(row["mut"]))
        except (ValueError, KeyError):
            group = 5  # default to group 5 (score 0) for unknown
        blosum_groups.append(group)

    B = 11  # groups 1..11
    data["B"] = B
    data["vMAPb"] = blosum_groups

    return data


def _compute_lfsr(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute LFSR from posterior samples.

    Parameters
    ----------
    samples:
        Array of shape (draws, variants).

    Returns
    -------
    tuple of (mean, sd, lfsr) arrays.
    """
    mu = samples.mean(axis=0)
    sd = samples.std(axis=0)
    lfsr_neg = scipy.stats.norm.sf(0, loc=mu, scale=sd + 1e-10)
    lfsr_pos = scipy.stats.norm.cdf(0, loc=mu, scale=sd + 1e-10)
    lfsr = np.minimum(lfsr_neg, lfsr_pos)
    return mu, sd, lfsr


def run_rosace(
    assay: Union[AssayGrowth, AssaySetGrowth],
    method: str = "ROSACE1",
    t: Optional[list[float]] = None,
    n_mean_groups: int = 5,
    var_info: Optional[pd.DataFrame] = None,
    chains: int = 4,
    parallel_chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 1000,
    seed: Optional[int] = None,
) -> Score:
    """Run ROSACE MCMC inference and return a Score object.

    Parameters
    ----------
    assay:
        AssayGrowth or AssaySetGrowth with norm_counts populated.
    method:
        Stan model to use: ROSACE0, ROSACE1, ROSACE2, or ROSACE3.
    t:
        Time vector.
    n_mean_groups:
        Number of mean count groups.
    var_info:
        DataFrame with variant metadata (required for ROSACE1/2/3).
    chains:
        Number of MCMC chains.
    parallel_chains:
        Number of chains to run in parallel.
    iter_warmup:
        Number of warmup iterations per chain.
    iter_sampling:
        Number of sampling iterations per chain.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Score
        Score object with posterior mean, sd, and LFSR per variant.
    """
    try:
        import cmdstanpy
    except ImportError as exc:
        raise ImportError(
            "cmdstanpy is required to run ROSACE. Install it with: pip install cmdstanpy"
        ) from exc

    stan_code = STAN_MODELS.get(method)
    if stan_code is None:
        raise ValueError(f"Unknown method {method!r}. Choose from {list(STAN_MODELS)}")

    stan_data = gen_rosace_input(
        assay=assay,
        method=method,
        t=t,
        n_mean_groups=n_mean_groups,
        var_info=var_info,
    )

    if isinstance(assay, AssayGrowth):
        var_names = assay.norm_var_names or assay.var_names
        assay_name = assay.key
        assay_type = "AssayGrowth"
    else:
        var_names = assay.var_names
        assay_name = assay.key
        assay_type = "AssaySetGrowth"

    with tempfile.NamedTemporaryFile(suffix=".stan", mode="w", delete=False) as fh:
        fh.write(stan_code)
        stan_file = fh.name

    try:
        model = cmdstanpy.CmdStanModel(stan_file=stan_file)
        fit = model.sample(
            data=stan_data,
            chains=chains,
            parallel_chains=parallel_chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            show_progress=False,
        )
    finally:
        os.unlink(stan_file)

    # Extract beta posteriors: shape (total_draws, V)
    beta_samples = fit.stan_variable("beta")  # (draws, V)
    mu, sd, lfsr = _compute_lfsr(beta_samples)

    score_df = pd.DataFrame({
        "variant": var_names,
        "mean": mu,
        "sd": sd,
        "lfsr": lfsr,
    })

    # Position-level scores for ROSACE1/2/3
    optional_score = None
    if method in ("ROSACE1", "ROSACE2", "ROSACE3") and var_info is not None:
        phi_samples = fit.stan_variable("phi")  # (draws, P)
        phi_mu = phi_samples.mean(axis=0)
        phi_sd = phi_samples.std(axis=0)
        vi = var_info.set_index("variant")
        pos_order: list = []
        seen: set = set()
        for v in var_names:
            if v in vi.index:
                pos = vi.loc[v, "pos"]
                if pos not in seen:
                    pos_order.append(pos)
                    seen.add(pos)
        optional_score = pd.DataFrame({
            "pos": pos_order,
            "mean": phi_mu,
            "sd": phi_sd,
        })

    return Score(
        method=method,
        type=assay_type,
        assay_name=assay_name,
        score=score_df,
        optional_score=optional_score,
    )
