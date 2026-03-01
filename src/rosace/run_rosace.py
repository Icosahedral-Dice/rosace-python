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


def _cmdstan_available() -> bool:
    """Return True if CmdStan is installed and usable."""
    try:
        import cmdstanpy
        cmdstanpy.cmdstan_path()
        return True
    except Exception:
        return False


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


def _bayesian_linreg(
    t_obs: np.ndarray,
    y_obs: np.ndarray,
    eps2: float,
    prior_var_b: float,
    prior_var_slope: float,
) -> tuple[float, float, float, float]:
    """Bayesian linear regression with Gaussian priors on intercept and slope.

    Returns (b_mean, slope_mean, b_var, slope_var).
    """
    X = np.column_stack([np.ones(len(t_obs)), t_obs])
    Lambda_prior = np.diag([1.0 / prior_var_b, 1.0 / prior_var_slope])
    A = X.T @ X / eps2 + Lambda_prior
    rhs = X.T @ y_obs / eps2
    try:
        params = np.linalg.solve(A, rhs)
        Sigma_post = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, prior_var_b, prior_var_slope
    return float(params[0]), float(params[1]), float(Sigma_post[0, 0]), float(Sigma_post[1, 1])


def _batch_bayesian_linreg(
    m: np.ndarray,
    t_arr: np.ndarray,
    eps2_per_variant: np.ndarray,
    prior_var_b: float = 0.0625,
    prior_var_slope_per_variant: Optional[np.ndarray] = None,
    prior_var_slope_scalar: float = 1.0,
    phi_per_variant: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised 2x2 Bayesian linear regression for all V variants at once.

    Solves the posterior for [b[v], slope[v]] jointly for all variants using
    closed-form 2x2 matrix inversion.  Handles missing data (NaN) via masked
    sufficient statistics.

    Parameters
    ----------
    m:
        Array of shape (V, T).  NaN = missing.
    t_arr:
        Time vector of length T.
    eps2_per_variant:
        Noise variance per variant, shape (V,).
    prior_var_b:
        Prior variance for intercept b (same for all variants).
    prior_var_slope_per_variant:
        Prior variance for slope (per-variant); overrides prior_var_slope_scalar.
    prior_var_slope_scalar:
        Scalar prior variance for slope when prior_var_slope_per_variant is None.
    phi_per_variant:
        Position-level mean slope phi[vMAPp[v]] to subtract from effective slope.
        When provided, the design matrix models y_tilde = y - phi*t so the
        returned slope is the residual u[v] = eta2[v]*sqrt(sigma2[p]).

    Returns
    -------
    (b_mean, slope_mean, b_var, slope_var)  each of shape (V,).
    """
    V, T = m.shape
    nan_mask = ~np.isnan(m)  # (V, T)
    t2 = t_arr ** 2

    # Sufficient statistics (masked)
    n_obs = nan_mask.sum(axis=1).astype(float)          # (V,)
    sum_t = (nan_mask * t_arr).sum(axis=1)               # (V,)
    sum_t2 = (nan_mask * t2).sum(axis=1)                 # (V,)

    # Residualised observations
    if phi_per_variant is not None:
        y_eff = np.where(nan_mask, m - phi_per_variant[:, None] * t_arr, 0.0)
    else:
        y_eff = np.where(nan_mask, m, 0.0)

    sum_y = y_eff.sum(axis=1)                            # (V,)
    sum_ty = (y_eff * t_arr).sum(axis=1)                 # (V,)

    # Prior precision
    prec_b = 1.0 / prior_var_b
    if prior_var_slope_per_variant is not None:
        prec_s = 1.0 / np.clip(prior_var_slope_per_variant, 1e-12, None)
    else:
        prec_s = np.full(V, 1.0 / prior_var_slope_scalar)

    eps = np.clip(eps2_per_variant, 1e-12, None)

    # 2x2 posterior precision matrix A (per variant)
    A00 = n_obs / eps + prec_b
    A01 = sum_t / eps
    A11 = sum_t2 / eps + prec_s

    # RHS vector (sufficient stats / eps)
    r0 = sum_y / eps
    r1 = sum_ty / eps

    # 2x2 inverse via Cramer's rule
    det = A00 * A11 - A01 ** 2
    det = np.where(det < 1e-30, 1e-30, det)

    Sigma00 = A11 / det   # Var[b]
    Sigma11 = A00 / det   # Var[slope]
    Sigma01 = -A01 / det

    b_mean = Sigma00 * r0 + Sigma01 * r1
    s_mean = Sigma01 * r0 + Sigma11 * r1

    # Zero-out estimates for variants with no observations
    no_obs = n_obs < 1
    b_mean = np.where(no_obs, 0.0, b_mean)
    s_mean = np.where(no_obs, 0.0, s_mean)
    Sigma00 = np.where(no_obs, prior_var_b, Sigma00)
    Sigma11 = np.where(no_obs, prior_var_slope_scalar if prior_var_slope_per_variant is None else 1.0, Sigma11)

    return b_mean, s_mean, Sigma00, Sigma11


def _run_rosace0_scipy(
    m: np.ndarray,
    t_arr: np.ndarray,
    vMAPm: np.ndarray,
    M: int,
    V: int,
    n_iter: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """MAP + Laplace approximation for the ROSACE0 model.

    Uses coordinate ascent: analytically update (b, beta) per variant given
    eps2, then MAP-update eps2 per mean-count group.

    Parameters
    ----------
    m:
        Normalised counts array of shape (V, T).  NaN is treated as missing.
    t_arr:
        Time vector of length T.
    vMAPm:
        0-indexed mean-count group assignments of length V.
    M:
        Number of mean-count groups.
    V:
        Number of variants.
    n_iter:
        Number of EM iterations.

    Returns
    -------
    (beta_mean, beta_sd)  arrays of length V.
    """
    # Initialise noise variance per group (sample variance or fallback)
    eps2 = np.full(M, 0.1)
    for k in range(M):
        vals = m[vMAPm == k].ravel()
        vals = vals[~np.isnan(vals)]
        if len(vals) > 1:
            eps2[k] = max(np.var(vals), 1e-6)

    b_map = np.zeros(V)
    beta_map = np.zeros(V)
    beta_var = np.full(V, 1.0)

    nan_mask = ~np.isnan(m)  # (V, T) – precompute once

    for _ in range(n_iter):
        # E-step: vectorised Bayesian linear regression for all variants
        b_map, beta_map, _, beta_var = _batch_bayesian_linreg(
            m, t_arr,
            eps2_per_variant=eps2[vMAPm],
            prior_var_b=0.0625,
            prior_var_slope_scalar=1.0,
        )

        # M-step: MAP-update eps2 per group (InvGamma(1,1) prior)
        resid = np.where(nan_mask, m - b_map[:, None] - beta_map[:, None] * t_arr, 0.0)
        n_obs_per_v = nan_mask.sum(axis=1).astype(float)
        ssr_per_v = (resid ** 2).sum(axis=1)
        for k in range(M):
            mask_k = vMAPm == k
            ssr = float(ssr_per_v[mask_k].sum())
            n_total = int(n_obs_per_v[mask_k].sum())
            if n_total > 0:
                eps2[k] = (1.0 + 0.5 * ssr) / (1.0 + 0.5 * n_total + 1.0)

    return beta_map, np.sqrt(np.clip(beta_var, 1e-12, None))


def _run_rosace1_scipy(
    m: np.ndarray,
    t_arr: np.ndarray,
    vMAPm: np.ndarray,
    vMAPp: np.ndarray,
    M: int,
    V: int,
    P: int,
    n_iter: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """MAP + Laplace approximation for the ROSACE1 model (position-level effects).

    Uses coordinate ascent over:
      - (b[v], u[v]) per variant, where u[v] = eta2[v] * sqrt(sigma2[vMAPp[v]])
      - phi[p] per position (Bayesian shrinkage mean)
      - sigma2[p] per position (MAP of InvGamma posterior)
      - eps2[k] per mean-count group (MAP of InvGamma posterior)

    Returns
    -------
    (beta_mean, beta_sd, phi_mean, phi_sd)
        Arrays of length V and P respectively.
    """
    # Initialise
    eps2 = np.full(M, 0.1)
    for k in range(M):
        vals = m[vMAPm == k].ravel()
        vals = vals[~np.isnan(vals)]
        if len(vals) > 1:
            eps2[k] = max(np.var(vals), 1e-6)

    phi = np.zeros(P)
    sigma2 = np.ones(P)
    b_map = np.zeros(V)
    u_map = np.zeros(V)      # u[v] = eta2[v] * sqrt(sigma2[p])
    u_var = np.ones(V)       # posterior variance of u[v]
    nan_mask = ~np.isnan(m)

    for _ in range(n_iter):
        # --- Update (b[v], u[v]) via vectorised Bayesian linear regression ---
        phi_at_v = phi[vMAPp]
        sigma2_at_v = sigma2[vMAPp]
        b_map, u_map, _, u_var = _batch_bayesian_linreg(
            m, t_arr,
            eps2_per_variant=eps2[vMAPm],
            prior_var_b=0.0625,
            prior_var_slope_per_variant=sigma2_at_v,
            phi_per_variant=phi_at_v,
        )

        # --- Update phi[p] and sigma2[p] (vectorised via np.bincount) ---
        n_p = np.bincount(vMAPp, minlength=P).astype(float)
        sum_beta_p = np.bincount(vMAPp, weights=phi[vMAPp] + u_map, minlength=P)
        # E[u[v]^2] = posterior second moment
        e_u2 = u_var + u_map ** 2
        sum_eu2_p = np.bincount(vMAPp, weights=e_u2, minlength=P)

        # phi posterior update
        var_phi_post = 1.0 / np.where(n_p > 0, 1.0 + n_p / np.clip(sigma2, 1e-8, None), 1.0)
        mu_phi_post = var_phi_post * sum_beta_p / np.clip(sigma2, 1e-8, None)
        phi = np.where(n_p > 0, mu_phi_post, 0.0)

        # sigma2 MAP update
        sigma2 = np.where(
            n_p > 0,
            np.maximum((1.0 + 0.5 * sum_eu2_p) / (1.0 + 0.5 * n_p + 1.0), 1e-6),
            1.0,
        )

        # --- Update eps2 (vectorised) ---
        beta_map_v = phi[vMAPp] + u_map
        resid = np.where(nan_mask, m - b_map[:, None] - beta_map_v[:, None] * t_arr, 0.0)
        n_obs_per_v = nan_mask.sum(axis=1).astype(float)
        ssr_per_v = (resid ** 2).sum(axis=1)
        for k in range(M):
            mask_k = vMAPm == k
            ssr = float(ssr_per_v[mask_k].sum())
            n_total = int(n_obs_per_v[mask_k].sum())
            if n_total > 0:
                eps2[k] = (1.0 + 0.5 * ssr) / (1.0 + 0.5 * n_total + 1.0)

    beta_mean = phi[vMAPp] + u_map
    beta_sd = np.sqrt(np.clip(u_var, 1e-12, None))

    # Position-level posterior SD (vectorised)
    n_per_pos = np.bincount(vMAPp, minlength=P).astype(float)
    var_phi_post = 1.0 / np.where(
        n_per_pos > 0,
        1.0 + n_per_pos / np.clip(sigma2, 1e-8, None),
        1.0,
    )
    phi_sd = np.sqrt(var_phi_post)

    return beta_mean, beta_sd, phi, phi_sd


def _run_rosace_scipy(
    stan_data: dict,
    method: str,
    var_names: list,
    assay_name: str,
    assay_type: str,
    var_info: Optional[pd.DataFrame],
    seed: Optional[int],
) -> Score:
    """Pure-Python MAP + Laplace approximation backend for ROSACE models.

    Implements ROSACE0 and ROSACE1 via coordinate-ascent MAP optimisation.
    Supports all four method strings (ROSACE0/1/2/3) but ROSACE2/3 fall back
    to ROSACE1 (BLOSUM grouping is ignored in this backend).
    """
    V = stan_data["V"]
    T = stan_data["T"]
    M = stan_data["M"]
    t_arr = np.array(stan_data["t"], dtype=float)
    m = np.array(stan_data["m"], dtype=float)  # (V, T)
    vMAPm = np.array(stan_data["vMAPm"], dtype=int) - 1  # convert to 0-indexed

    if method == "ROSACE0":
        beta_mean, beta_sd = _run_rosace0_scipy(m, t_arr, vMAPm, M, V)
        mu, sd = beta_mean, beta_sd
        optional_score = None
    else:
        # ROSACE1/2/3: use position-level model (ROSACE1 approximation)
        P = stan_data["P"]
        vMAPp = np.array(stan_data["vMAPp"], dtype=int) - 1  # 0-indexed

        beta_mean, beta_sd, phi_mean, phi_sd = _run_rosace1_scipy(
            m, t_arr, vMAPm, vMAPp, M, V, P
        )
        mu, sd = beta_mean, beta_sd

        # Recover position order from var_info
        if var_info is not None:
            vi = var_info.set_index("variant")
            pos_order: list = []
            seen: set = set()
            for v in var_names:
                if v in vi.index:
                    pos = vi.loc[v, "pos"]
                    if pos not in seen:
                        pos_order.append(pos)
                        seen.add(pos)
        else:
            pos_order = list(range(P))

        optional_score = pd.DataFrame({
            "pos": pos_order,
            "mean": phi_mean,
            "sd": phi_sd,
        })

    lfsr_neg = scipy.stats.norm.sf(0, loc=mu, scale=sd + 1e-10)
    lfsr_pos = scipy.stats.norm.cdf(0, loc=mu, scale=sd + 1e-10)
    lfsr = np.minimum(lfsr_neg, lfsr_pos)

    score_df = pd.DataFrame({
        "variant": var_names,
        "mean": mu,
        "sd": sd,
        "lfsr": lfsr,
    })

    return Score(
        method=method,
        type=assay_type,
        assay_name=assay_name,
        score=score_df,
        optional_score=optional_score,
    )


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
    backend: str = "auto",
) -> Score:
    """Run ROSACE Bayesian inference and return a Score object.

    Parameters
    ----------
    assay:
        AssayGrowth or AssaySetGrowth with norm_counts populated.
    method:
        Model to use: ROSACE0, ROSACE1, ROSACE2, or ROSACE3.
    t:
        Time vector.
    n_mean_groups:
        Number of mean count groups.
    var_info:
        DataFrame with variant metadata (required for ROSACE1/2/3).
    chains:
        Number of MCMC chains (cmdstan backend only).
    parallel_chains:
        Number of chains to run in parallel (cmdstan backend only).
    iter_warmup:
        Number of warmup iterations per chain (cmdstan backend only).
    iter_sampling:
        Number of sampling iterations per chain (cmdstan backend only).
    seed:
        Random seed for reproducibility.
    backend:
        Inference backend.  One of:

        * ``"auto"`` (default) – use ``"cmdstan"`` when CmdStan is installed,
          otherwise fall back to ``"scipy"``.
        * ``"cmdstan"`` – full MCMC via CmdStanPy (requires CmdStan).
        * ``"scipy"`` – MAP + Laplace approximation using pure NumPy/SciPy.
          No external compiler required; suitable for interactive exploration
          and notebooks.  ROSACE2/3 methods are treated as ROSACE1 (BLOSUM
          grouping is not applied).

    Returns
    -------
    Score
        Score object with posterior mean, sd, and LFSR per variant.
    """
    if method not in STAN_MODELS:
        raise ValueError(f"Unknown method {method!r}. Choose from {list(STAN_MODELS)}")

    if backend == "auto":
        backend = "cmdstan" if _cmdstan_available() else "scipy"

    if isinstance(assay, AssayGrowth):
        var_names = assay.norm_var_names or assay.var_names
        assay_name = assay.key
        assay_type = "AssayGrowth"
    else:
        var_names = assay.var_names
        assay_name = assay.key
        assay_type = "AssaySetGrowth"

    stan_data = gen_rosace_input(
        assay=assay,
        method=method,
        t=t,
        n_mean_groups=n_mean_groups,
        var_info=var_info,
    )

    if backend == "scipy":
        return _run_rosace_scipy(
            stan_data=stan_data,
            method=method,
            var_names=var_names,
            assay_name=assay_name,
            assay_type=assay_type,
            var_info=var_info,
            seed=seed,
        )

    # --- CmdStan backend ---
    try:
        import cmdstanpy
    except ImportError as exc:
        raise ImportError(
            "cmdstanpy is required for the cmdstan backend. "
            "Install it with: pip install cmdstanpy, then run "
            "cmdstanpy.install_cmdstan().  Alternatively, pass backend='scipy'."
        ) from exc

    stan_code = STAN_MODELS[method]

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
