"""Functions to prepare input and run ROSACE Stan models."""

from __future__ import annotations
import importlib.resources
from typing import Optional, Union
import numpy as np
import pandas as pd
import scipy.stats

from rosace.assay import AssayGrowth, AssaySetGrowth
from rosace.score import Score
from rosace.utils import map_blosum_score, compute_blosum_groups

# Map method names to bundled .stan file names.
_STAN_FILE_MAP = {
    "ROSACE0": "growth_nopos.stan",
    "ROSACE1": "growth_pos.stan",
    "ROSACE2_NOSYN": "growth_pos_blosum_nosyn.stan",
    "ROSACE2_SYN": "growth_pos_blosum.stan",
    "ROSACE3_NOSYN": "growth_pos_blosum_act_nosyn.stan",
    "ROSACE3_SYN": "growth_pos_blosum_act.stan",
}


def _build_position_index_map(
    pos_labels: list[int],
    thred: int = 10,
    ctrl_labels: Optional[list[bool]] = None,
    stop_labels: Optional[list[bool]] = None,
) -> tuple[list[int], int, int]:
    """Map variant positions to grouped position indices, mirroring R varPosIndexMap.

    Positions are sorted by position label, then grouped into consecutive
    index bins whose cumulative variant count reaches ``thred``. Optional
    control and stop labels are assigned to dedicated trailing indices.
    """
    if len(pos_labels) == 0:
        return [], 0, 0

    if ctrl_labels is None:
        ctrl_labels = [False] * len(pos_labels)
    if stop_labels is None:
        stop_labels = [False] * len(pos_labels)

    # Count variants per position and process positions in sorted order.
    counts: dict[int, int] = {}
    for p in pos_labels:
        counts[p] = counts.get(p, 0) + 1
    sorted_pos = sorted(counts.keys())

    pos_to_index: dict[int, int] = {}
    curr_index = 1
    counter = 0
    for p in sorted_pos:
        pos_to_index[p] = curr_index
        counter += counts[p]
        if counter >= thred:
            counter = 0
            curr_index += 1

    # Merge trailing small group into previous group when possible.
    if counter < thred and curr_index > 1:
        trailing_index = curr_index
        merge_to = curr_index - 1
        for p in sorted_pos:
            if pos_to_index[p] == trailing_index:
                pos_to_index[p] = merge_to

    vmap = [int(pos_to_index[p]) for p in pos_labels]
    base_group_max = max(vmap)
    n_syn_group = max(counts.values()) - 1

    def _assign_special_group(labels: list[bool], curr_idx: int) -> int:
        nonlocal vmap
        if n_syn_group <= 0:
            n_group = 1
        else:
            n_group = n_syn_group
        label_indices = [i for i, b in enumerate(labels) if b]
        if not label_indices:
            return curr_idx

        # R: for (i in which(label)[order(pos[label])]) ...
        label_indices = sorted(label_indices, key=lambda i: pos_labels[i])
        ctr = 0
        for i in label_indices:
            vmap[i] = curr_idx
            ctr += 1
            if ctr >= n_group:
                ctr = 0
                curr_idx += 1

        if (ctr < thred) and (len(label_indices) >= thred):
            trailing = curr_idx
            for j, g in enumerate(vmap):
                if g == trailing:
                    vmap[j] = trailing - 1
        else:
            curr_idx += 1
        return curr_idx

    curr_index = base_group_max + 1
    curr_index = _assign_special_group(ctrl_labels, curr_index)
    curr_index = _assign_special_group(stop_labels, curr_index)

    p_syn = 0
    ctrl_groups = {vmap[i] for i, b in enumerate(ctrl_labels) if b}
    stop_groups = {vmap[i] for i, b in enumerate(stop_labels) if b}
    p_syn += len(ctrl_groups)
    p_syn += len(stop_groups)

    return vmap, int(max(vmap)), int(p_syn)


def gen_rosace_input(
    assay: Union[AssayGrowth, AssaySetGrowth],
    method: str = "ROSACE1",
    t: Optional[list[float]] = None,
    n_mean_groups: int = 5,
    var_info: Optional[pd.DataFrame] = None,
    matrix: str = "BLOSUM90",
    pos_group_threshold: int = 10,
) -> dict:
    """Generate the Stan data dictionary for a ROSACE growth model.

    Parameters
    ----------
    assay:
        AssayGrowth or AssaySetGrowth with norm_counts populated.
    method:
        One of ROSACE0, ROSACE1, ROSACE2, ROSACE3.
    t:
        Time vector. If None, generates a normalised time vector matching the
        R implementation: ``seq(0, rounds)/rounds`` for a single replicate, or
        ``seq(0, r_i)/max(rounds)`` for each replicate ``i`` in a set.
    n_mean_groups:
        Fallback number of quantile groups for vMAPm when raw counts are
        unavailable. When raw counts are present (the normal case), vMAPm is
        computed from raw counts as ``ceiling(rank(rowSums)/25)``, matching R.
    var_info:
        DataFrame with columns ``variant``, ``pos``, ``wt``, ``mut``.
        Required for ROSACE1/2/3 (position-level grouping).
        For ROSACE2/3, also needs ``wt`` and ``mut`` for substitution matrix
        grouping.
    matrix:
        Name of the BioPython substitution matrix used to assign amino acid
        groups for ROSACE2/3.  Defaults to ``"BLOSUM90"`` to match the R
        rosaceAA package.
    pos_group_threshold:
        Position-grouping threshold used when building ``vMAPp``. Matches the
        R default ``thred = 10``.

    Returns
    -------
    dict
        Stan data dictionary ready to pass to CmdStanPy.
    """
    if isinstance(assay, AssayGrowth):
        norm_counts = assay.norm_counts
        var_names = assay.norm_var_names or assay.var_names
        T = assay.counts.shape[1]
        rounds = assay.rounds  # T - 1
        # Raw counts for the normalized variants (wt may have been removed)
        if assay.norm_var_names is not None:
            _name_to_raw_idx = {v: i for i, v in enumerate(assay.var_names)}
            raw_for_norm = assay.counts[
                [_name_to_raw_idx[v] for v in var_names if v in _name_to_raw_idx], :
            ]
        else:
            raw_for_norm = assay.counts
    else:
        norm_counts = assay.combined_counts
        var_names = assay.var_names
        T = assay.combined_counts.shape[1]
        rounds = assay.rounds  # list of per-replicate rounds
        raw_for_norm = assay.raw_counts if assay.raw_counts is not None else None
        # R GenRosaceInput.AssaySetGrowth applies KNN imputation to combined
        # counts before building Stan input (imputeAssaysCountKNN).
        if np.isnan(norm_counts).any():
            from sklearn.impute import KNNImputer

            norm_counts = KNNImputer(n_neighbors=10).fit_transform(norm_counts)

    if norm_counts is None:
        raise ValueError("assay must have norm_counts populated.")

    V = len(var_names)

    # Default time vector: match R normalisation
    # AssayGrowth:    t = seq(0, rounds)/rounds     → [0, 1/rounds, …, 1]
    # AssaySetGrowth: t = seq(0, r_i)/max(rounds)   for each replicate i
    if t is None:
        if isinstance(assay, AssayGrowth):
            denom = rounds if rounds > 0 else 1
            t = [i / denom for i in range(T)]
        else:
            r_list = rounds if rounds is not None else [T - 1]
            max_r = max(r_list) if r_list else 1
            t = [i / max_r for r in r_list for i in range(r + 1)]

    # Replace any remaining NaN with 0 in norm_counts
    m = np.nan_to_num(norm_counts, nan=0.0)

    # Mean count groups (vMAPm): match R — ceiling(rank(rowSums(raw.counts))/25)
    # Groups variants by total raw count; each group has ~25 variants.
    if raw_for_norm is not None and raw_for_norm.shape[0] == V:
        row_sums = np.nansum(raw_for_norm, axis=1)
        ranks = scipy.stats.rankdata(row_sums)  # 1-indexed fractional ranks
        vMAPm = np.ceil(ranks / 25).astype(int)
    else:
        # Fallback: quantile-based grouping on abs normalised counts
        mean_abs = np.abs(m).mean(axis=1)
        quantiles = np.percentile(mean_abs, np.linspace(0, 100, n_mean_groups + 1))
        vMAPm = np.ceil(
            np.searchsorted(quantiles[1:-1], mean_abs, side="right") + 1
        ).astype(int)
    M = int(vMAPm.max())

    data: dict = {
        "T": T,
        "V": V,
        "M": M,
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

    pos_labels: list[int] = []
    for v in var_names:
        if v not in vi.index:
            raise KeyError(f"Variant {v!r} not found in var_info")
        pos_labels.append(int(vi.loc[v, "pos"]))

    ctrl_labels = None
    stop_labels = None
    if "is_ctrl" in vi.columns:
        ctrl_labels = [bool(vi.loc[v, "is_ctrl"]) for v in var_names]
    if "is_stop" in vi.columns:
        stop_labels = [bool(vi.loc[v, "is_stop"]) for v in var_names]

    vMAPp, P, P_syn = _build_position_index_map(
        pos_labels=pos_labels,
        thred=pos_group_threshold,
        ctrl_labels=ctrl_labels,
        stop_labels=stop_labels,
    )

    data["P"] = P
    data["vMAPp"] = vMAPp

    if method == "ROSACE1":
        return data

    # ROSACE2/3: BLOSUM grouping via coverage-based merging (mirrors R)
    # Requires wt and mut columns in var_info
    wt_list = [str(vi.loc[v, "wt"]) for v in var_names]
    mut_list = [str(vi.loc[v, "mut"]) for v in var_names]

    # R helperRunRosaceGrowth default: if ctrl labels are not provided and
    # BLOSUM is available, synonymous substitutions are controls.
    if ctrl_labels is None:
        ctrl_labels = [wt == mut for wt, mut in zip(wt_list, mut_list)]
    if stop_labels is None:
        stop_labels = [False] * len(var_names)

    vMAPb, B, blosum_count = compute_blosum_groups(
        wt_list=wt_list,
        mut_list=mut_list,
        pos_index_list=vMAPp,  # use position indices for coverage calculation
        coverage_threshold=0.2,
        matrix=matrix,
    )

    data["B"] = B
    data["vMAPb"] = vMAPb
    data["blosum_count"] = blosum_count
    data["P_syn"] = int(P_syn)

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
    matrix: str = "BLOSUM90",
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
    matrix:
        Name of the BioPython substitution matrix for ROSACE2/3 amino acid
        grouping.  Defaults to ``"BLOSUM90"`` to match the R rosaceAA package.

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

    if method not in {"ROSACE0", "ROSACE1", "ROSACE2", "ROSACE3"}:
        raise ValueError("Unknown method. Choose from ['ROSACE0', 'ROSACE1', 'ROSACE2', 'ROSACE3']")

    stan_data = gen_rosace_input(
        assay=assay,
        method=method,
        t=t,
        n_mean_groups=n_mean_groups,
        var_info=var_info,
        matrix=matrix,
    )

    if isinstance(assay, AssayGrowth):
        var_names = assay.norm_var_names or assay.var_names
        assay_name = assay.key
        assay_type = "AssayGrowth"
    else:
        var_names = assay.var_names
        assay_name = assay.key
        assay_type = "AssaySetGrowth"

    vi = var_info.set_index("variant") if var_info is not None else None

    if method == "ROSACE2":
        stan_key = "ROSACE2_SYN" if int(stan_data.get("P_syn", 0)) > 0 else "ROSACE2_NOSYN"
    elif method == "ROSACE3":
        stan_key = "ROSACE3_SYN" if int(stan_data.get("P_syn", 0)) > 0 else "ROSACE3_NOSYN"
    else:
        stan_key = method

    stan_file_ref = importlib.resources.files("rosace") / "stan" / _STAN_FILE_MAP[stan_key]
    with importlib.resources.as_file(stan_file_ref) as stan_path:
        model = cmdstanpy.CmdStanModel(stan_file=str(stan_path))
        fit = model.sample(
            data=stan_data,
            chains=chains,
            parallel_chains=parallel_chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            show_progress=False,
        )

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
    misc: dict = {}
    pos_order: list[int] = []
    if method in ("ROSACE1", "ROSACE2", "ROSACE3") and vi is not None:
        seen: set[int] = set()
        for v in var_names:
            if v in vi.index:
                pos = int(vi.loc[v, "pos"])
                if pos not in seen:
                    pos_order.append(pos)
                    seen.add(pos)

        phi_samples = fit.stan_variable("phi")  # (draws, P)
        phi_mu = phi_samples.mean(axis=0)
        phi_sd = phi_samples.std(axis=0)
        sigma2_samples = fit.stan_variable("sigma2")  # (draws, P)
        sigma2_mu = sigma2_samples.mean(axis=0)
        sigma2_sd = sigma2_samples.std(axis=0)
        optional_score = pd.DataFrame({
            "pos": pos_order,
            "mean": phi_mu,  # backwards-compatible aliases
            "sd": phi_sd,
            "phi_mean": phi_mu,
            "phi_sd": phi_sd,
            "sigma2_mean": sigma2_mu,
            "sigma2_sd": sigma2_sd,
        })
        score_df["pos"] = [int(vi.loc[v, "pos"]) for v in var_names]

        if method in ("ROSACE2", "ROSACE3") and {"wt", "mut"}.issubset(vi.columns):
            score_df["wt"] = [str(vi.loc[v, "wt"]) for v in var_names]
            score_df["mut"] = [str(vi.loc[v, "mut"]) for v in var_names]
            score_df["blosum_score"] = [
                map_blosum_score(str(vi.loc[v, "wt"]), str(vi.loc[v, "mut"]), matrix=matrix)
                for v in var_names
            ]

    # BLOSUM group (nu) scores for ROSACE2/3
    if method in ("ROSACE2", "ROSACE3"):
        if "vMAPb" in stan_data:
            score_df["blosum_group"] = np.array(stan_data["vMAPb"], dtype=int)

        nu_samples = fit.stan_variable("nu")  # (draws, B)
        nu_mu = nu_samples.mean(axis=0)
        nu_sd = nu_samples.std(axis=0)
        misc["blosum_scores"] = pd.DataFrame({
            "blosum_group": list(range(1, len(nu_mu) + 1)),
            "mean": nu_mu,
            "sd": nu_sd,
        })
        if "blosum_group" in score_df.columns:
            nu_df = misc["blosum_scores"].rename(
                columns={"mean": "nu_mean", "sd": "nu_sd"}
            )
            score_df = score_df.merge(nu_df, on="blosum_group", how="left")

    # Per-position activation fraction rho for ROSACE3 (vector of length P)
    if method == "ROSACE3":
        rho_samples = fit.stan_variable("rho")  # (draws, P)
        rho_mu = rho_samples.mean(axis=0)
        rho_sd = rho_samples.std(axis=0)
        misc["rho_scores"] = pd.DataFrame({
            "pos": pos_order,
            "rho_mean": rho_mu,
            "rho_sd": rho_sd,
        })
        if "pos" in score_df.columns:
            rho_lookup = {
                int(p): (float(rm), float(rs))
                for p, rm, rs in zip(pos_order, rho_mu, rho_sd)
            }
            rho_vals = [rho_lookup.get(int(p), (np.nan, np.nan)) for p in score_df["pos"]]
            score_df["rho_mean"] = [r[0] for r in rho_vals]
            score_df["rho_sd"] = [r[1] for r in rho_vals]

    return Score(
        method=method,
        type=assay_type,
        assay_name=assay_name,
        score=score_df,
        optional_score=optional_score,
        misc=misc,
    )
