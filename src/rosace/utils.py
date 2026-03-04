"""Utility functions for rosace."""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import scipy.stats

from rosace.assay import AssayGrowth, AssaySetGrowth

# Cache for loaded substitution matrices (avoids repeated disk reads)
_MATRIX_CACHE: dict = {}

_DEFAULT_MATRIX = "BLOSUM90"


def _load_matrix(matrix_name: str):
    """Load a substitution matrix from BioPython, with caching.

    Parameters
    ----------
    matrix_name:
        Name of the substitution matrix (e.g. ``"BLOSUM90"``, ``"BLOSUM62"``).
        Any matrix available in ``Bio.Align.substitution_matrices`` is accepted.

    Returns
    -------
    Bio.Align.substitution_matrices.Array
        The loaded substitution matrix, cached for subsequent calls.
    """
    if matrix_name not in _MATRIX_CACHE:
        from Bio.Align import substitution_matrices
        _MATRIX_CACHE[matrix_name] = substitution_matrices.load(matrix_name)
    return _MATRIX_CACHE[matrix_name]


_TRIPLE_TO_SINGLE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def map_blosum_score(
    wt: str,
    mut: str,
    aa_code: str = "single",
    matrix: str = _DEFAULT_MATRIX,
) -> int:
    """Return the substitution matrix score for a wildtype/mutant pair, capped at 5.

    Matches the R rosaceAA ``MapBlosumScore`` function which uses BLOSUM90 and
    caps the return value at 5 (``min(blosum90[wt, mut], 5)``).  All synonymous
    substitutions (diagonal entries ≥ 5) therefore map to 5.  Missense
    substitution scores are typically in the range −6 to +3.

    Parameters
    ----------
    wt:
        Wildtype amino acid (single-letter code).
    mut:
        Mutant amino acid (single-letter code).
    aa_code:
        Amino acid code type: ``"single"`` or ``"triple"``.
    matrix:
        Name of the BioPython substitution matrix to use.  Defaults to
        ``"BLOSUM90"`` to match the R rosaceAA package.  Any matrix available
        in ``Bio.Align.substitution_matrices`` is accepted (e.g. ``"BLOSUM62"``).

    Returns
    -------
    int
        Substitution matrix score, capped at 5.
    """
    wt = str(wt).upper()
    mut = str(mut).upper()
    if aa_code == "triple":
        wt = _TRIPLE_TO_SINGLE.get(wt, wt)
        mut = _TRIPLE_TO_SINGLE.get(mut, mut)
    elif aa_code != "single":
        raise ValueError("Invalid aa_code. Use 'single' or 'triple'.")

    if mut == "DEL":
        return -7
    if mut.startswith("INS"):
        return -8

    mat = _load_matrix(matrix)
    try:
        score = mat[wt, mut]
    except (KeyError, IndexError):
        return -9
    return int(min(score, 5))


def compute_blosum_groups(
    wt_list: list,
    mut_list: list,
    pos_index_list: list,
    coverage_threshold: float = 0.2,
    matrix: str = _DEFAULT_MATRIX,
) -> tuple:
    """Compute substitution-matrix group assignments with rare-group merging.

    Mirrors the R ``GenRosaceInput`` BLOSUM group computation:

    1. Map each (wt, mut) pair to a substitution matrix score via
       ``map_blosum_score``.
    2. Merge groups whose position coverage (fraction of positions containing
       at least one variant in the group) is below *coverage_threshold*,
       iterating from the most-negative to the second-most-positive score,
       merging each rare group into the adjacent higher-scoring group.
    3. Dense-rank the final groups to produce contiguous 1-based integers.

    Parameters
    ----------
    wt_list:
        Wildtype amino acids, one per variant.
    mut_list:
        Mutant amino acids, one per variant.
    pos_index_list:
        Position index (1-based ``vMAPp`` values) for each variant, used to
        compute per-group position coverage.
    coverage_threshold:
        Groups whose coverage is below this value are merged with the next
        group.  Default matches R's ``cove = 0.2``.
    matrix:
        Name of the BioPython substitution matrix to use.  Defaults to
        ``"BLOSUM90"`` to match the R rosaceAA package.

    Returns
    -------
    vMAPb : list[int]
        1-based substitution-matrix group index for each variant.
    B : int
        Total number of groups after merging.
    blosum_count : list[float]
        Count of variants in each group (length *B*).
    """
    # Step 1: get raw substitution matrix scores
    scores = np.array(
        [map_blosum_score(wt, mut, matrix=matrix) for wt, mut in zip(wt_list, mut_list)],
        dtype=float,
    )
    pos_arr = np.array(pos_index_list)
    n_pos = len(set(pos_arr))

    # Step 2: rare-group merging — mirrors R's for-loop over sorted unique labels
    original_labels = sorted(set(scores.tolist()))  # original sorted unique scores
    removed_indices: list[int] = []  # 0-based indices of merged-away labels

    for i in range(len(original_labels) - 1):  # iterate up to second-to-last (R: 1:(len-1))
        label_i = original_labels[i]
        # Coverage uses the CURRENT (possibly already merged) scores array
        positions_with = set(pos_arr[scores == label_i].tolist())
        coverage = len(positions_with) / n_pos if n_pos > 0 else 0.0

        if coverage < coverage_threshold:
            if i < len(original_labels) - 2:
                # Not the second-to-last: merge into next original label
                scores[scores == label_i] = original_labels[i + 1]
                removed_indices.append(i)
            else:
                # Second-to-last: merge into the second-to-last of the
                # remaining labels (mirrors R's label_new[length(label_new)-1])
                removed_indices.append(i)
                remaining = [
                    original_labels[j]
                    for j in range(len(original_labels))
                    if j not in set(removed_indices)
                ]
                target = remaining[-2] if len(remaining) >= 2 else remaining[-1]
                scores[scores == label_i] = target

    # Step 3: dense-rank to contiguous 1-based group indices
    unique_groups = sorted(set(scores.tolist()))
    group_map = {g: k + 1 for k, g in enumerate(unique_groups)}
    vMAPb = [int(group_map[s]) for s in scores.tolist()]
    B = len(unique_groups)
    blosum_count = [float(np.sum(scores == g)) for g in unique_groups]

    return vMAPb, B, blosum_count


def estimate_disp(
    assay: AssayGrowth,
    ctrl_label: Optional[list[str]] = None,
) -> float:
    """Estimate negative binomial sequencing dispersion from control variants.

    Uses the method-of-moments estimator on the last timepoint counts relative
    to the first timepoint.

    Parameters
    ----------
    assay:
        Input AssayGrowth object.
    ctrl_label:
        List of control variant names. If None, uses all variants.

    Returns
    -------
    float
        Estimated dispersion parameter (1/r in the NB parameterisation where
        variance = mean + mean^2 * disp).
    """
    counts = assay.counts.astype(float)
    var_names = list(assay.var_names)

    if ctrl_label is not None:
        idx = [i for i, v in enumerate(var_names) if v in ctrl_label]
        if len(idx) == 0:
            raise ValueError("No ctrl_label variants found in assay.")
        counts = counts[idx, :]

    # Use last timepoint
    last = counts[:, -1]
    last = last[~np.isnan(last)]
    last = last[last > 0]

    if len(last) < 2:
        return 0.1  # fallback

    mean = np.mean(last)
    var = np.var(last, ddof=1)
    if var <= mean:
        return 1e-6  # Poisson-like, essentially 0 dispersion
    disp = (var - mean) / (mean ** 2)
    return float(disp)


def estimate_disp_start(
    assay: AssayGrowth,
) -> float:
    """Estimate library (initial timepoint) dispersion.

    Parameters
    ----------
    assay:
        Input AssayGrowth object.

    Returns
    -------
    float
        Estimated dispersion parameter for the initial counts.
    """
    counts = assay.counts.astype(float)
    first = counts[:, 0]
    first = first[~np.isnan(first)]
    first = first[first > 0]

    if len(first) < 2:
        return 0.1

    mean = np.mean(first)
    var = np.var(first, ddof=1)
    if var <= mean:
        return 1e-6
    disp = (var - mean) / (mean ** 2)
    return float(disp)


def _add_lfsr_and_labels(df: pd.DataFrame, sig_test: float) -> pd.DataFrame:
    """Add lfsr/label columns using normal-approximation tails (R-compatible)."""
    import scipy.stats as stats

    out = df.copy()
    mu = out["mean"].to_numpy(dtype=float)
    sd = out["sd"].to_numpy(dtype=float)
    safe_sd = np.where(np.isnan(sd) | (sd <= 0), 1e-10, sd)

    lfsr_neg = stats.norm.sf(0, loc=mu, scale=safe_sd)  # P(beta > 0)
    lfsr_pos = stats.norm.cdf(0, loc=mu, scale=safe_sd)  # P(beta < 0)
    lfsr = np.minimum(lfsr_neg, lfsr_pos)

    out["lfsr_neg"] = lfsr_neg
    out["lfsr_pos"] = lfsr_pos
    out["lfsr"] = lfsr
    out["test_neg"] = lfsr_neg <= (sig_test / 2)
    out["test_pos"] = lfsr_pos <= (sig_test / 2)
    out["label"] = np.where(out["test_neg"], "Neg", "Neutral")
    out["label"] = np.where(out["test_pos"], "Pos", out["label"])
    return out


def output_score(
    score: "Score",
    sig_test: float = 0.05,
    pos_info: bool = False,
    blosum_info: bool = False,
    pos_act_info: bool = False,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Compute LFSR/labels and optional ROSACE auxiliary output tables.

    Parameters
    ----------
    score:
        Score object with ``score`` DataFrame containing ``mean`` and ``sd``.
    sig_test:
        LFSR threshold for two-sided sign testing (per-tail threshold is
        ``sig_test / 2`` to match R).
    pos_info:
        If True, return position-level score table alongside variant scores.
    blosum_info:
        If True, require ROSACE2/ROSACE3 and include BLOSUM annotations in the
        returned variant table.
    pos_act_info:
        If True, require ROSACE3 and include rho annotations in the returned
        variant/position tables.

    Returns
    -------
    pd.DataFrame | dict[str, pd.DataFrame]
        Variant score table by default. If ``pos_info=True``, returns a dict:
        ``{"df_variant": ..., "df_position": ...}``.
    """
    if not str(score.method).startswith("ROSACE"):
        # Keep non-ROSACE behavior simple and backwards-compatible.
        return _add_lfsr_and_labels(score.score.copy(), sig_test=sig_test)

    model_code = int(str(score.method)[-1])  # ROSACE0..ROSACE3
    df_variant = _add_lfsr_and_labels(score.score.copy(), sig_test=sig_test)

    if blosum_info and model_code < 2:
        raise ValueError(
            "blosum_info=True requires ROSACE2 or ROSACE3 output."
        )
    if pos_act_info and model_code < 3:
        raise ValueError(
            "pos_act_info=True requires ROSACE3 output."
        )
    if pos_info and (model_code < 1 or score.optional_score is None):
        raise ValueError(
            "pos_info=True requires ROSACE1/2/3 output with optional_score."
        )

    # Validate optional variant annotations requested by flags.
    if blosum_info and "blosum_score" not in df_variant.columns and "blosum_group" not in df_variant.columns:
        raise ValueError(
            "blosum_info=True requested but no BLOSUM columns are present."
        )
    if pos_act_info and "rho_mean" not in df_variant.columns:
        raise ValueError(
            "pos_act_info=True requested but rho columns are missing."
        )

    if not pos_info:
        return df_variant

    pos_df = score.optional_score.copy()
    if "phi_mean" in pos_df.columns:
        pos_df = pos_df.rename(columns={"phi_mean": "mean", "phi_sd": "sd"})
    pos_df = pos_df.loc[:, ~pos_df.columns.duplicated()]
    required_pos_cols = {"pos", "mean", "sd"}
    missing = required_pos_cols.difference(pos_df.columns)
    if missing:
        raise ValueError(
            f"optional_score is missing required position columns: {sorted(missing)}"
        )
    pos_df = _add_lfsr_and_labels(pos_df, sig_test=sig_test)
    if not pos_act_info and "rho_mean" in pos_df.columns:
        pos_df = pos_df.drop(columns=[c for c in ("rho_mean", "rho_sd") if c in pos_df.columns])

    return {
        "df_variant": df_variant,
        "df_position": pos_df,
    }
