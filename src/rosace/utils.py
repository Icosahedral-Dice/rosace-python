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
    """
    if matrix_name not in _MATRIX_CACHE:
        from Bio.Align import substitution_matrices
        _MATRIX_CACHE[matrix_name] = substitution_matrices.load(matrix_name)
    return _MATRIX_CACHE[matrix_name]


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
        Amino acid code type (only ``"single"`` is currently supported).
    matrix:
        Name of the BioPython substitution matrix to use.  Defaults to
        ``"BLOSUM90"`` to match the R rosaceAA package.  Any matrix available
        in ``Bio.Align.substitution_matrices`` is accepted (e.g. ``"BLOSUM62"``).

    Returns
    -------
    int
        Substitution matrix score, capped at 5.
    """
    if aa_code != "single":
        raise NotImplementedError("Only single-letter amino acid codes are supported.")
    wt = wt.upper()
    mut = mut.upper()
    mat = _load_matrix(matrix)
    try:
        score = mat[wt, mut]
    except (KeyError, IndexError):
        raise ValueError(f"Unknown amino acid pair: ({wt!r}, {mut!r})")
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


def output_score(
    score: "Score",
    sig_test: float = 0.05,
) -> pd.DataFrame:
    """Compute LFSR and significance labels from a Score object.

    Parameters
    ----------
    score:
        Score object with a ``score`` DataFrame containing at least
        ``mean`` and ``sd`` columns.
    sig_test:
        Significance threshold for LFSR.

    Returns
    -------
    pd.DataFrame
        Copy of score.score with additional columns: ``lfsr``, ``label``.
        label is one of "Neutral", "Neg", "Pos".
    """
    import scipy.stats as stats

    df = score.score.copy()
    mu = df["mean"].values
    sd = df["sd"].values

    lfsr_neg = stats.norm.sf(0, loc=mu, scale=sd)   # P(beta > 0)
    lfsr_pos = stats.norm.cdf(0, loc=mu, scale=sd)   # P(beta < 0)
    lfsr = np.minimum(lfsr_neg, lfsr_pos)

    df["lfsr"] = lfsr
    label = np.where(
        lfsr >= sig_test / 2,
        "Neutral",
        np.where(mu > 0, "Pos", "Neg"),
    )
    df["label"] = label
    return df
