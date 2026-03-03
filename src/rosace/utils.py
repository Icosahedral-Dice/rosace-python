"""Utility functions for rosace."""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import scipy.stats

from rosace.assay import AssayGrowth, AssaySetGrowth

# BLOSUM90 matrix (single-letter amino acid codes)
# Sourced from NCBI BLOSUM90. Matches the matrix used in the R rosaceAA package.
# AA order: ARNDCQEGHILKMFPSTWYV
_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"

_BLOSUM90_FLAT = {
    ('A','A'): 5, ('A','R'):-2, ('A','N'):-2, ('A','D'):-3, ('A','C'):-1,
    ('A','Q'):-1, ('A','E'):-1, ('A','G'): 0, ('A','H'):-2, ('A','I'):-2,
    ('A','L'):-2, ('A','K'):-1, ('A','M'):-2, ('A','F'):-3, ('A','P'):-1,
    ('A','S'): 1, ('A','T'): 0, ('A','W'):-4, ('A','Y'):-3, ('A','V'):-1,
    ('R','A'):-2, ('R','R'): 6, ('R','N'):-1, ('R','D'):-3, ('R','C'):-5,
    ('R','Q'): 1, ('R','E'):-1, ('R','G'):-3, ('R','H'): 0, ('R','I'):-4,
    ('R','L'):-3, ('R','K'): 2, ('R','M'):-2, ('R','F'):-4, ('R','P'):-3,
    ('R','S'):-1, ('R','T'):-2, ('R','W'):-4, ('R','Y'):-3, ('R','V'):-3,
    ('N','A'):-2, ('N','R'):-1, ('N','N'): 7, ('N','D'): 1, ('N','C'):-4,
    ('N','Q'): 0, ('N','E'):-1, ('N','G'): 0, ('N','H'): 0, ('N','I'):-4,
    ('N','L'):-4, ('N','K'): 0, ('N','M'):-3, ('N','F'):-4, ('N','P'):-3,
    ('N','S'): 1, ('N','T'): 0, ('N','W'):-5, ('N','Y'):-3, ('N','V'):-4,
    ('D','A'):-3, ('D','R'):-3, ('D','N'): 1, ('D','D'): 7, ('D','C'):-5,
    ('D','Q'):-1, ('D','E'): 1, ('D','G'):-2, ('D','H'):-2, ('D','I'):-5,
    ('D','L'):-5, ('D','K'):-1, ('D','M'):-4, ('D','F'):-5, ('D','P'):-2,
    ('D','S'):-1, ('D','T'):-2, ('D','W'):-6, ('D','Y'):-4, ('D','V'):-4,
    ('C','A'):-1, ('C','R'):-5, ('C','N'):-4, ('C','D'):-5, ('C','C'): 9,
    ('C','Q'):-4, ('C','E'):-6, ('C','G'):-4, ('C','H'):-5, ('C','I'):-2,
    ('C','L'):-2, ('C','K'):-4, ('C','M'):-2, ('C','F'):-3, ('C','P'):-4,
    ('C','S'):-2, ('C','T'):-2, ('C','W'):-4, ('C','Y'):-4, ('C','V'):-2,
    ('Q','A'):-1, ('Q','R'): 1, ('Q','N'): 0, ('Q','D'):-1, ('Q','C'):-4,
    ('Q','Q'): 7, ('Q','E'): 2, ('Q','G'):-3, ('Q','H'): 1, ('Q','I'):-4,
    ('Q','L'):-3, ('Q','K'): 1, ('Q','M'): 0, ('Q','F'):-4, ('Q','P'):-2,
    ('Q','S'): 0, ('Q','T'):-1, ('Q','W'):-3, ('Q','Y'):-2, ('Q','V'):-3,
    ('E','A'):-1, ('E','R'):-1, ('E','N'):-1, ('E','D'): 1, ('E','C'):-6,
    ('E','Q'): 2, ('E','E'): 6, ('E','G'):-3, ('E','H'):-1, ('E','I'):-4,
    ('E','L'):-4, ('E','K'): 0, ('E','M'):-3, ('E','F'):-5, ('E','P'):-2,
    ('E','S'):-1, ('E','T'):-1, ('E','W'):-5, ('E','Y'):-4, ('E','V'):-3,
    ('G','A'): 0, ('G','R'):-3, ('G','N'): 0, ('G','D'):-2, ('G','C'):-4,
    ('G','Q'):-3, ('G','E'):-3, ('G','G'): 6, ('G','H'):-3, ('G','I'):-5,
    ('G','L'):-5, ('G','K'):-2, ('G','M'):-4, ('G','F'):-5, ('G','P'):-3,
    ('G','S'):-1, ('G','T'):-3, ('G','W'):-4, ('G','Y'):-5, ('G','V'):-4,
    ('H','A'):-2, ('H','R'): 0, ('H','N'): 0, ('H','D'):-2, ('H','C'):-5,
    ('H','Q'): 1, ('H','E'):-1, ('H','G'):-3, ('H','H'): 8, ('H','I'):-4,
    ('H','L'):-4, ('H','K'):-1, ('H','M'):-3, ('H','F'):-2, ('H','P'):-3,
    ('H','S'):-2, ('H','T'):-2, ('H','W'):-3, ('H','Y'): 1, ('H','V'):-4,
    ('I','A'):-2, ('I','R'):-4, ('I','N'):-4, ('I','D'):-5, ('I','C'):-2,
    ('I','Q'):-4, ('I','E'):-4, ('I','G'):-5, ('I','H'):-4, ('I','I'): 5,
    ('I','L'): 1, ('I','K'):-4, ('I','M'): 1, ('I','F'):-1, ('I','P'):-4,
    ('I','S'):-3, ('I','T'):-1, ('I','W'):-4, ('I','Y'):-2, ('I','V'): 3,
    ('L','A'):-2, ('L','R'):-3, ('L','N'):-4, ('L','D'):-5, ('L','C'):-2,
    ('L','Q'):-3, ('L','E'):-4, ('L','G'):-5, ('L','H'):-4, ('L','I'): 1,
    ('L','L'): 5, ('L','K'):-3, ('L','M'): 2, ('L','F'): 0, ('L','P'):-4,
    ('L','S'):-3, ('L','T'):-2, ('L','W'):-3, ('L','Y'):-2, ('L','V'): 0,
    ('K','A'):-1, ('K','R'): 2, ('K','N'): 0, ('K','D'):-1, ('K','C'):-4,
    ('K','Q'): 1, ('K','E'): 0, ('K','G'):-2, ('K','H'):-1, ('K','I'):-4,
    ('K','L'):-3, ('K','K'): 6, ('K','M'):-2, ('K','F'):-4, ('K','P'):-2,
    ('K','S'):-1, ('K','T'):-1, ('K','W'):-5, ('K','Y'):-3, ('K','V'):-3,
    ('M','A'):-2, ('M','R'):-2, ('M','N'):-3, ('M','D'):-4, ('M','C'):-2,
    ('M','Q'): 0, ('M','E'):-3, ('M','G'):-4, ('M','H'):-3, ('M','I'): 1,
    ('M','L'): 2, ('M','K'):-2, ('M','M'): 7, ('M','F'):-1, ('M','P'):-3,
    ('M','S'):-2, ('M','T'):-1, ('M','W'):-2, ('M','Y'):-2, ('M','V'): 0,
    ('F','A'):-3, ('F','R'):-4, ('F','N'):-4, ('F','D'):-5, ('F','C'):-3,
    ('F','Q'):-4, ('F','E'):-5, ('F','G'):-5, ('F','H'):-2, ('F','I'):-1,
    ('F','L'): 0, ('F','K'):-4, ('F','M'):-1, ('F','F'): 7, ('F','P'):-4,
    ('F','S'):-3, ('F','T'):-3, ('F','W'): 0, ('F','Y'): 3, ('F','V'):-2,
    ('P','A'):-1, ('P','R'):-3, ('P','N'):-3, ('P','D'):-2, ('P','C'):-4,
    ('P','Q'):-2, ('P','E'):-2, ('P','G'):-3, ('P','H'):-3, ('P','I'):-4,
    ('P','L'):-4, ('P','K'):-2, ('P','M'):-3, ('P','F'):-4, ('P','P'): 8,
    ('P','S'):-2, ('P','T'):-2, ('P','W'):-5, ('P','Y'):-4, ('P','V'):-3,
    ('S','A'): 1, ('S','R'):-1, ('S','N'): 1, ('S','D'):-1, ('S','C'):-2,
    ('S','Q'): 0, ('S','E'):-1, ('S','G'):-1, ('S','H'):-2, ('S','I'):-3,
    ('S','L'):-3, ('S','K'):-1, ('S','M'):-2, ('S','F'):-3, ('S','P'):-2,
    ('S','S'): 5, ('S','T'): 1, ('S','W'):-4, ('S','Y'):-3, ('S','V'):-2,
    ('T','A'): 0, ('T','R'):-2, ('T','N'): 0, ('T','D'):-2, ('T','C'):-2,
    ('T','Q'):-1, ('T','E'):-1, ('T','G'):-3, ('T','H'):-2, ('T','I'):-1,
    ('T','L'):-2, ('T','K'):-1, ('T','M'):-1, ('T','F'):-3, ('T','P'):-2,
    ('T','S'): 1, ('T','T'): 6, ('T','W'):-4, ('T','Y'):-2, ('T','V'): 0,
    ('W','A'):-4, ('W','R'):-4, ('W','N'):-5, ('W','D'):-6, ('W','C'):-4,
    ('W','Q'):-3, ('W','E'):-5, ('W','G'):-4, ('W','H'):-3, ('W','I'):-4,
    ('W','L'):-3, ('W','K'):-5, ('W','M'):-2, ('W','F'): 0, ('W','P'):-5,
    ('W','S'):-4, ('W','T'):-4, ('W','W'):11, ('W','Y'): 2, ('W','V'):-4,
    ('Y','A'):-3, ('Y','R'):-3, ('Y','N'):-3, ('Y','D'):-4, ('Y','C'):-4,
    ('Y','Q'):-2, ('Y','E'):-4, ('Y','G'):-5, ('Y','H'): 1, ('Y','I'):-2,
    ('Y','L'):-2, ('Y','K'):-3, ('Y','M'):-2, ('Y','F'): 3, ('Y','P'):-4,
    ('Y','S'):-3, ('Y','T'):-2, ('Y','W'): 2, ('Y','Y'): 8, ('Y','V'):-3,
    ('V','A'):-1, ('V','R'):-3, ('V','N'):-4, ('V','D'):-4, ('V','C'):-2,
    ('V','Q'):-3, ('V','E'):-3, ('V','G'):-4, ('V','H'):-4, ('V','I'): 3,
    ('V','L'): 0, ('V','K'):-3, ('V','M'): 0, ('V','F'):-2, ('V','P'):-3,
    ('V','S'):-2, ('V','T'): 0, ('V','W'):-4, ('V','Y'):-3, ('V','V'): 5,
}


def map_blosum_score(wt: str, mut: str, aa_code: str = "single") -> int:
    """Return the BLOSUM90 score for a wildtype/mutant amino acid pair, capped at 5.

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

    Returns
    -------
    int
        BLOSUM90 score, capped at 5.
    """
    if aa_code != "single":
        raise NotImplementedError("Only single-letter amino acid codes are supported.")
    wt = wt.upper()
    mut = mut.upper()
    score = _BLOSUM90_FLAT.get((wt, mut))
    if score is None:
        raise ValueError(f"Unknown amino acid pair: ({wt!r}, {mut!r})")
    return min(score, 5)


def compute_blosum_groups(
    wt_list: list,
    mut_list: list,
    pos_index_list: list,
    coverage_threshold: float = 0.2,
) -> tuple:
    """Compute BLOSUM90 group assignments with rare-group merging.

    Mirrors the R ``GenRosaceInput`` BLOSUM group computation:

    1. Map each (wt, mut) pair to a BLOSUM90 score via ``map_blosum_score``.
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

    Returns
    -------
    vMAPb : list[int]
        1-based BLOSUM group index for each variant.
    B : int
        Total number of BLOSUM groups after merging.
    blosum_count : list[float]
        Count of variants in each group (length *B*).
    """
    # Step 1: get raw BLOSUM90 scores
    scores = np.array(
        [map_blosum_score(wt, mut) for wt, mut in zip(wt_list, mut_list)],
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
