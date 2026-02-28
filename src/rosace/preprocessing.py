"""Preprocessing functions for rosace assays.

Mirrors the R preprocessing functions: filter, impute, normalize, and integrate.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

from rosace.assay import AssayGrowth, AssaySetGrowth


def filter_data(
    assay: AssayGrowth,
    na_rmax: float = 0.5,
    min_count: int = 20,
) -> AssayGrowth:
    """Filter variants from an AssayGrowth based on missingness and minimum counts.

    Parameters
    ----------
    assay:
        Input AssayGrowth object.
    na_rmax:
        Maximum allowed fraction of NA values per variant (row). Variants with
        a higher NA fraction are removed.
    min_count:
        Minimum total count (summed over all non-NA timepoints) required to
        keep a variant.

    Returns
    -------
    AssayGrowth
        A new AssayGrowth with filtered variants.
    """
    counts = assay.counts.astype(float)
    var_names = list(assay.var_names)

    # Fraction of NA per variant
    na_frac = np.isnan(counts).mean(axis=1)
    keep_na = na_frac <= na_rmax

    # Total count per variant (ignoring NaN)
    total = np.nansum(counts, axis=1)
    keep_count = total >= min_count

    keep = keep_na & keep_count
    filtered_counts = counts[keep, :]
    filtered_names = [v for v, k in zip(var_names, keep) if k]

    return AssayGrowth(
        counts=filtered_counts,
        var_names=filtered_names,
        key=assay.key,
        rep=assay.rep,
        norm_counts=assay.norm_counts[keep, :] if assay.norm_counts is not None else None,
        norm_var_names=(
            [v for v, k in zip(assay.norm_var_names, keep) if k]
            if assay.norm_var_names is not None
            else None
        ),
        rounds=assay.rounds,
    )


def impute_data(
    assay: AssayGrowth,
    method: str = "zero",
) -> AssayGrowth:
    """Impute missing values in counts.

    Parameters
    ----------
    assay:
        Input AssayGrowth object.
    method:
        Imputation method.
        - ``"zero"``: Replace NA with 0.
        - ``"knn"``: K-nearest neighbours imputation using sklearn.

    Returns
    -------
    AssayGrowth
        A new AssayGrowth with imputed counts.
    """
    counts = assay.counts.astype(float).copy()

    if method == "zero":
        counts = np.where(np.isnan(counts), 0.0, counts)
    elif method == "knn":
        from sklearn.impute import KNNImputer

        imputer = KNNImputer(n_neighbors=5)
        counts = imputer.fit_transform(counts)
    else:
        raise ValueError(f"Unknown imputation method {method!r}. Use 'zero' or 'knn'.")

    return AssayGrowth(
        counts=counts,
        var_names=assay.var_names,
        key=assay.key,
        rep=assay.rep,
        norm_counts=assay.norm_counts,
        norm_var_names=assay.norm_var_names,
        rounds=assay.rounds,
    )


def normalize_data(
    assay: AssayGrowth,
    method: str = "wt",
    wt_var_names: Optional[list[str]] = None,
    wt_rm: bool = True,
) -> AssayGrowth:
    """Normalize counts to produce log-ratio growth scores.

    Parameters
    ----------
    assay:
        Input AssayGrowth object (should have been imputed before calling this).
    method:
        Normalization method.
        - ``"wt"``: Wildtype normalization. Requires ``wt_var_names``.
          Computes log(count + 0.5) - log(wt_sum + 0.5), then subtracts t=0.
        - ``"total"``: Total count normalization.
          Computes log(count + 0.5) - log(total + 0.5), then subtracts t=0.
    wt_var_names:
        Names of wildtype/synonymous control variants (required for ``method="wt"``).
    wt_rm:
        If True (default), remove wildtype variants from the output when using
        ``method="wt"``.

    Returns
    -------
    AssayGrowth
        A new AssayGrowth with ``norm_counts`` and ``norm_var_names`` populated.
    """
    counts = assay.counts.astype(float)
    var_names = list(assay.var_names)

    if method == "wt":
        if wt_var_names is None:
            raise ValueError("wt_var_names must be provided for method='wt'")
        wt_idx = [i for i, v in enumerate(var_names) if v in wt_var_names]
        if len(wt_idx) == 0:
            raise ValueError("None of wt_var_names found in assay.var_names")

        if len(wt_idx) > 1:
            wt = counts[wt_idx, :].sum(axis=0)
        else:
            wt = counts[wt_idx[0], :]

        if wt_rm:
            non_wt = [i for i in range(len(var_names)) if i not in set(wt_idx)]
            var_names_out = [var_names[i] for i in non_wt]
            mat = counts[non_wt, :]
        else:
            var_names_out = var_names
            mat = counts

        log_wt = np.log(wt + 0.5)
        log_mat = np.log(mat + 0.5)
        norm = log_mat - log_wt  # broadcast over rows
        norm = norm - norm[:, 0:1]  # subtract t=0

    elif method == "total":
        # Sum over non-NaN values per timepoint
        total = np.nansum(counts, axis=0) + 0.5
        log_total = np.log(total)
        log_mat = np.log(counts + 0.5)
        norm = log_mat - log_total
        norm = norm - norm[:, 0:1]
        var_names_out = var_names

    else:
        raise ValueError(f"Unknown normalization method {method!r}. Use 'wt' or 'total'.")

    return AssayGrowth(
        counts=assay.counts,
        var_names=assay.var_names,
        key=assay.key,
        rep=assay.rep,
        norm_counts=norm,
        norm_var_names=var_names_out,
        rounds=assay.rounds,
    )


def integrate_data(
    assay1: AssayGrowth,
    assay2: AssayGrowth,
) -> AssaySetGrowth:
    """Combine normalized counts from two replicates via full outer join.

    Variants present in only one replicate will have NaN for the other replicate's
    timepoints.

    Parameters
    ----------
    assay1:
        First replicate (must have norm_counts populated).
    assay2:
        Second replicate (must have norm_counts populated).

    Returns
    -------
    AssaySetGrowth
        A combined assay set with norm counts from both replicates.
    """
    if assay1.norm_counts is None or assay2.norm_counts is None:
        raise ValueError("Both assays must have norm_counts populated before integrating.")

    names1 = assay1.norm_var_names or assay1.var_names
    names2 = assay2.norm_var_names or assay2.var_names

    T1 = assay1.norm_counts.shape[1]
    T2 = assay2.norm_counts.shape[1]

    df1 = pd.DataFrame(
        assay1.norm_counts,
        index=names1,
        columns=[f"r{assay1.rep}_t{t}" for t in range(T1)],
    )
    df2 = pd.DataFrame(
        assay2.norm_counts,
        index=names2,
        columns=[f"r{assay2.rep}_t{t}" for t in range(T2)],
    )

    combined_df = df1.join(df2, how="outer")
    combined_counts = combined_df.values
    all_var_names = list(combined_df.index)

    # Raw counts: outer join of raw counts too
    raw1 = pd.DataFrame(
        assay1.counts,
        index=assay1.var_names,
        columns=[f"r{assay1.rep}_raw_t{t}" for t in range(assay1.counts.shape[1])],
    )
    raw2 = pd.DataFrame(
        assay2.counts,
        index=assay2.var_names,
        columns=[f"r{assay2.rep}_raw_t{t}" for t in range(assay2.counts.shape[1])],
    )
    raw_df = raw1.join(raw2, how="outer")

    key = f"{assay1.key}_{assay2.key}" if assay1.key != assay2.key else assay1.key

    return AssaySetGrowth(
        combined_counts=combined_counts,
        var_names=all_var_names,
        reps=[assay1.rep, assay2.rep],
        key=key,
        raw_counts=raw_df.values,
        rounds=[assay1.rounds, assay2.rounds],
    )
