"""Visualization functions for rosace scores.

Mirrors the R rosace visualization functions: scoreHeatmap, scoreVlnplot,
scoreDensity.

All functions expect a merged DataFrame that contains both score columns
(mean, sd, lfsr, label) and variant metadata columns (position, wildtype,
mutation, type).  Such a DataFrame is obtained by joining the output of
``output_score()`` with the variant metadata used during the analysis.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

# Amino-acid order used for heatmap columns (standard one-letter codes,
# alphabetical by residue type, matching common DMS convention)
_AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

# Colour palette that matches R ggplot2 defaults used in the original rosace
_TYPE_PALETTE = {
    "synonymous": "#66C2A5",
    "missense":   "#FC8D62",
    "deletion":   "#8DA0CB",
    "nonsense":   "#E78AC3",
    "other":      "#A6D854",
}


def _get_mpl():
    """Lazy import matplotlib to avoid hard dependency at module level."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        ) from exc


def _get_sns():
    """Lazy import seaborn to avoid hard dependency at module level."""
    try:
        import seaborn as sns
        return sns
    except ImportError as exc:
        raise ImportError(
            "seaborn is required for visualization. "
            "Install it with: pip install seaborn"
        ) from exc


def _save_figure(fig, savedir: Optional[str], name: str) -> None:
    """Save a figure to *savedir/name.pdf* if ``savedir`` is given."""
    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
        path = os.path.join(savedir, f"{name}.pdf")
        fig.savefig(path, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def score_heatmap(
    data: pd.DataFrame,
    pos_col: str = "position",
    wt_col: str = "wildtype",
    mut_col: str = "mutation",
    type_col: str = "type",
    score_col: str = "mean",
    ctrl_name: str = "synonymous",
    title: Optional[str] = "Score Heatmap",
    savedir: Optional[str] = None,
    name: str = "Heatmap",
    show: bool = True,
    figsize: Optional[tuple] = None,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: float = 0.0,
) -> "plt.Figure":  # type: ignore[name-defined]
    """Plot a score heatmap (position × amino acid).

    Mirrors the R ``scoreHeatmap`` function.  Each cell is coloured by the
    functional score; wildtype residues are marked with a white diamond (◆);
    missing variants are shown in light grey.

    Parameters
    ----------
    data:
        DataFrame with at least ``pos_col``, ``wt_col``, ``mut_col``, and
        ``score_col`` columns (typically the output of ``output_score()``
        joined with variant metadata).
    pos_col:
        Column name for the numeric position.
    wt_col:
        Column name for the wildtype amino acid.
    mut_col:
        Column name for the mutant amino acid.
    type_col:
        Column name for mutation type (used to identify synonymous/ctrl rows).
    score_col:
        Column name for the numeric score to display.
    ctrl_name:
        Value in *type_col* that identifies synonymous/control variants.
        These are excluded from the heatmap (wildtype cells are already shown
        via the WT marker).
    title:
        Figure title.
    savedir:
        If given, saves the figure as ``savedir/name.pdf``.
    name:
        Base filename (without extension) when saving.
    show:
        If True, calls ``plt.show()``.
    figsize:
        Override the automatic figure size.
    cmap:
        Matplotlib diverging colourmap.
    vmin, vmax:
        Colour-scale limits.  Defaults to the 5th/95th percentile of scores.
    center:
        Centre of the diverging colour scale (default 0).

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_mpl()
    sns = _get_sns()

    # --- Filter out synonymous/control rows (they don't have a missense score)
    df = data.copy()
    if type_col in df.columns:
        df = df[df[type_col] != ctrl_name]

    # Drop rows with missing key columns
    df = df.dropna(subset=[pos_col, wt_col, mut_col, score_col])
    df[pos_col] = df[pos_col].astype(int)

    positions = sorted(df[pos_col].unique())
    n_pos = len(positions)

    # Build a pivot: rows = positions, columns = amino acids
    pivot = pd.pivot_table(
        df,
        values=score_col,
        index=pos_col,
        columns=mut_col,
        aggfunc="mean",
    )
    # Reindex columns to standard AA order (keep only observed AAs)
    aa_cols = [aa for aa in _AA_ORDER if aa in pivot.columns]
    # Add any non-standard AAs
    extra = [aa for aa in pivot.columns if aa not in _AA_ORDER]
    pivot = pivot.reindex(columns=aa_cols + extra)
    pivot = pivot.reindex(index=positions)

    # Auto-choose colour limits
    scores_flat = df[score_col].dropna().values
    if vmin is None:
        vmin = float(np.percentile(scores_flat, 5))
    if vmax is None:
        vmax = float(np.percentile(scores_flat, 95))

    # Create a mask for NaN cells
    mask = pivot.isna()

    # --- Figure sizing
    n_aa = len(pivot.columns)
    if figsize is None:
        width = max(8, n_aa * 0.45)
        height = max(4, n_pos * 0.25)
        figsize = (width, height)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        mask=mask,
        linewidths=0.4,
        linecolor="white",
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": score_col, "shrink": 0.6},
    )

    # Mark wildtype cells with a diamond
    if wt_col in df.columns:
        wt_map = df.groupby(pos_col)[wt_col].first()
        col_names = list(pivot.columns)
        for pos_val, wt_aa in wt_map.items():
            if wt_aa in col_names:
                col_idx = col_names.index(wt_aa)
                row_idx = positions.index(pos_val)
                ax.text(
                    col_idx + 0.5,
                    row_idx + 0.5,
                    "◆",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white",
                    fontweight="bold",
                )

    ax.set_xlabel("Mutant amino acid")
    ax.set_ylabel("Position")
    if title:
        ax.set_title(title)

    # Rotate x labels for legibility
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)

    plt.tight_layout()
    _save_figure(fig, savedir, name)
    if show:
        plt.show()
    return fig


def score_violin(
    data: pd.DataFrame,
    type_col: str = "type",
    score_col: str = "mean",
    order: Optional[list[str]] = None,
    title: Optional[str] = "Score Distribution by Mutation Type",
    savedir: Optional[str] = None,
    name: str = "ViolinPlot",
    show: bool = True,
    figsize: tuple = (8, 5),
) -> "plt.Figure":  # type: ignore[name-defined]
    """Violin plot of scores grouped by mutation type.

    Mirrors the R ``scoreVlnplot`` function.

    Parameters
    ----------
    data:
        DataFrame with at least *type_col* and *score_col* columns.
    type_col:
        Column that identifies mutation type (e.g. synonymous, missense).
    score_col:
        Column with numeric scores.
    order:
        Display order of mutation types.  Defaults to alphabetical.
    title:
        Figure title.
    savedir:
        If given, saves the figure as ``savedir/name.pdf``.
    name:
        Base filename.
    show:
        Call ``plt.show()`` after plotting.
    figsize:
        Figure dimensions in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_mpl()
    sns = _get_sns()

    df = data.dropna(subset=[type_col, score_col]).copy()
    if order is None:
        order = sorted(df[type_col].unique())

    # Build a colour palette; fall back to seaborn default for unknown types
    palette = {t: _TYPE_PALETTE.get(t, "#BBBBBB") for t in order}

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(
        data=df,
        x=type_col,
        y=score_col,
        hue=type_col,
        order=order,
        palette=palette,
        inner="quartile",
        cut=0,
        legend=False,
        ax=ax,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("Mutation type")
    ax.set_ylabel("Functional score")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    _save_figure(fig, savedir, name)
    if show:
        plt.show()
    return fig


def score_density(
    data: pd.DataFrame,
    type_col: str = "type",
    score_col: str = "mean",
    hist: bool = False,
    nbins: int = 30,
    scale_free: bool = False,
    order: Optional[list[str]] = None,
    title: Optional[str] = "Score Density by Mutation Type",
    savedir: Optional[str] = None,
    name: str = "DensityPlot",
    show: bool = True,
    figsize: Optional[tuple] = None,
) -> "plt.Figure":  # type: ignore[name-defined]
    """Density (or histogram) plot of scores by mutation type.

    Mirrors the R ``scoreDensity`` function.

    Parameters
    ----------
    data:
        DataFrame with at least *type_col* and *score_col* columns.
    type_col:
        Column that identifies mutation type.
    score_col:
        Column with numeric scores.
    hist:
        If True, plots a histogram; otherwise a KDE density curve.
    nbins:
        Number of histogram bins (only used when ``hist=True``).
    scale_free:
        If True, each facet has its own y-axis scale.
    order:
        Display order / subset of mutation types.
    title:
        Figure title.
    savedir:
        If given, saves the figure as ``savedir/name.pdf``.
    name:
        Base filename.
    show:
        Call ``plt.show()`` after plotting.
    figsize:
        Figure dimensions.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_mpl()
    sns = _get_sns()

    df = data.dropna(subset=[type_col, score_col]).copy()
    if order is None:
        order = sorted(df[type_col].unique())

    n_types = len(order)
    if figsize is None:
        figsize = (5 * n_types, 4)

    palette = {t: _TYPE_PALETTE.get(t, "#BBBBBB") for t in order}

    fig, axes = plt.subplots(
        1, n_types, figsize=figsize, sharey=(not scale_free)
    )
    if n_types == 1:
        axes = [axes]

    for ax, mut_type in zip(axes, order):
        subset = df[df[type_col] == mut_type][score_col].dropna()
        color = palette[mut_type]
        if hist:
            ax.hist(subset, bins=nbins, color=color, edgecolor="white",
                    alpha=0.85, density=True)
        else:
            sns.kdeplot(subset, ax=ax, color=color, fill=True, alpha=0.6,
                        linewidth=1.5)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_xlabel("Functional score")
        ax.set_ylabel("Density")
        ax.set_title(mut_type)

    fig.suptitle(title, y=1.02) if title else None
    plt.tight_layout()
    _save_figure(fig, savedir, name)
    if show:
        plt.show()
    return fig
