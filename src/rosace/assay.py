"""Data structures for rosace assays (mirrors R Assay, AssayGrowth, AssaySet, AssaySetGrowth)."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Assay:
    """Base assay class holding raw counts.
    
    Attributes:
        counts: Raw counts array of shape (variants, timepoints).
        var_names: List of variant names.
        key: Experiment key/name.
        rep: Replicate number.
    """
    counts: np.ndarray
    var_names: list[str]
    key: str
    rep: int

    def __post_init__(self):
        if len(self.var_names) != self.counts.shape[0]:
            raise ValueError(
                f"var_names length ({len(self.var_names)}) must match "
                f"counts rows ({self.counts.shape[0]})"
            )


@dataclass
class AssayGrowth(Assay):
    """Assay for growth-based screens with normalization support.
    
    Attributes:
        norm_counts: Normalized counts array of shape (variants, timepoints).
        norm_var_names: Variant names after normalization (may differ if wt removed).
        rounds: Number of growth rounds (= timepoints - 1).
    """
    norm_counts: Optional[np.ndarray] = None
    norm_var_names: Optional[list[str]] = None
    rounds: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.rounds == 0 and self.counts.ndim == 2:
            self.rounds = self.counts.shape[1] - 1


@dataclass
class AssaySet:
    """Combined assay set holding data from multiple replicates.
    
    Attributes:
        combined_counts: Combined counts of shape (variants, columns).
        var_names: Variant names.
        reps: List of replicate numbers included.
        key: Experiment key/name.
    """
    combined_counts: np.ndarray
    var_names: list[str]
    reps: list[int]
    key: str

    def __post_init__(self):
        if len(self.var_names) != self.combined_counts.shape[0]:
            raise ValueError(
                f"var_names length ({len(self.var_names)}) must match "
                f"combined_counts rows ({self.combined_counts.shape[0]})"
            )


@dataclass
class AssaySetGrowth(AssaySet):
    """AssaySet for growth-based screens.
    
    Attributes:
        raw_counts: Raw counts before normalization.
        rounds: List of growth rounds per replicate.
    """
    raw_counts: Optional[np.ndarray] = None
    rounds: Optional[list[int]] = None
