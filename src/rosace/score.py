"""Score class for storing ROSACE analysis results."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class Score:
    """Stores the result of a ROSACE scoring run.
    
    Attributes:
        method: Scoring method used. One of ROSACE0, ROSACE1, ROSACE2, ROSACE3, SLR.
        type: Assay type. One of AssayGrowth, AssaySetGrowth.
        assay_name: Name of the assay that was scored.
        score: DataFrame with columns: variant, mean, sd, test_statistic (or lfsr).
        optional_score: Optional additional score DataFrame (e.g., position-level).
        misc: Miscellaneous metadata.
    """
    method: str
    type: str
    assay_name: str
    score: pd.DataFrame
    optional_score: Optional[pd.DataFrame] = None
    misc: dict = field(default_factory=dict)

    def __post_init__(self):
        valid_methods = {"ROSACE0", "ROSACE1", "ROSACE2", "ROSACE3", "SLR"}
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {self.method!r}")
        valid_types = {"AssayGrowth", "AssaySetGrowth"}
        if self.type not in valid_types:
            raise ValueError(f"type must be one of {valid_types}, got {self.type!r}")
