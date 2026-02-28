"""Rosette class for DMS simulation and analysis."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class Rosette:
    """Container for Rosette simulation/analysis results.
    
    Attributes:
        score_df: Per-variant scores with columns: index, score, pos, wt, mut, ctrl, test, blosum.
        pos_df: Per-position summary with columns: pos, wt, mean.
        var_label: List of variant class labels used.
            One of: ["Neutral"], ["Neutral","Neg"], ["Neutral","Pos"], ["Neutral","Neg","Pos"]
        var_dist: Distribution parameters per label: label, mean, sd, count.
        disp: Sequencing dispersion parameter.
        disp_start: Library (initial) dispersion parameter.
        rounds: Number of growth rounds simulated.
        project_name: Human-readable project name.
    """
    score_df: pd.DataFrame
    pos_df: pd.DataFrame
    var_label: list[str]
    var_dist: pd.DataFrame
    disp: float
    disp_start: float
    rounds: int
    project_name: str = "RosetteProject"

    def __post_init__(self):
        valid_label_sets = [
            ["Neutral"],
            ["Neutral", "Neg"],
            ["Neutral", "Pos"],
            ["Neutral", "Neg", "Pos"],
        ]
        if self.var_label not in valid_label_sets:
            raise ValueError(
                f"var_label must be one of {valid_label_sets}, got {self.var_label}"
            )
