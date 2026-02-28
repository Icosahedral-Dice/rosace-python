"""Core Rosace container class."""

from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd

from rosace.assay import Assay, AssaySet
from rosace.score import Score


@dataclass
class Rosace:
    """Top-level container for a rosace deep mutational scanning experiment.
    
    Attributes:
        assays: Mapping of assay name to Assay (or AssayGrowth).
        assay_sets: Mapping of assay set name to AssaySet (or AssaySetGrowth).
        var_data: DataFrame with variant metadata (one row per variant).
        scores: Mapping of score name to Score.
        project_name: Human-readable project name.
        misc: Miscellaneous metadata.
    """
    assays: dict = field(default_factory=dict)
    assay_sets: dict = field(default_factory=dict)
    var_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    scores: dict = field(default_factory=dict)
    project_name: str = "RosaceProject"
    misc: dict = field(default_factory=dict)

    def add_assay(self, name: str, assay: Assay) -> None:
        """Add an assay to the container."""
        self.assays[name] = assay

    def add_assay_set(self, name: str, assay_set: AssaySet) -> None:
        """Add an assay set to the container."""
        self.assay_sets[name] = assay_set

    def add_score(self, name: str, score: Score) -> None:
        """Add a scoring result to the container."""
        self.scores[name] = score

    def list_assays(self) -> list[str]:
        """Return list of assay names."""
        return list(self.assays.keys())

    def list_assay_sets(self) -> list[str]:
        """Return list of assay set names."""
        return list(self.assay_sets.keys())

    def list_scores(self) -> list[str]:
        """Return list of score names."""
        return list(self.scores.keys())
