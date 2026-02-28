"""Tests for rosace assay data structures."""
import numpy as np
import pytest
from rosace.assay import Assay, AssayGrowth, AssaySet, AssaySetGrowth


def test_assay_basic():
    counts = np.array([[10, 20, 30], [5, 15, 25]], dtype=float)
    a = Assay(counts=counts, var_names=["v1", "v2"], key="test", rep=1)
    assert a.counts.shape == (2, 3)
    assert a.key == "test"
    assert a.rep == 1


def test_assay_var_names_mismatch():
    counts = np.array([[10, 20], [5, 15]], dtype=float)
    with pytest.raises(ValueError, match="var_names length"):
        Assay(counts=counts, var_names=["v1"], key="test", rep=1)


def test_assay_growth_basic():
    counts = np.array([[10, 20, 30], [5, 15, 25]], dtype=float)
    ag = AssayGrowth(counts=counts, var_names=["v1", "v2"], key="test", rep=1)
    assert ag.rounds == 2  # 3 timepoints - 1


def test_assay_growth_with_norm():
    counts = np.array([[10, 20], [5, 15]], dtype=float)
    norm = np.array([[0.0, 0.5], [0.0, 0.3]], dtype=float)
    ag = AssayGrowth(
        counts=counts,
        var_names=["v1", "v2"],
        key="test",
        rep=1,
        norm_counts=norm,
        norm_var_names=["v1", "v2"],
    )
    assert ag.norm_counts.shape == (2, 2)


def test_assay_set_basic():
    combined = np.zeros((3, 4))
    s = AssaySet(combined_counts=combined, var_names=["a", "b", "c"], reps=[1, 2], key="k")
    assert s.combined_counts.shape == (3, 4)


def test_assay_set_growth_basic():
    combined = np.zeros((3, 4))
    sg = AssaySetGrowth(
        combined_counts=combined,
        var_names=["a", "b", "c"],
        reps=[1, 2],
        key="k",
        rounds=[3, 3],
    )
    assert sg.rounds == [3, 3]
