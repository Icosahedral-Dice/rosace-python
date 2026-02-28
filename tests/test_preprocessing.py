"""Tests for rosace preprocessing functions."""
import numpy as np
import pytest
from rosace.assay import AssayGrowth
from rosace.preprocessing import filter_data, impute_data, normalize_data, integrate_data


def make_assay(counts, var_names=None, key="test", rep=1):
    if var_names is None:
        var_names = [f"v{i}" for i in range(counts.shape[0])]
    return AssayGrowth(counts=counts.astype(float), var_names=var_names, key=key, rep=rep)


def test_filter_data_min_count():
    counts = np.array([
        [5, 5],    # total 10 < 20 -> filtered
        [20, 20],  # total 40 >= 20 -> kept
        [30, 30],  # total 60 >= 20 -> kept
    ], dtype=float)
    a = make_assay(counts)
    filtered = filter_data(a, min_count=20)
    assert filtered.counts.shape[0] == 2
    assert filtered.var_names == ["v1", "v2"]


def test_filter_data_na_fraction():
    counts = np.array([
        [np.nan, np.nan, 10],  # 2/3 NA > 0.5 -> filtered
        [10, 20, 30],
    ], dtype=float)
    a = make_assay(counts)
    filtered = filter_data(a, na_rmax=0.5, min_count=0)
    assert filtered.counts.shape[0] == 1
    assert filtered.var_names == ["v1"]


def test_impute_zero():
    counts = np.array([[1, np.nan, 3], [4, 5, 6]], dtype=float)
    a = make_assay(counts)
    imp = impute_data(a, method="zero")
    assert not np.any(np.isnan(imp.counts))
    assert imp.counts[0, 1] == 0.0


def test_impute_knn():
    counts = np.array([
        [1, np.nan, 3],
        [2, 4, 6],
        [1, 2, 3],
    ], dtype=float)
    a = make_assay(counts)
    imp = impute_data(a, method="knn")
    assert not np.any(np.isnan(imp.counts))
    assert imp.counts[0, 1] > 0  # should be imputed to something positive


def test_impute_invalid():
    counts = np.ones((2, 3))
    a = make_assay(counts)
    with pytest.raises(ValueError, match="Unknown imputation method"):
        impute_data(a, method="invalid")


def test_normalize_wt():
    counts = np.array([
        [100, 200, 400],  # WT
        [50,  80,  120],  # variant
    ], dtype=float)
    a = make_assay(counts, var_names=["wt", "v1"])
    normed = normalize_data(a, method="wt", wt_var_names=["wt"], wt_rm=True)
    # WT removed; only v1 remains
    assert normed.norm_counts.shape[0] == 1
    assert normed.norm_var_names == ["v1"]
    # t=0 should be 0 after subtraction
    assert abs(normed.norm_counts[0, 0]) < 1e-10


def test_normalize_wt_no_rm():
    counts = np.array([
        [100, 200, 400],
        [50,  80,  120],
    ], dtype=float)
    a = make_assay(counts, var_names=["wt", "v1"])
    normed = normalize_data(a, method="wt", wt_var_names=["wt"], wt_rm=False)
    assert normed.norm_counts.shape[0] == 2
    # first column should be 0
    np.testing.assert_allclose(normed.norm_counts[:, 0], 0.0, atol=1e-10)


def test_normalize_total():
    counts = np.array([
        [100, 200, 400],
        [50,  80,  120],
    ], dtype=float)
    a = make_assay(counts, var_names=["v0", "v1"])
    normed = normalize_data(a, method="total")
    assert normed.norm_counts.shape == (2, 3)
    np.testing.assert_allclose(normed.norm_counts[:, 0], 0.0, atol=1e-10)


def test_normalize_invalid_method():
    counts = np.ones((2, 3))
    a = make_assay(counts)
    with pytest.raises(ValueError, match="Unknown normalization method"):
        normalize_data(a, method="bad")


def test_normalize_wt_missing():
    counts = np.ones((2, 3))
    a = make_assay(counts)
    with pytest.raises(ValueError, match="wt_var_names must be provided"):
        normalize_data(a, method="wt")


def test_integrate_data():
    counts1 = np.array([[100, 200], [50, 80]], dtype=float)
    a1 = make_assay(counts1, var_names=["v1", "v2"], key="exp", rep=1)
    a1 = normalize_data(a1, method="total")

    counts2 = np.array([[110, 210], [60, 90]], dtype=float)
    a2 = make_assay(counts2, var_names=["v1", "v3"], key="exp", rep=2)
    a2 = normalize_data(a2, method="total")

    combined = integrate_data(a1, a2)
    assert "v1" in combined.var_names
    assert "v2" in combined.var_names
    assert "v3" in combined.var_names
    assert combined.combined_counts.shape[0] == 3  # v1, v2, v3
    assert combined.combined_counts.shape[1] == 4  # 2 timepoints x 2 reps
