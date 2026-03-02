"""Tests for SLR scoring, focusing on correct time normalization and LFSR threshold."""
import numpy as np
import pytest
from rosace.assay import AssayGrowth, AssaySetGrowth
from rosace.preprocessing import normalize_data
from rosace.slr import run_slr
from rosace.utils import output_score


def make_normed_assay(counts, var_names=None, key="test", rep=1):
    if var_names is None:
        var_names = [f"v{i}" for i in range(counts.shape[0])]
    a = AssayGrowth(counts=counts.astype(float), var_names=var_names, key=key, rep=rep)
    return normalize_data(a, method="total")


class TestSLRTimeNormalization:
    """SLR slopes must match the R implementation, which normalizes time to [0, 1]."""

    def test_assay_growth_slope_scale(self):
        # Variant 0 grows perfectly: counts ~ [1, 2, 4, 8]
        # After total normalization the slope should be in [0, 1] space (time 0..1).
        counts = np.array([[1, 2, 4, 8], [10, 10, 10, 10]], dtype=float)
        a = make_normed_assay(counts)
        score = run_slr(a)
        # R: xt = seq(0, 3)/3 = [0, 1/3, 2/3, 1]
        # The slope for the growing variant should be positive.
        slope_v0 = score.score.loc[score.score["variant"] == "v0", "mean"].iloc[0]
        assert slope_v0 > 0.0
        # With normalised time [0, 1] the slope magnitude must be ≤ max(norm_count).
        # For unnormalized time [0, 3] slope would be 3× larger.
        # Check magnitude is in the normalized range (not 3× inflated).
        assert abs(slope_v0) < 3.0  # would be >> 3 if unnormalized

    def test_assay_growth_rounds_3(self):
        # Perfect linear growth over 4 timepoints (rounds=3).
        # After total normalization, t=0 column is 0; slope ≈ Δy per unit normalized time.
        counts = np.array([[10, 20, 30, 40], [10, 10, 10, 10]], dtype=float)
        a = make_normed_assay(counts)
        score_r = run_slr(a)
        # Supply explicit normalized time; results must match default
        explicit_t = [0.0, 1 / 3, 2 / 3, 1.0]
        score_e = run_slr(a, t=explicit_t)
        np.testing.assert_allclose(
            score_r.score["mean"].values,
            score_e.score["mean"].values,
            rtol=1e-10,
        )

    def test_assay_set_growth_normalized_by_max_rounds(self):
        # Two replicates with rounds=3 each.
        # Default time for each replicate should be [0, 1/3, 2/3, 1], not [0,1,2,3].
        counts = np.array([[10, 20, 30, 40]], dtype=float)
        a1 = make_normed_assay(counts, rep=1)
        a2 = make_normed_assay(counts, rep=2)
        from rosace.preprocessing import integrate_data
        combined = integrate_data(a1, a2)
        score = run_slr(combined)
        # With max_rounds=3, time is [0, 1/3, 2/3, 1, 0, 1/3, 2/3, 1]
        # Supply identical explicit time and compare
        explicit_t = [0.0, 1 / 3, 2 / 3, 1.0]
        score_e = run_slr(combined, t=explicit_t)
        np.testing.assert_allclose(
            score.score["mean"].values,
            score_e.score["mean"].values,
            rtol=1e-10,
        )


class TestOutputScoreLfsrThreshold:
    """output_score must use sig_test/2 as the per-tail threshold (matching R)."""

    def test_neutral_at_sig_test_half(self):
        """A variant with lfsr = sig_test/2 + ε should be Neutral."""
        from rosace.score import Score
        import pandas as pd
        # Construct a score where lfsr is just above 0.025
        # mean=1, sd chosen so that pnorm(0, 1, sd) ≈ 0.026 > sig_test/2
        import scipy.stats
        sig_test = 0.05
        target_lfsr = sig_test / 2 + 0.001  # 0.026 — just above threshold
        # Find sd: pnorm(0, 1, sd) = target_lfsr  →  sd = 1 / qnorm(1 - target_lfsr)
        sd = 1.0 / scipy.stats.norm.ppf(1 - target_lfsr)
        df = pd.DataFrame({"variant": ["v0"], "mean": [1.0], "sd": [sd], "lfsr": [0.0]})
        s = Score(method="ROSACE1", type="AssayGrowth", assay_name="test", score=df)
        result = output_score(s, sig_test=sig_test)
        assert result.loc[0, "label"] == "Neutral", (
            f"Expected Neutral for lfsr≈{target_lfsr:.4f} with sig_test/2={sig_test/2}"
        )

    def test_pos_just_below_threshold(self):
        """A variant with lfsr just below sig_test/2 should be Pos (positive mean)."""
        from rosace.score import Score
        import pandas as pd
        import scipy.stats
        sig_test = 0.05
        target_lfsr = sig_test / 2 - 0.001  # 0.024 — just below threshold
        sd = 1.0 / scipy.stats.norm.ppf(1 - target_lfsr)
        df = pd.DataFrame({"variant": ["v0"], "mean": [1.0], "sd": [sd], "lfsr": [0.0]})
        s = Score(method="ROSACE1", type="AssayGrowth", assay_name="test", score=df)
        result = output_score(s, sig_test=sig_test)
        assert result.loc[0, "label"] == "Pos"

    def test_neg_label(self):
        """Negative mean with lfsr below sig_test/2 should be Neg."""
        from rosace.score import Score
        import pandas as pd
        import scipy.stats
        sig_test = 0.05
        target_lfsr = sig_test / 2 - 0.001
        sd = 1.0 / scipy.stats.norm.ppf(1 - target_lfsr)
        df = pd.DataFrame({"variant": ["v0"], "mean": [-1.0], "sd": [sd], "lfsr": [0.0]})
        s = Score(method="ROSACE1", type="AssayGrowth", assay_name="test", score=df)
        result = output_score(s, sig_test=sig_test)
        assert result.loc[0, "label"] == "Neg"
