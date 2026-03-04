"""Tests for output_score compatibility options."""

import pandas as pd
import pytest

from rosace.score import Score
from rosace.utils import output_score


def _base_score(method: str = "ROSACE1") -> Score:
    score_df = pd.DataFrame(
        {
            "variant": ["v1", "v2"],
            "mean": [0.5, -0.4],
            "sd": [0.1, 0.2],
            "lfsr": [0.0, 0.0],
        }
    )
    optional = pd.DataFrame(
        {
            "pos": [10, 20],
            "mean": [0.2, -0.2],
            "sd": [0.05, 0.06],
            "phi_mean": [0.2, -0.2],
            "phi_sd": [0.05, 0.06],
            "sigma2_mean": [0.1, 0.2],
            "sigma2_sd": [0.01, 0.02],
        }
    )
    return Score(
        method=method,
        type="AssaySetGrowth",
        assay_name="test",
        score=score_df,
        optional_score=optional,
    )


def test_output_score_default_dataframe():
    s = _base_score("ROSACE1")
    df = output_score(s, sig_test=0.05)
    assert isinstance(df, pd.DataFrame)
    assert {"lfsr", "label", "test_neg", "test_pos"}.issubset(df.columns)


def test_output_score_pos_info_returns_dict():
    s = _base_score("ROSACE1")
    out = output_score(s, pos_info=True)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"df_variant", "df_position"}
    assert {"pos", "mean", "sd", "label"}.issubset(out["df_position"].columns)


def test_output_score_blosum_requires_model2_or_3():
    s = _base_score("ROSACE1")
    with pytest.raises(ValueError, match="ROSACE2 or ROSACE3"):
        output_score(s, blosum_info=True)


def test_output_score_pos_act_requires_model3():
    s = _base_score("ROSACE2")
    with pytest.raises(ValueError, match="ROSACE3"):
        output_score(s, pos_act_info=True)


def test_output_score_blosum_and_rho_columns_present():
    s = _base_score("ROSACE3")
    s.score["blosum_group"] = [1, 2]
    s.score["blosum_score"] = [3, -1]
    s.score["rho_mean"] = [0.7, 0.2]
    s.score["rho_sd"] = [0.1, 0.1]
    out = output_score(s, pos_info=True, blosum_info=True, pos_act_info=True)
    assert "blosum_group" in out["df_variant"].columns
    assert "rho_mean" in out["df_variant"].columns
