"""Tests for rosace core classes and Stan input preparation."""
import numpy as np
import pandas as pd
import pytest
from rosace.assay import AssayGrowth
from rosace.rosace import Rosace
from rosace.score import Score
from rosace.stan_models import STAN_MODELS, GROWTH_NOPOS, GROWTH_POS
from rosace.run_rosace import gen_rosace_input
from rosace.utils import map_blosum_score
from rosace.preprocessing import normalize_data


def make_normed_assay(n_vars=10, n_tp=4, key="test", rep=1):
    rng = np.random.default_rng(42)
    counts = rng.integers(10, 200, size=(n_vars, n_tp)).astype(float)
    var_names = [f"v{i}" for i in range(n_vars)]
    a = AssayGrowth(counts=counts, var_names=var_names, key=key, rep=rep)
    return normalize_data(a, method="total")


class TestRosace:
    def test_init(self):
        r = Rosace()
        assert r.project_name == "RosaceProject"
        assert r.assays == {}

    def test_add_assay(self):
        r = Rosace()
        a = make_normed_assay()
        r.add_assay("test", a)
        assert "test" in r.assays
        assert r.list_assays() == ["test"]

    def test_add_score(self):
        r = Rosace()
        score_df = pd.DataFrame({"variant": ["v0"], "mean": [0.1], "sd": [0.05], "lfsr": [0.01]})
        s = Score(method="ROSACE1", type="AssayGrowth", assay_name="test", score=score_df)
        r.add_score("s1", s)
        assert "s1" in r.scores


class TestScore:
    def test_valid_score(self):
        df = pd.DataFrame({"variant": ["v0"], "mean": [0.0], "sd": [0.1]})
        s = Score(method="ROSACE1", type="AssayGrowth", assay_name="a", score=df)
        assert s.method == "ROSACE1"

    def test_invalid_method(self):
        df = pd.DataFrame({"variant": ["v0"], "mean": [0.0], "sd": [0.1]})
        with pytest.raises(ValueError, match="method must be one of"):
            Score(method="BAD", type="AssayGrowth", assay_name="a", score=df)

    def test_invalid_type(self):
        df = pd.DataFrame({"variant": ["v0"], "mean": [0.0], "sd": [0.1]})
        with pytest.raises(ValueError, match="type must be one of"):
            Score(method="ROSACE1", type="BAD", assay_name="a", score=df)


class TestStanModels:
    def test_stan_models_are_strings(self):
        for name, model in STAN_MODELS.items():
            assert isinstance(model, str), f"{name} should be a string"
            assert "data {" in model
            assert "parameters {" in model
            assert "model {" in model

    def test_growth_nopos_has_required_fields(self):
        assert "int<lower=0> T" in GROWTH_NOPOS
        assert "int<lower=0> V" in GROWTH_NOPOS
        assert "vector[V] beta" in GROWTH_NOPOS
        assert "vMAPm" in GROWTH_NOPOS

    def test_growth_pos_has_position_fields(self):
        assert "int<lower=0> P" in GROWTH_POS
        assert "vMAPp" in GROWTH_POS
        assert "vector[P] phi" in GROWTH_POS


class TestGenRosaceInput:
    def test_rosace0_input(self):
        a = make_normed_assay(n_vars=5, n_tp=4)
        data = gen_rosace_input(a, method="ROSACE0")
        assert data["T"] == 4
        assert data["V"] == 5
        assert len(data["vMAPm"]) == 5
        assert "vMAPp" not in data

    def test_rosace1_input(self):
        a = make_normed_assay(n_vars=6, n_tp=3)
        var_info = pd.DataFrame({
            "variant": [f"v{i}" for i in range(6)],
            "pos": [1, 1, 2, 2, 3, 3],
            "wt": ["A"] * 6,
            "mut": ["R", "N", "D", "C", "Q", "E"],
        })
        data = gen_rosace_input(a, method="ROSACE1", var_info=var_info)
        assert data["P"] == 3
        assert len(data["vMAPp"]) == 6
        assert "vMAPb" not in data

    def test_rosace2_input(self):
        a = make_normed_assay(n_vars=6, n_tp=3)
        var_info = pd.DataFrame({
            "variant": [f"v{i}" for i in range(6)],
            "pos": [1, 1, 2, 2, 3, 3],
            "wt": ["A"] * 6,
            "mut": ["R", "N", "D", "C", "Q", "E"],
        })
        data = gen_rosace_input(a, method="ROSACE2", var_info=var_info)
        assert "vMAPb" in data
        assert data["B"] == 11
        assert all(1 <= g <= 11 for g in data["vMAPb"])

    def test_time_vector(self):
        a = make_normed_assay(n_vars=5, n_tp=4)
        data = gen_rosace_input(a, method="ROSACE0", t=[0, 2, 4, 6])
        assert data["t"] == [0, 2, 4, 6]

    def test_default_time_vector_normalized(self):
        # Default time vector must match R: seq(0, rounds)/rounds → [0, 1/3, 2/3, 1]
        a = make_normed_assay(n_vars=5, n_tp=4)  # rounds = 3
        data = gen_rosace_input(a, method="ROSACE0")
        expected = [0.0, 1 / 3, 2 / 3, 1.0]
        assert len(data["t"]) == 4
        for got, exp in zip(data["t"], expected):
            assert abs(got - exp) < 1e-12, f"t mismatch: {data['t']} != {expected}"

    def test_vmapm_from_raw_counts(self):
        # vMAPm should be rank-based (ceiling(rank/25)), not quantile-based
        a = make_normed_assay(n_vars=30, n_tp=4)
        data = gen_rosace_input(a, method="ROSACE0")
        # With 30 variants, ceiling(rank/25) gives groups 1 and 2
        assert all(g >= 1 for g in data["vMAPm"])
        assert data["M"] == max(data["vMAPm"])


class TestBlosumMapping:
    def test_synonymous(self):
        assert map_blosum_score("A", "A") == 11

    def test_known_negative(self):
        # W->A has BLOSUM score -3 -> group 2
        result = map_blosum_score("W", "A")
        assert result == 2

    def test_known_positive(self):
        # I->V has BLOSUM score 3 -> group 8
        result = map_blosum_score("I", "V")
        assert result == 8

    def test_all_groups_1_to_10(self):
        # Just check function returns valid groups for various pairs
        for wt in "ARND":
            for mut in "CQEGH":
                g = map_blosum_score(wt, mut)
                assert 1 <= g <= 11, f"Group {g} out of range for ({wt},{mut})"
