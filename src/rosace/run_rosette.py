"""Rosette simulation functions.

Implements the simulation pipeline from the R rosace package for generating
synthetic DMS data with known ground truth effects.
"""

from __future__ import annotations
import os
from typing import Optional, Union
import numpy as np
import pandas as pd
import scipy.stats

from rosace.rosette import Rosette


def generate_effect(config: dict) -> pd.DataFrame:
    """Generate variant effects from distributions defined in config.

    Parameters
    ----------
    config:
        Dictionary with keys:
        - ``n_pos``: int, number of positions
        - ``n_mut``: int, number of mutations per position
        - ``var_label``: list[str], e.g. ["Neutral", "Neg", "Pos"]
        - ``neutral_mean``: float (default 0)
        - ``neutral_sd``: float (default 0.1)
        - ``neg_mean``: float (default -1.0)
        - ``neg_sd``: float (default 0.3)
        - ``pos_mean``: float (default 0.5)
        - ``pos_sd``: float (default 0.2)
        - ``neg_frac``: float (default 0.3), fraction of non-neutral that are negative
        - ``pos_frac``: float (default 0.1), fraction of non-neutral that are positive
        - ``seed``: int (optional)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: variant, pos, mut, effect, label.
    """
    rng = np.random.default_rng(config.get("seed"))
    n_pos = config["n_pos"]
    n_mut = config.get("n_mut", 19)
    var_label = config.get("var_label", ["Neutral", "Neg", "Pos"])

    neutral_mean = config.get("neutral_mean", 0.0)
    neutral_sd = config.get("neutral_sd", 0.1)
    neg_mean = config.get("neg_mean", -1.0)
    neg_sd = config.get("neg_sd", 0.3)
    pos_mean = config.get("pos_mean", 0.5)
    pos_sd = config.get("pos_sd", 0.2)

    neg_frac = config.get("neg_frac", 0.3) if "Neg" in var_label else 0.0
    pos_frac = config.get("pos_frac", 0.1) if "Pos" in var_label else 0.0
    neutral_frac = 1.0 - neg_frac - pos_frac

    records = []
    for p in range(1, n_pos + 1):
        for m in range(n_mut):
            variant = f"p{p}m{m}"
            roll = rng.random()
            if roll < neutral_frac:
                effect = rng.normal(neutral_mean, neutral_sd)
                label = "Neutral"
            elif roll < neutral_frac + neg_frac:
                effect = rng.normal(neg_mean, neg_sd)
                label = "Neg"
            else:
                effect = rng.normal(pos_mean, pos_sd)
                label = "Pos"
            records.append({"variant": variant, "pos": p, "mut": m, "effect": effect, "label": label})

    return pd.DataFrame(records)


def generate_count(
    config: dict,
    effects: pd.DataFrame,
) -> pd.DataFrame:
    """Simulate sequencing counts for a DMS experiment.

    Simulates cell growth with Poisson noise per round, then negative binomial
    sequencing sampling.

    Parameters
    ----------
    config:
        Dictionary with keys:
        - ``rounds``: int, number of growth rounds (default 4)
        - ``n_reps``: int, number of biological replicates (default 2)
        - ``init_count``: int, initial library count per variant (default 100)
        - ``depth``: int, sequencing depth per timepoint (default 1_000_000)
        - ``disp``: float, sequencing negative binomial dispersion (default 0.05)
        - ``disp_start``: float, initial library dispersion (default 0.1)
    effects:
        DataFrame from generate_effect with columns: variant, effect.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with columns: variant, rep_{r}_t{t} for each
        replicate r and timepoint t.
    """
    rng = np.random.default_rng(config.get("seed"))
    rounds = config.get("rounds", 4)
    n_reps = config.get("n_reps", 2)
    init_count = config.get("init_count", 100)
    depth = config.get("depth", 1_000_000)
    disp = config.get("disp", 0.05)
    disp_start = config.get("disp_start", 0.1)

    n_vars = len(effects)
    effect_arr = effects["effect"].values

    result = {"variant": effects["variant"].tolist()}

    for rep in range(1, n_reps + 1):
        # Initial counts: negative binomial around init_count
        if disp_start > 0:
            r_start = 1.0 / disp_start
            p_start = r_start / (r_start + init_count)
            init = rng.negative_binomial(r_start, p_start, size=n_vars).astype(float)
        else:
            init = rng.poisson(init_count, size=n_vars).astype(float)

        # Growth simulation
        cell_counts = [init.copy()]
        for _ in range(rounds):
            prev = cell_counts[-1]
            growth = np.exp(effect_arr)
            new_cells = prev * growth
            new_cells = rng.poisson(np.maximum(new_cells, 0.1))
            cell_counts.append(new_cells.astype(float))

        # Sequencing: negative binomial sampling at each timepoint
        for t, cells in enumerate(cell_counts):
            total = cells.sum()
            if total <= 0:
                total = 1.0
            fracs = cells / total
            expected = fracs * depth
            if disp > 0:
                r_seq = 1.0 / disp
                p_seq = r_seq / (r_seq + expected)
                obs = rng.negative_binomial(r_seq, np.clip(p_seq, 1e-10, 1 - 1e-10))
            else:
                obs = rng.poisson(expected)
            result[f"rep_{rep}_t{t}"] = obs.astype(int).tolist()

    return pd.DataFrame(result)


def run_rosette(config: dict) -> None:
    """Run a full Rosette simulation and save results as TSV files.

    Parameters
    ----------
    config:
        Dictionary with all simulation parameters. Required keys:
        - ``output_dir``: str, directory to save output files.
        - ``n_pos``: int, number of positions.
        All other keys are optional with sensible defaults.

    Output files:
        - ``effects.tsv``: ground-truth variant effects.
        - ``counts.tsv``: simulated counts.
        - ``rosette.tsv``: summary with effects and counts combined.
    """
    output_dir = config.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)

    effects = generate_effect(config)
    counts = generate_count(config, effects)

    effects_path = os.path.join(output_dir, "effects.tsv")
    counts_path = os.path.join(output_dir, "counts.tsv")

    effects.to_csv(effects_path, sep="\t", index=False)
    counts.to_csv(counts_path, sep="\t", index=False)

    # Combined summary
    combined = effects.merge(counts, on="variant")
    combined_path = os.path.join(output_dir, "rosette.tsv")
    combined.to_csv(combined_path, sep="\t", index=False)
