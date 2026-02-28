"""
rosace: Statistical inference for growth-based deep mutational scanning screens.
Python port of the R packages rosace and rosace-aa.
"""

__version__ = "0.1.0"

from rosace.assay import Assay, AssayGrowth, AssaySet, AssaySetGrowth
from rosace.rosace import Rosace
from rosace.score import Score
from rosace.rosette import Rosette
from rosace.preprocessing import filter_data, impute_data, normalize_data, integrate_data
from rosace.run_rosace import gen_rosace_input, run_rosace
from rosace.run_rosette import run_rosette
from rosace.slr import run_slr
from rosace.utils import map_blosum_score, estimate_disp, estimate_disp_start, output_score
from rosace.visualization import score_heatmap, score_violin, score_density

__all__ = [
    "__version__",
    "Assay", "AssayGrowth", "AssaySet", "AssaySetGrowth",
    "Rosace",
    "Score",
    "Rosette",
    "filter_data", "impute_data", "normalize_data", "integrate_data",
    "gen_rosace_input", "run_rosace",
    "run_rosette",
    "run_slr",
    "map_blosum_score", "estimate_disp", "estimate_disp_start", "output_score",
    "score_heatmap", "score_violin", "score_density",
]
