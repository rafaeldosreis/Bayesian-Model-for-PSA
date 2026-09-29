"""Bayesian Model for PSA (Potential Source Areas).

Core, UI-independent logic: robust file reading, data cleaning,
Bayesian source-area classification and plotting.
"""

from .io import read_table, smart_to_float, clean_text, guess_column
from .model import (
    CHUR_DEFAULT,
    eps_nd_from_ratio,
    ratio_from_eps_nd,
    robust_mean_cov,
    fit_sources,
    compute_posteriors,
)
from .plotting import plot_region_diagram, plot_posterior_heatmap

__version__ = "1.0.0"

__all__ = [
    "read_table",
    "smart_to_float",
    "clean_text",
    "guess_column",
    "CHUR_DEFAULT",
    "eps_nd_from_ratio",
    "ratio_from_eps_nd",
    "robust_mean_cov",
    "fit_sources",
    "compute_posteriors",
    "plot_region_diagram",
    "plot_posterior_heatmap",
]
