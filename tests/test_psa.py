from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from psa import (
    compute_posteriors,
    eps_nd_from_ratio,
    fit_sources,
    ratio_from_eps_nd,
    read_table,
    smart_to_float,
)
from psa.plotting import plot_posterior_heatmap, plot_region_diagram

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "raw, expected",
    [("0,7123", 0.7123), ("0.7123", 0.7123), ("1.234,5", 1234.5), ("1,234.5", 1234.5),
     ("−5.2", -5.2), ("NA", np.nan), ("", np.nan), (None, np.nan)],
)
def test_smart_to_float(raw, expected):
    out = smart_to_float(raw)
    if np.isnan(expected):
        assert np.isnan(out)
    else:
        assert out == pytest.approx(expected)


@pytest.mark.parametrize("sep", [",", ";", "\t"])
def test_read_table_detects_separator(sep):
    value = "0.71" if sep == "," else "0,71"
    text = sep.join(["Sample", "Sr"]) + "\n" + sep.join(["A", value]) + "\n"
    df = read_table(text.encode())
    assert list(df.columns) == ["Sample", "Sr"]
    assert smart_to_float(df.iloc[0, 1]) == pytest.approx(0.71)


def test_eps_roundtrip():
    assert eps_nd_from_ratio(ratio_from_eps_nd(-7.3)) == pytest.approx(-7.3)


def _two_sources():
    rng = np.random.default_rng(0)
    a = rng.multivariate_normal([0.706, 0.51260], np.diag([1e-6, 1e-10]), 40)
    b = rng.multivariate_normal([0.730, 0.51180], np.diag([1e-6, 1e-10]), 40)
    return pd.DataFrame({
        "region": ["A"] * 40 + ["B"] * 40,
        "Sr": np.r_[a[:, 0], b[:, 0]],
        "Nd": np.r_[a[:, 1], b[:, 1]],
    })


def test_posteriors_pick_correct_source():
    src = _two_sources()
    stats = fit_sources(src)
    samples = pd.DataFrame({"sample_id": ["nearA", "nearB"], "Sr": [0.7061, 0.7295],
                            "Nd": [0.51259, 0.51181]})
    long_df, wide, diag = compute_posteriors(samples, stats)
    assert wide.loc["A", "nearA"] > 0.99
    assert wide.loc["B", "nearB"] > 0.99
    np.testing.assert_allclose(wide.sum(axis=0), 1.0)
    assert list(diag["most_probable_source"]) == ["A", "B"]
    assert not diag["outside_all_95pct_ellipses"].any()


def test_only_selected_regions_are_used():
    stats = fit_sources(_two_sources(), regions=["B"])
    samples = pd.DataFrame({"sample_id": ["x"], "Sr": [0.706], "Nd": [0.5126]})
    _, wide, diag = compute_posteriors(samples, stats)
    assert list(wide.index) == ["B"]
    assert wide.iloc[0, 0] == pytest.approx(1.0)
    assert diag["outside_all_95pct_ellipses"].iloc[0]


def test_custom_priors_shift_posterior():
    stats = pd.DataFrame({"region": ["A", "B"], "n": [10, 10], "mu_Sr": [0.70, 0.72], "mu_Nd": [0.5126, 0.5126],
                          "cov_SrSr": [1e-4, 1e-4], "cov_SrNd": [0.0, 0.0], "cov_NdNd": [1e-8, 1e-8]})
    mid = pd.DataFrame({"sample_id": ["m"], "Sr": [0.71], "Nd": [0.5126]})
    _, w_uniform, _ = compute_posteriors(mid, stats)
    _, w_prior, _ = compute_posteriors(mid, stats, priors={"A": 0.9, "B": 0.1})
    assert w_uniform.loc["A", "m"] == pytest.approx(0.5)
    assert w_prior.loc["A", "m"] == pytest.approx(0.9)


def test_small_region_does_not_crash():
    src = pd.DataFrame({"region": ["X", "Y", "Y"], "Sr": [0.71, 0.72, 0.721], "Nd": [0.5121, 0.5119, 0.51191]})
    stats = fit_sources(src)
    assert set(stats["region"]) == {"X", "Y"}
    _, wide, _ = compute_posteriors(pd.DataFrame({"sample_id": ["s"], "Sr": [0.72], "Nd": [0.5119]}), stats)
    assert wide.shape == (2, 1)


def test_example_files_and_plots():
    src_raw = read_table(str(ROOT / "examples" / "example_sources_synthetic.csv"))
    smp_raw = read_table(str(ROOT / "examples" / "example_samples_synthetic.csv"))
    src = pd.DataFrame({
        "region": src_raw["Region"],
        "Sr": src_raw["87Sr/86Sr"].apply(smart_to_float),
        "Nd": src_raw["143Nd/144Nd"].apply(smart_to_float),
    })
    src["epsNd"] = eps_nd_from_ratio(src["Nd"])
    smp = pd.DataFrame({
        "sample_id": smp_raw["Sample"],
        "group": smp_raw["Group"],
        "Sr": smp_raw["87Sr/86Sr"].apply(smart_to_float),
        "Nd": smp_raw["143Nd/144Nd"].apply(smart_to_float),
    })
    smp["epsNd"] = eps_nd_from_ratio(smp["Nd"])
    regions = sorted(src["region"].unique())
    _, wide, _ = compute_posteriors(smp, fit_sources(src, regions))
    fa = plot_region_diagram(src, smp, regions)
    fb = plot_posterior_heatmap(wide, annotate=True)
    assert fa and fb
