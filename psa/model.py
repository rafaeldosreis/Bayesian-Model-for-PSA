"""Bayesian classification of samples into Potential Source Areas (PSAs).

For every source region *k* a bivariate normal distribution is fitted in
(87Sr/86Sr, 143Nd/144Nd) space using a robust estimator (Minimum
Covariance Determinant). For a sample x with analytical uncertainties
(sigma_Sr, sigma_Nd) the likelihood is

    p(x | k) = N(x; mu_k, Sigma_k + diag(sigma_Sr^2, sigma_Nd^2))

and the posterior probability follows Bayes' rule

    P(k | x) = pi_k p(x | k) / sum_j pi_j p(x | j)

where pi_k are the prior probabilities (uniform by default).
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2, multivariate_normal
from sklearn.covariance import MinCovDet

CHUR_DEFAULT = 0.512638  # present-day CHUR 143Nd/144Nd (Jacobsen & Wasserburg, 1980)


def eps_nd_from_ratio(nd_ratio, chur: float = CHUR_DEFAULT):
    """epsilon-Nd(0) from a measured 143Nd/144Nd ratio."""
    return (np.asarray(nd_ratio, dtype=float) / chur - 1.0) * 1e4


def ratio_from_eps_nd(eps_nd, chur: float = CHUR_DEFAULT):
    """143Nd/144Nd ratio from epsilon-Nd(0)."""
    return (np.asarray(eps_nd, dtype=float) / 1e4 + 1.0) * chur


def robust_mean_cov(
    X,
    cov_regularization: float = 1e-8,
    min_points_for_cov: int = 4,
    robust: bool = True,
):
    """Location and covariance of a 2-D point cloud.

    * n == 1  -> the point itself with a tiny diagonal covariance
    * n <  min_points_for_cov -> mean and diagonal sample variance
    * otherwise -> MCD robust estimate (falls back to the classic one)
    """
    X = np.asarray(X, dtype=float)
    X = X[~np.isnan(X).any(axis=1)]
    n, d = X.shape if X.ndim == 2 else (0, 2)

    if n == 0:
        raise ValueError("No valid points.")
    if n == 1:
        return X[0], np.eye(d) * 1e-6
    if n < min_points_for_cov:
        var = np.var(X, axis=0, ddof=1)
        var = np.where(np.isfinite(var) & (var > 0), var, 1e-6)
        return X.mean(axis=0), np.diag(var) + np.eye(d) * cov_regularization

    if robust:
        try:
            mcd = MinCovDet(random_state=0).fit(X)
            return mcd.location_, mcd.covariance_ + np.eye(d) * cov_regularization
        except Exception:
            pass
    cov = np.atleast_2d(np.cov(X.T))
    return X.mean(axis=0), cov + np.eye(d) * cov_regularization


def fit_sources(
    sources: pd.DataFrame,
    regions: Optional[Sequence[str]] = None,
    robust: bool = True,
    cov_regularization: float = 1e-8,
) -> pd.DataFrame:
    """Fit one bivariate normal per region.

    ``sources`` must contain the columns ``region``, ``Sr`` and ``Nd``
    (143Nd/144Nd). Returns one row per region with n, mean and covariance.
    """
    df = sources.dropna(subset=["region", "Sr", "Nd"])
    if regions is None:
        regions = sorted(df["region"].unique())

    rows = []
    for reg in regions:
        X = df.loc[df["region"] == reg, ["Sr", "Nd"]].to_numpy(dtype=float)
        if len(X) == 0:
            continue
        mu, cov = robust_mean_cov(X, cov_regularization=cov_regularization, robust=robust)
        rows.append(
            {
                "region": reg,
                "n": len(X),
                "mu_Sr": mu[0],
                "mu_Nd": mu[1],
                "cov_SrSr": cov[0, 0],
                "cov_SrNd": cov[0, 1],
                "cov_NdNd": cov[1, 1],
            }
        )
    return pd.DataFrame(
        rows, columns=["region", "n", "mu_Sr", "mu_Nd", "cov_SrSr", "cov_SrNd", "cov_NdNd"]
    )


def _prior_vector(regions: Sequence[str], priors: Optional[Dict[str, float]]) -> np.ndarray:
    if not priors:
        p = np.ones(len(regions), dtype=float)
    else:
        p = np.array([float(priors.get(r, 0.0)) for r in regions], dtype=float)
        p = np.where(np.isfinite(p) & (p > 0), p, 0.0)
        if p.sum() <= 0:
            raise ValueError("Priors must contain at least one positive value.")
    return p / p.sum()


def compute_posteriors(
    samples: pd.DataFrame,
    source_stats: pd.DataFrame,
    priors: Optional[Dict[str, float]] = None,
    use_sample_errors: bool = True,
    cov_regularization: float = 1e-8,
):
    """Posterior probability of each source region for each sample.

    ``samples`` needs ``sample_id``, ``Sr`` and ``Nd`` and optionally
    ``Sr_err`` / ``Nd_err`` (absolute 1-sigma uncertainties).

    Returns ``(long_df, wide_df, diagnostics_df)``:

    * ``long_df``  - sample_id, region, prior, log_likelihood, posterior
    * ``wide_df``  - regions x samples matrix of posterior probabilities
    * ``diagnostics_df`` - per sample: most probable region, its
      probability, the smallest Mahalanobis distance to any region and a
      flag telling whether the sample lies outside the 95 % ellipse of
      every region (i.e. the posterior is a *relative* statement only).
    """
    if source_stats.empty:
        raise ValueError("No source region could be fitted.")
    samples = samples.dropna(subset=["sample_id", "Sr", "Nd"]).reset_index(drop=True)
    if samples.empty:
        raise ValueError("No valid samples (need sample ID, 87Sr/86Sr and 143Nd/144Nd).")

    regions = list(source_stats["region"])
    pri = _prior_vector(regions, priors)
    log_pri = np.log(np.where(pri > 0, pri, np.finfo(float).tiny))
    chi2_95 = chi2.ppf(0.95, df=2)

    records, diag = [], []
    for _, s in samples.iterrows():
        x = np.array([s["Sr"], s["Nd"]], dtype=float)
        sr_err = s.get("Sr_err", np.nan) if use_sample_errors else np.nan
        nd_err = s.get("Nd_err", np.nan) if use_sample_errors else np.nan
        sr_err = float(sr_err) if pd.notna(sr_err) else 0.0
        nd_err = float(nd_err) if pd.notna(nd_err) else 0.0
        sample_cov = np.diag([sr_err**2, nd_err**2])

        loglik = np.empty(len(regions))
        maha = np.empty(len(regions))
        for i, (_, rs) in enumerate(source_stats.iterrows()):
            mu = np.array([rs["mu_Sr"], rs["mu_Nd"]])
            cov = np.array([[rs["cov_SrSr"], rs["cov_SrNd"]], [rs["cov_SrNd"], rs["cov_NdNd"]]])
            cov_total = cov + sample_cov + np.eye(2) * cov_regularization
            loglik[i] = multivariate_normal(mean=mu, cov=cov_total, allow_singular=True).logpdf(x)
            diff = x - mu
            maha[i] = float(np.sqrt(max(diff @ np.linalg.pinv(cov_total) @ diff, 0.0)))

        log_unnorm = log_pri + loglik
        log_unnorm = np.where(pri > 0, log_unnorm, -np.inf)
        post = np.exp(log_unnorm - np.max(log_unnorm))
        post /= post.sum()

        for i, reg in enumerate(regions):
            records.append(
                {
                    "sample_id": s["sample_id"],
                    "region": reg,
                    "prior": pri[i],
                    "log_likelihood": loglik[i],
                    "mahalanobis_distance": maha[i],
                    "posterior_probability": post[i],
                }
            )
        best = int(np.argmax(post))
        diag.append(
            {
                "sample_id": s["sample_id"],
                "most_probable_source": regions[best],
                "max_posterior": post[best],
                "min_mahalanobis_distance": maha.min(),
                "outside_all_95pct_ellipses": bool(maha.min() ** 2 > chi2_95),
            }
        )

    long_df = pd.DataFrame(records)
    order = list(dict.fromkeys(samples["sample_id"]))
    wide_df = (
        long_df.pivot_table(index="region", columns="sample_id", values="posterior_probability", sort=False)
        .reindex(index=regions, columns=order)
    )
    wide_df.index.name = "Potential Source Area"
    wide_df.columns.name = "Sample"
    return long_df, wide_df, pd.DataFrame(diag)


def normalise_priors(regions: Iterable[str], values: Dict[str, float]) -> Dict[str, float]:
    regions = list(regions)
    p = _prior_vector(regions, values)
    return dict(zip(regions, p))
