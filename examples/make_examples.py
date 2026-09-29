"""Generate the SYNTHETIC example files used by the demo mode.

These numbers are random draws, not real measurements. They only exist
so that new users can try the tool without their own data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

CHUR = 0.512638
rng = np.random.default_rng(42)
here = Path(__file__).parent

# name: (mean Sr, mean epsNd, sd Sr, sd epsNd, correlation, n)
regions = {
    "Volcanic Arc": (0.7062, -1.5, 0.0009, 1.4, -0.5, 30),
    "Northern Plateau": (0.7118, -7.5, 0.0020, 1.8, -0.6, 25),
    "Southern Plateau": (0.7090, -4.8, 0.0015, 1.5, -0.4, 25),
    "Coastal Basin": (0.7195, -10.5, 0.0030, 2.0, -0.5, 20),
    "Foreland": (0.7260, -13.5, 0.0040, 2.0, -0.4, 20),
    "Craton": (0.7380, -19.0, 0.0060, 2.5, -0.3, 18),
    "Isolated Outcrop": (0.7150, -9.0, 0.0010, 1.0, 0.0, 3),
}
rows = []
for name, (msr, meps, ssr, seps, rho, n) in regions.items():
    cov = [[ssr**2, rho * ssr * seps], [rho * ssr * seps, seps**2]]
    X = rng.multivariate_normal([msr, meps], cov, size=n)
    for sr, eps in X:
        nd = (eps / 1e4 + 1) * CHUR
        rows.append({"Region": name, "87Sr/86Sr": round(sr, 6), "143Nd/144Nd": round(nd, 6),
                     "eNd(0)": round(eps, 2)})
pd.DataFrame(rows).to_csv(here / "example_sources_synthetic.csv", index=False)

samples = [
    ("S1-01", "Site 1", 0.7105, -6.9), ("S1-02", "Site 1", 0.7112, -7.4),
    ("S1-03", "Site 1", 0.7093, -5.3), ("S2-01", "Site 2", 0.7188, -10.1),
    ("S2-02", "Site 2", 0.7240, -12.8), ("S2-03", "Site 2", 0.7171, -9.6),
    ("F-01", "Fine fraction", 0.7071, -2.2), ("F-02", "Fine fraction", 0.7130, -8.1),
    ("F-03", "Fine fraction", 0.7310, -16.0),
]
out = []
for sid, grp, sr, eps in samples:
    nd = (eps / 1e4 + 1) * CHUR
    # semicolon-separated with decimal commas, to demonstrate robust parsing
    out.append({"Sample": sid, "Group": grp,
                "87Sr/86Sr": f"{sr:.6f}".replace(".", ","),
                "87Sr/86Sr_err (2s)": "0,000012",
                "143Nd/144Nd": f"{nd:.6f}".replace(".", ","),
                "143Nd/144Nd_err (2s)": "0,000010"})
pd.DataFrame(out).to_csv(here / "example_samples_synthetic.csv", index=False, sep=";")
print("written")
