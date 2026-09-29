"""Figures: (A) isotope region diagram and (B) posterior-probability heatmap."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402
from scipy.interpolate import splev, splprep  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}

# Tableau-10 based palette (the first colours match the original study)
REGION_PALETTE = [
    "#4C78A8", "#F28E2B", "#E15759", "#59A14F", "#9C755F",
    "#D4A72C", "#B07AA1", "#7F7FCE", "#1F9AC9", "#FF9DA7",
    "#76B7B2", "#BAB0AC", "#8C564B", "#17BECF", "#BCBD22",
]
SAMPLE_COLORS = ["navy", "red", "#2CA02C", "#9467BD", "#FF7F0E", "#8C564B", "#E377C2", "#17BECF", "black"]
SAMPLE_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "<", ">", "*"]


def default_region_colors(regions: Sequence[str]) -> Dict[str, str]:
    return {r: REGION_PALETTE[i % len(REGION_PALETTE)] for i, r in enumerate(regions)}


def default_group_styles(groups: Sequence[str]) -> Dict[str, Tuple[str, str]]:
    """Colour and marker for each sample group."""
    return {
        g: (SAMPLE_COLORS[i % len(SAMPLE_COLORS)], SAMPLE_MARKERS[i % len(SAMPLE_MARKERS)])
        for i, g in enumerate(groups)
    }


# ---------------------------------------------------------------------------
# Smoothed region envelopes
# ---------------------------------------------------------------------------
def chaikin_smooth(points, refinements: int = 4) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    for _ in range(refinements):
        nxt = np.roll(pts, -1, axis=0)
        q = 0.75 * pts + 0.25 * nxt
        r = 0.25 * pts + 0.75 * nxt
        pts = np.empty((2 * len(pts), 2))
        pts[0::2], pts[1::2] = q, r
    return pts


def spline_closed_curve(points, n_points: int = 400, smooth: float = 1e-5) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return pts
    x = np.r_[pts[:, 0], pts[0, 0]]
    y = np.r_[pts[:, 1], pts[0, 1]]
    try:
        tck, _ = splprep([x, y], s=smooth, per=True)
        xn, yn = splev(np.linspace(0, 1, n_points), tck)
        return np.column_stack([xn, yn])
    except Exception:
        return pts


def add_region_polygon(ax, pts: np.ndarray, color: str, alpha: float = 0.21) -> None:
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) == 0:
        return
    if len(pts) == 1:
        ax.scatter(pts[:, 0], pts[:, 1], s=10, color=color, alpha=0.2, zorder=1)
        return
    if len(pts) == 2:
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=0.28, alpha=0.32, zorder=1)
        return
    try:
        # Hull in normalised coordinates: Sr and eps-Nd have very different scales
        span = np.ptp(pts, axis=0)
        span[span == 0] = 1.0
        lo = pts.min(axis=0)
        norm = (pts - lo) / span
        hull = ConvexHull(norm)
        smooth = chaikin_smooth(norm[hull.vertices], refinements=4)
        curve = spline_closed_curve(smooth, n_points=400, smooth=1e-5) * span + lo
        ax.add_patch(
            Polygon(curve, closed=True, facecolor=color, edgecolor=color,
                    linewidth=0.25, alpha=alpha, joinstyle="round", zorder=1)
        )
        ax.plot(curve[:, 0], curve[:, 1], color=color, lw=0.24, alpha=0.4, zorder=1.1)
    except Exception:
        ax.scatter(pts[:, 0], pts[:, 1], s=8, color=color, alpha=0.2, zorder=1)


# ---------------------------------------------------------------------------
# Figure A - region diagram
# ---------------------------------------------------------------------------
def plot_region_diagram(
    sources: pd.DataFrame,
    samples: pd.DataFrame,
    regions: Sequence[str],
    y: str = "epsNd",
    region_colors: Optional[Dict[str, str]] = None,
    group_col: Optional[str] = "group",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    show_labels: bool = True,
    show_source_points: bool = False,
    region_legend_ncol: int = 3,
    figsize: Tuple[float, float] = (14, 9),
    dpi: int = 150,
    panel_label: Optional[str] = "A",
):
    """87Sr/86Sr vs epsNd(0) (or 143Nd/144Nd) with smoothed PSA envelopes.

    ``sources``: columns region, Sr, and ``y``.
    ``samples``: columns sample_id, Sr, ``y`` and optionally ``group_col``.
    """
    region_colors = region_colors or default_region_colors(regions)
    with mpl.rc_context(STYLE), sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        src = sources.dropna(subset=["region", "Sr", y])
        for reg in regions:
            pts = src.loc[src["region"] == reg, ["Sr", y]].to_numpy(dtype=float)
            col = region_colors.get(reg, "#999999")
            add_region_polygon(ax, pts, col)
            if show_source_points and len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], s=6, color=col, alpha=0.55, lw=0, zorder=1.2)

        smp = samples.dropna(subset=["Sr", y])
        has_groups = group_col and group_col in smp and smp[group_col].notna().any()
        if has_groups:
            groups = list(dict.fromkeys(smp[group_col].fillna("Other").astype(str)))
        else:
            groups = ["Samples"]
        styles = default_group_styles(groups)

        # label offset proportional to the axis span
        dx = 0.007 * (xlim[1] - xlim[0]) if xlim else 0.007 * max(np.ptp(smp["Sr"]) if len(smp) else 0, 1e-3)
        yspan = (ylim[1] - ylim[0]) if ylim else max(np.ptp(smp[y]) if len(smp) else 0, 1.0)
        dy = 0.009 * yspan

        for _, row in smp.iterrows():
            if has_groups:
                g = str(row[group_col]) if pd.notna(row[group_col]) else "Other"
            else:
                g = groups[0]
            color, marker = styles.get(g, ("black", "o"))
            ax.scatter(row["Sr"], row[y], s=80, marker=marker, facecolors=color,
                       edgecolors="black", linewidths=0.40, zorder=5)
            if show_labels:
                ax.text(row["Sr"] + dx, row[y] + dy, str(row["sample_id"]),
                        fontsize=13, color="0.20", zorder=6)

        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel(r"$^{87}$Sr/$^{86}$Sr")
        ax.set_ylabel(r"$\varepsilon$Nd(0)" if y == "epsNd" else r"$^{143}$Nd/$^{144}$Nd")
        ax.tick_params(axis="both", direction="out")
        if y == "Nd":
            ax.ticklabel_format(axis="y", useOffset=False, style="plain")
        ax.ticklabel_format(axis="x", useOffset=False, style="plain")

        sample_handles = [
            Line2D([0], [0], marker=styles[g][1], linestyle="None", markerfacecolor=styles[g][0],
                   markeredgecolor="black", markeredgewidth=0.40, markersize=13, label=g)
            for g in groups
        ]
        leg1 = ax.legend(handles=sample_handles, loc="upper right", frameon=False,
                         borderpad=0.2, handletextpad=0.5)
        ax.add_artist(leg1)

        bg_handles = [
            Patch(facecolor=region_colors.get(r, "#999999"), edgecolor=region_colors.get(r, "#999999"),
                  linewidth=0.34, alpha=0.18, label=r)
            for r in regions
        ]
        if bg_handles:
            ax.legend(handles=bg_handles, loc="lower left", frameon=False, ncol=region_legend_ncol,
                      columnspacing=1.0, handlelength=1.2, borderpad=0.2)

        if panel_label:
            ax.text(-0.12, 1.0, panel_label, transform=ax.transAxes, ha="left", va="top",
                    fontsize=24, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# Figure B - posterior heatmap
# ---------------------------------------------------------------------------
def plot_posterior_heatmap(
    wide: pd.DataFrame,
    cmap: str = "viridis",
    annotate: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
    panel_label: Optional[str] = "B",
):
    """Heatmap of posterior probabilities (rows = PSAs, columns = samples)."""
    n_reg, n_smp = wide.shape
    if figsize is None:
        figsize = (max(8.0, min(0.75 * n_smp + 5, 40)), max(4.5, 0.6 * n_reg + 2.5))
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
        sns.heatmap(
            wide.astype(float),
            cmap=cmap,
            vmin=0,
            vmax=1,
            linewidths=0.3,
            linecolor="white",
            annot=annotate,
            fmt=".2f",
            annot_kws={"fontsize": 11},
            cbar_kws={"label": "Posterior probability"},
            ax=ax,
        )
        ax.set_xlabel("Sample")
        ax.set_ylabel("Potential Source Area")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
            lbl.set_rotation_mode("anchor")
        if panel_label:
            ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=24, fontweight="bold")
    return fig


def fig_to_bytes(fig, fmt: str = "png", dpi: int = 300) -> bytes:
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


__all__ = [
    "plot_region_diagram",
    "plot_posterior_heatmap",
    "default_region_colors",
    "default_group_styles",
    "fig_to_bytes",
]
