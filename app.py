"""Bayesian Model for PSA - Streamlit web application.

Run locally with:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from psa import CHUR_DEFAULT, clean_text, compute_posteriors, eps_nd_from_ratio, fit_sources, ratio_from_eps_nd
from psa.io import guess_column, read_table, to_numeric
from psa.plotting import default_region_colors, fig_to_bytes, plot_posterior_heatmap, plot_region_diagram

APP_TITLE = "Bayesian Model for PSA"
EXAMPLES = Path(__file__).parent / "examples"
NONE = "— none —"

st.set_page_config(page_title=APP_TITLE, page_icon="🌍", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_table(data: bytes, sep_choice: str) -> pd.DataFrame:
    sep = {"Auto-detect": None, "Comma (,)": ",", "Semicolon (;)": ";", "Tab": "\t", "Pipe (|)": "|",
           "Whitespace": r"\s+"}[sep_choice]
    return read_table(data, sep=sep)


def file_input(label: str, key: str, example: Path):
    """Uploader + 'use example' toggle. Returns (bytes, name) or (None, None)."""
    c1, c2 = st.columns([3, 1])
    with c1:
        up = st.file_uploader(label, type=["csv", "txt", "tsv", "dat"], key=f"{key}_file")
    with c2:
        st.write("")
        use_ex = st.toggle("Use example data", key=f"{key}_example", value=False,
                           help="Synthetic demonstration data shipped with the tool.")
        sep_choice = st.selectbox("Delimiter", ["Auto-detect", "Comma (,)", "Semicolon (;)", "Tab",
                                                "Pipe (|)", "Whitespace"], key=f"{key}_sep")
    if up is not None:
        return up.getvalue(), up.name, sep_choice
    if use_ex and example.exists():
        return example.read_bytes(), example.name, sep_choice
    return None, None, sep_choice


def col_select(label, columns, guess=None, optional=False, key=None, help=None):
    opts = ([NONE] if optional else []) + list(columns)
    idx = opts.index(guess) if guess in opts else 0
    val = st.selectbox(label, opts, index=idx, key=key, help=help)
    return None if val == NONE else val


def make_unique(ids: pd.Series) -> pd.Series:
    counts: dict = {}
    out = []
    for v in ids:
        v = str(v)
        if v in counts:
            counts[v] += 1
            out.append(f"{v} ({counts[v]})")
        else:
            counts[v] = 1
            out.append(v)
    return pd.Series(out, index=ids.index)


def download_fig(fig, stem: str, key: str):
    cols = st.columns(3)
    for col, fmt in zip(cols, ["png", "pdf", "svg"]):
        with col:
            st.download_button(f"Download {fmt.upper()}", fig_to_bytes(fig, fmt, dpi=600 if fmt == "png" else 300),
                               file_name=f"{stem}.{fmt}", mime={"png": "image/png", "pdf": "application/pdf",
                                                              "svg": "image/svg+xml"}[fmt],
                               key=f"{key}_{fmt}", width="stretch")


# ---------------------------------------------------------------------------
# Header & sidebar
# ---------------------------------------------------------------------------
st.title(APP_TITLE)
st.caption("Bayesian provenance of samples from Potential Source Areas (PSAs) "
           "using ⁸⁷Sr/⁸⁶Sr and ¹⁴³Nd/¹⁴⁴Nd isotope signatures.")

with st.sidebar:
    st.header("Model settings")
    chur = st.number_input("CHUR ¹⁴³Nd/¹⁴⁴Nd", value=CHUR_DEFAULT, format="%.6f", step=0.000001,
                           help="Used to convert between 143Nd/144Nd and εNd(0).")
    robust = st.checkbox("Robust covariance (MCD)", value=True,
                         help="Minimum Covariance Determinant – reduces the influence of outliers "
                              "in the source data. Regions with < 4 points use a diagonal covariance.")
    use_err = st.checkbox("Propagate sample analytical errors", value=True,
                          help="Adds diag(σSr², σNd²) of each sample to the covariance of each source.")
    err_level = st.radio("Sample uncertainties are given as", ["1σ (absolute)", "2σ (absolute)"],
                         help="2σ values are divided by 2 before being used.")
    prior_mode = st.radio("Prior probabilities", ["Uniform", "Custom (edit in the PSA table)"])
    st.divider()
    with st.expander("How does the model work?"):
        st.markdown(
            r"""
For each selected PSA *k* a bivariate normal distribution
$\mathcal{N}(\mu_k, \Sigma_k)$ is fitted to its (⁸⁷Sr/⁸⁶Sr, ¹⁴³Nd/¹⁴⁴Nd) data.

For a sample $x$ with analytical covariance $S = \mathrm{diag}(\sigma_{Sr}^2, \sigma_{Nd}^2)$:

$$p(x\mid k)=\mathcal{N}(x;\,\mu_k,\,\Sigma_k+S)$$

$$P(k\mid x)=\frac{\pi_k\,p(x\mid k)}{\sum_j \pi_j\,p(x\mid j)}$$

$\pi_k$ are the priors (uniform by default). Posteriors are **relative**:
they sum to 1 over the PSAs you include, so a sample far from every
PSA can still get a high probability – check the *diagnostics* table.
"""
        )

tab_src, tab_smp, tab_res = st.tabs(["① Potential Source Areas", "② Samples", "③ Results"])

# ---------------------------------------------------------------------------
# 1. Source areas
# ---------------------------------------------------------------------------
sources = None
region_table = None
with tab_src:
    st.subheader("Upload reference data of the Potential Source Areas")
    st.markdown("One row per analysis. Required: a **region** column, **⁸⁷Sr/⁸⁶Sr** and "
                "**¹⁴³Nd/¹⁴⁴Nd** and/or **εNd(0)**.")
    data, name, sep_choice = file_input("PSA file (CSV / TXT)", "src", EXAMPLES / "example_sources_synthetic.csv")

    if data is not None:
        try:
            raw = load_table(data, sep_choice)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read **{name}**: {e}")
            raw = None
        if raw is not None and not raw.empty:
            st.success(f"**{name}** – {len(raw)} rows × {raw.shape[1]} columns")
            with st.expander("Preview data", expanded=False):
                st.dataframe(raw, width="stretch", height=250)

            cols = list(raw.columns)
            st.markdown("##### Column mapping")
            c1, c2, c3 = st.columns(3)
            with c1:
                region_col = col_select("Region / PSA name", cols, guess_column(cols, ["region", "psa", "source", "area"]),
                                        key="src_region")
            with c2:
                sr_col = col_select("⁸⁷Sr/⁸⁶Sr", cols, guess_column(cols, ["87sr/86sr", "87sr", "sr87", "sr"],
                                                                    exclude=["corr", "err"]), key="src_sr")
                sr_pref_col = col_select("Preferred ⁸⁷Sr/⁸⁶Sr (e.g. grain-size corrected)", cols,
                                         guess_column(cols, ["grain size corrected", "sr corrected", "corrected"]),
                                         optional=True, key="src_srpref",
                                         help="If set, this value is used when available and the column "
                                              "above is used as fallback.")
            with c3:
                nd_col = col_select("¹⁴³Nd/¹⁴⁴Nd", cols, guess_column(cols, ["143nd/144nd", "143nd", "nd143"],
                                                                      exclude=["err"]),
                                    optional=True, key="src_nd")
                eps_col = col_select("εNd(0)", cols, guess_column(cols, ["εnd", "epsnd", "eps nd", "end (0)", "end(0)",
                                                                         "nd (0)", "nd(0)", "epsilon"]),
                                     optional=True, key="src_eps")

            if nd_col is None and eps_col is None:
                st.error("Select at least one Nd column (¹⁴³Nd/¹⁴⁴Nd or εNd(0)).")
            elif region_col == sr_col:
                st.error("Region and Sr columns must be different.")
            else:
                sources = pd.DataFrame({"region": raw[region_col].apply(clean_text)})
                sr = to_numeric(raw[sr_col])
                if sr_pref_col:
                    sr = to_numeric(raw[sr_pref_col]).fillna(sr)
                sources["Sr"] = sr
                nd = to_numeric(raw[nd_col]) if nd_col else pd.Series(np.nan, index=raw.index)
                eps = to_numeric(raw[eps_col]) if eps_col else pd.Series(np.nan, index=raw.index)
                # fill each Nd representation from the other when missing
                sources["Nd"] = nd.fillna(pd.Series(ratio_from_eps_nd(eps, chur), index=raw.index))
                sources["epsNd"] = eps.fillna(pd.Series(eps_nd_from_ratio(nd, chur), index=raw.index))
                sources = sources.dropna(subset=["region", "Sr", "Nd"])

                if sources.empty:
                    st.error("No valid rows after cleaning – check the column mapping.")
                    sources = None
                else:
                    st.markdown("##### Select the Potential Source Areas")
                    st.caption("**Model** = used by the Bayesian classifier · **Diagram** = drawn in figure A. "
                               "You can also edit the prior weight (if *Custom* priors are on) and the colour (hex).")
                    counts = sources.groupby("region").size()
                    regions_all = list(counts.index)
                    colors = default_region_colors(regions_all)
                    base = pd.DataFrame({
                        "Region": regions_all,
                        "n": counts.values,
                        "Model": True,
                        "Diagram": True,
                        "Prior weight": 1.0,
                        "Colour": [colors[r] for r in regions_all],
                    })
                    b1, b2, _ = st.columns([1, 1, 4])
                    if b1.button("Select all", key="src_all"):
                        st.session_state["src_editor_ver"] = st.session_state.get("src_editor_ver", 0) + 1
                        st.session_state["src_default"] = True
                    if b2.button("Clear all", key="src_none"):
                        st.session_state["src_editor_ver"] = st.session_state.get("src_editor_ver", 0) + 1
                        st.session_state["src_default"] = False
                    base["Model"] = base["Diagram"] = st.session_state.get("src_default", True)
                    region_table = st.data_editor(
                        base,
                        hide_index=True,
                        width="stretch",
                        disabled=["Region", "n"],
                        column_config={
                            "n": st.column_config.NumberColumn("n points", help="Valid analyses in the region"),
                            "Model": st.column_config.CheckboxColumn("Model"),
                            "Diagram": st.column_config.CheckboxColumn("Diagram"),
                            "Prior weight": st.column_config.NumberColumn(min_value=0.0, step=0.1,
                                                                          disabled=prior_mode == "Uniform"),
                            "Colour": st.column_config.TextColumn(validate=r"^#[0-9A-Fa-f]{6}$"),
                        },
                        key=f"src_editor_{name}_{st.session_state.get('src_editor_ver', 0)}",
                    )
                    small = region_table[(region_table["Model"]) & (region_table["n"] < 4)]
                    if len(small):
                        st.warning("Few data points (< 4) in: " + ", ".join(small["Region"]) +
                                   ". A diagonal covariance is used for these regions; interpret with care.")
    else:
        st.info("Upload a file or switch on *Use example data*.")

# ---------------------------------------------------------------------------
# 2. Samples
# ---------------------------------------------------------------------------
samples = None
with tab_smp:
    st.subheader("Upload the samples to be classified")
    st.markdown("Required: **sample ID**, **⁸⁷Sr/⁸⁶Sr** and **¹⁴³Nd/¹⁴⁴Nd** (or εNd(0)). "
                "Optional: absolute uncertainties and a group column (used for colours/markers in figure A).")
    data, name, sep_choice = file_input("Samples file (CSV / TXT)", "smp", EXAMPLES / "example_samples_synthetic.csv")

    if data is not None:
        try:
            raw = load_table(data, sep_choice)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read **{name}**: {e}")
            raw = None
        if raw is not None and not raw.empty:
            st.success(f"**{name}** – {len(raw)} rows × {raw.shape[1]} columns")
            with st.expander("Preview data", expanded=False):
                st.dataframe(raw, width="stretch", height=250)

            cols = list(raw.columns)
            st.markdown("##### Column mapping")
            c1, c2, c3 = st.columns(3)
            with c1:
                id_col = col_select("Sample ID", cols, guess_column(cols, ["sample", "amostra", "id", "name"]),
                                    key="smp_id")
                grp_col = col_select("Group (optional)", cols, guess_column(cols, ["group", "grupo", "type", "class"]),
                                     optional=True, key="smp_grp")
            with c2:
                sr_col = col_select("⁸⁷Sr/⁸⁶Sr", cols, guess_column(cols, ["87sr/86sr", "87sr", "sr"],
                                                                    exclude=["err", "sd", "σ", "unc"]), key="smp_sr")
                sr_err_col = col_select("⁸⁷Sr/⁸⁶Sr uncertainty (abs.)", cols,
                                        guess_column(cols, ["87sr/86sr_err", "87sr/86sr err", "sr err", "sr_err",
                                                            "sr error", "sr erro", "sr sd", "sr 2s"]),
                                        optional=True, key="smp_srerr")
            with c3:
                nd_col = col_select("¹⁴³Nd/¹⁴⁴Nd", cols, guess_column(cols, ["143nd/144nd", "143nd", "nd"],
                                                                      exclude=["err", "sd", "σ", "unc", "ε", "eps"]),
                                    optional=True, key="smp_nd")
                eps_col = col_select("εNd(0) (used if ¹⁴³Nd/¹⁴⁴Nd missing)", cols,
                                     guess_column(cols, ["εnd", "epsnd", "eps nd", "epsilon"],
                                                  exclude=["err", "sd"]), optional=True, key="smp_eps")
                nd_err_col = col_select("¹⁴³Nd/¹⁴⁴Nd uncertainty (abs.)", cols,
                                        guess_column(cols, ["143nd/144nd_err", "143nd/144nd err", "nd err", "nd_err",
                                                            "nd error", "nd erro", "nd sd", "nd 2s"]),
                                        optional=True, key="smp_nderr")

            if nd_col is None and eps_col is None:
                st.error("Select at least one Nd column (¹⁴³Nd/¹⁴⁴Nd or εNd(0)).")
            else:
                s = pd.DataFrame({"sample_id": raw[id_col].apply(clean_text)})
                s["Sr"] = to_numeric(raw[sr_col])
                nd = to_numeric(raw[nd_col]) if nd_col else pd.Series(np.nan, index=raw.index)
                eps = to_numeric(raw[eps_col]) if eps_col else pd.Series(np.nan, index=raw.index)
                s["Nd"] = nd.fillna(pd.Series(ratio_from_eps_nd(eps, chur), index=raw.index))
                s["epsNd"] = eps_nd_from_ratio(s["Nd"], chur)
                div = 2.0 if err_level.startswith("2") else 1.0
                s["Sr_err"] = to_numeric(raw[sr_err_col]) / div if sr_err_col else np.nan
                s["Nd_err"] = to_numeric(raw[nd_err_col]) / div if nd_err_col else np.nan
                s["group"] = raw[grp_col].apply(clean_text) if grp_col else np.nan
                n_before = len(s)
                s = s.dropna(subset=["sample_id", "Sr", "Nd"]).copy()
                if len(s) < n_before:
                    st.warning(f"{n_before - len(s)} row(s) skipped (missing ID, Sr or Nd).")
                if s["sample_id"].duplicated().any():
                    st.warning("Duplicate sample IDs found – a numeric suffix was added.")
                    s["sample_id"] = make_unique(s["sample_id"])

                if s.empty:
                    st.error("No valid samples after cleaning – check the column mapping.")
                else:
                    st.markdown("##### Select the samples to analyse")
                    groups = sorted(s["group"].dropna().unique()) if grp_col else []
                    if groups:
                        chosen_groups = st.multiselect("Filter by group", groups, default=groups, key="smp_groups")
                        s = s[s["group"].isin(chosen_groups) | s["group"].isna()]
                    b1, b2, _ = st.columns([1, 1, 4])
                    if b1.button("Select all", key="smp_all"):
                        st.session_state["smp_editor_ver"] = st.session_state.get("smp_editor_ver", 0) + 1
                        st.session_state["smp_default"] = True
                    if b2.button("Clear all", key="smp_none"):
                        st.session_state["smp_editor_ver"] = st.session_state.get("smp_editor_ver", 0) + 1
                        st.session_state["smp_default"] = False
                    view = s[["sample_id", "group", "Sr", "Nd", "epsNd", "Sr_err", "Nd_err"]].copy()
                    view.insert(0, "Include", st.session_state.get("smp_default", True))
                    edited = st.data_editor(
                        view,
                        hide_index=True,
                        width="stretch",
                        disabled=[c for c in view.columns if c != "Include"],
                        column_config={
                            "sample_id": "Sample",
                            "group": "Group",
                            "Sr": st.column_config.NumberColumn("⁸⁷Sr/⁸⁶Sr", format="%.6f"),
                            "Nd": st.column_config.NumberColumn("¹⁴³Nd/¹⁴⁴Nd", format="%.6f"),
                            "epsNd": st.column_config.NumberColumn("εNd(0)", format="%.2f"),
                            "Sr_err": st.column_config.NumberColumn("σ Sr", format="%.2e"),
                            "Nd_err": st.column_config.NumberColumn("σ Nd", format="%.2e"),
                        },
                        key=f"smp_editor_{name}_{st.session_state.get('smp_editor_ver', 0)}",
                    )
                    samples = s[edited["Include"].to_numpy()].copy()
                    st.caption(f"{len(samples)} of {len(s)} samples selected.")
    else:
        st.info("Upload a file or switch on *Use example data*.")

# ---------------------------------------------------------------------------
# 3. Results
# ---------------------------------------------------------------------------
with tab_res:
    ready = True
    if sources is None or region_table is None:
        st.info("Step ① – load the Potential Source Area data.")
        ready = False
    if samples is None or samples.empty:
        st.info("Step ② – load and select at least one sample.")
        ready = False

    if ready:
        model_regions = list(region_table.loc[region_table["Model"], "Region"])
        diagram_regions = list(region_table.loc[region_table["Diagram"], "Region"])
        colors = dict(zip(region_table["Region"], region_table["Colour"]))
        priors = None
        if prior_mode.startswith("Custom"):
            priors = dict(zip(region_table["Region"], region_table["Prior weight"]))

        if len(model_regions) == 0:
            st.warning("Select at least one PSA for the model (column **Model** in step ①).")
        else:
            try:
                stats = fit_sources(sources, regions=model_regions, robust=robust)
                long_df, wide_df, diag = compute_posteriors(samples, stats, priors=priors, use_sample_errors=use_err)
            except Exception as e:  # noqa: BLE001
                st.error(f"Model failed: {e}")
                st.stop()

            with st.expander("Figure options", expanded=False):
                o1, o2, o3 = st.columns(3)
                with o1:
                    yvar = st.radio("Y axis of figure A", ["εNd(0)", "¹⁴³Nd/¹⁴⁴Nd"], horizontal=True)
                    ycol = "epsNd" if yvar.startswith("ε") else "Nd"
                    show_labels = st.checkbox("Label samples", value=True)
                    show_pts = st.checkbox("Show individual PSA data points", value=False)
                with o2:
                    manual = st.checkbox("Manual axis limits", value=False)
                    src_d = sources[sources["region"].isin(diagram_regions)]
                    allx = pd.concat([src_d["Sr"], samples["Sr"]])
                    ally = pd.concat([src_d[ycol], samples[ycol]])
                    pad_x = 0.03 * (allx.max() - allx.min() or 0.01)
                    pad_y = 0.03 * (ally.max() - ally.min() or 1.0)
                    xa = st.number_input("X min", value=float(allx.min() - pad_x), format="%.5f", disabled=not manual)
                    xb = st.number_input("X max", value=float(allx.max() + pad_x), format="%.5f", disabled=not manual)
                    ya = st.number_input("Y min", value=float(ally.min() - pad_y), format="%.5f", disabled=not manual)
                    yb = st.number_input("Y max", value=float(ally.max() + pad_y), format="%.5f", disabled=not manual)
                with o3:
                    cmap = st.selectbox("Heatmap colour map", ["viridis", "cividis", "magma", "Blues", "YlGnBu"])
                    annotate = st.checkbox("Show probabilities in heatmap", value=True)
                    panel_letters = st.checkbox("Panel letters (A, B)", value=True)
                    ncol = st.slider("Columns in PSA legend", 1, 5, 3)

            xlim = (xa, xb) if manual else (float(allx.min() - pad_x), float(allx.max() + pad_x))
            ylim = (ya, yb) if manual else (float(ally.min() - pad_y), float(ally.max() + pad_y))

            st.subheader("A · Potential Source Areas and samples")
            fig_a = plot_region_diagram(
                sources, samples, diagram_regions, y=ycol, region_colors=colors,
                group_col="group", xlim=xlim, ylim=ylim, show_labels=show_labels,
                show_source_points=show_pts, region_legend_ncol=ncol,
                panel_label="A" if panel_letters else None,
            )
            st.pyplot(fig_a, width="stretch")
            download_fig(fig_a, "PSA_region_diagram", "figA")

            st.subheader("B · Posterior probability of each PSA")
            fig_b = plot_posterior_heatmap(wide_df, cmap=cmap, annotate=annotate,
                                           panel_label="B" if panel_letters else None)
            st.pyplot(fig_b, width="stretch")
            download_fig(fig_b, "PSA_posterior_heatmap", "figB")

            st.subheader("Tables")
            outside = diag[diag["outside_all_95pct_ellipses"]]
            if len(outside):
                st.warning("These samples lie outside the 95 % ellipse of **every** selected PSA, so their "
                           "posteriors only say which PSA is *least unlikely*: " + ", ".join(outside["sample_id"]))
            t1, t2, t3, t4 = st.tabs(["Posterior matrix", "Diagnostics", "Long format", "PSA statistics"])
            with t1:
                st.dataframe(wide_df.style.format("{:.3f}").background_gradient(cmap=cmap, vmin=0, vmax=1),
                             width="stretch")
                st.download_button("Download CSV", wide_df.to_csv().encode(), "posterior_matrix.csv", "text/csv")
            with t2:
                st.dataframe(diag, width="stretch", hide_index=True)
                st.download_button("Download CSV", diag.to_csv(index=False).encode(), "diagnostics.csv", "text/csv")
            with t3:
                st.dataframe(long_df, width="stretch", hide_index=True)
                st.download_button("Download CSV", long_df.to_csv(index=False).encode(), "posterior_long.csv",
                                   "text/csv")
            with t4:
                st.dataframe(stats, width="stretch", hide_index=True)
                st.download_button("Download CSV", stats.to_csv(index=False).encode(), "psa_statistics.csv",
                                   "text/csv")

st.divider()
st.caption("Bayesian Model for PSA · open-source · see the GitHub repository for documentation and citation.")
