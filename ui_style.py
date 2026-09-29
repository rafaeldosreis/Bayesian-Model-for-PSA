"""Visual styling for the Streamlit interface (CSS, header, footer)."""

from __future__ import annotations

import streamlit as st

REPO_URL = "https://github.com/rafaeldosreis/Bayesian-Model-for-PSA"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
}
.block-container, [data-testid="stMainBlockContainer"] { padding-top: 3.6rem; max-width: 1250px; }

/* ---------- hero ---------- */
.psa-hero {
    background: linear-gradient(120deg, #0F3D56 0%, #1F6F8B 55%, #3E9E8F 100%);
    border-radius: 18px;
    padding: 2.1rem 2.4rem 1.9rem 2.4rem;
    color: #FFFFFF;
    margin-bottom: 1.4rem;
    box-shadow: 0 8px 24px rgba(15, 61, 86, 0.18);
    position: relative;
    overflow: hidden;
}
.psa-hero::after {
    content: "";
    position: absolute; right: -60px; top: -60px;
    width: 260px; height: 260px; border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.16) 0%, rgba(255,255,255,0) 70%);
}
.psa-hero h1 {
    color: #FFFFFF; font-weight: 700; font-size: 2.3rem;
    margin: 0 0 0.35rem 0; padding: 0; letter-spacing: -0.02em;
}
.psa-hero p.lead { color: rgba(255,255,255,0.88); font-size: 1.05rem; margin: 0 0 1.3rem 0; max-width: 780px; }
.psa-credit {
    position: absolute; top: 1.1rem; right: 1.4rem; z-index: 2; text-align: right;
    background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.30);
    border-radius: 12px; padding: 0.45rem 0.85rem; line-height: 1.25;
}
.psa-credit small { display: block; font-size: 0.68rem; letter-spacing: 0.06em;
    text-transform: uppercase; color: rgba(255,255,255,0.75); }
.psa-credit b { font-size: 0.92rem; color: #FFFFFF; font-weight: 600; }
@media (max-width: 760px) {
    .psa-credit { position: static; display: inline-block; text-align: left; margin-bottom: 0.8rem; }
}
.psa-badge {
    display: inline-block; background: rgba(255,255,255,0.16); color: #FFFFFF;
    border: 1px solid rgba(255,255,255,0.28); border-radius: 999px;
    padding: 0.18rem 0.7rem; font-size: 0.78rem; font-weight: 500; margin-bottom: 0.8rem;
}
.psa-steps { display: flex; gap: 0.8rem; flex-wrap: wrap; }
.psa-step {
    flex: 1 1 200px; background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.22); border-radius: 12px; padding: 0.75rem 0.95rem;
}
.psa-step b { display: block; font-size: 0.95rem; color: #FFFFFF; }
.psa-step span { font-size: 0.83rem; color: rgba(255,255,255,0.82); }
.psa-step .num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 1.5rem; height: 1.5rem; border-radius: 50%; background: #FFFFFF; color: #0F3D56;
    font-weight: 700; font-size: 0.8rem; margin-right: 0.45rem;
}

/* ---------- tabs ---------- */
[role="tablist"] { gap: 0.5rem; }
[data-testid="stTab"] {
    background: #F3F6F8; border: 1px solid #E3E8EC; border-radius: 999px;
    padding: 0.45rem 1.1rem !important; height: auto !important; min-height: 2.4rem;
}
[data-testid="stTab"] p { font-weight: 600; font-size: 0.95rem; white-space: nowrap; }
[data-testid="stTab"][aria-selected="true"] { background: #1F6F8B; border-color: #1F6F8B; }
[data-testid="stTab"][aria-selected="true"] p { color: #FFFFFF !important; }
[data-testid="stTab"] .react-aria-SelectionIndicator { display: none; }

/* ---------- headings & cards ---------- */
h3 { color: #0F3D56; font-weight: 700 !important; letter-spacing: -0.01em; }
h5 { color: #1F6F8B; font-weight: 600 !important; margin-top: 0.6rem; }
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #9CC3CF; background: #F5FAFB; border-radius: 12px;
}
[data-testid="stExpander"] { border-radius: 12px; }
[data-testid="stMetric"] {
    background: #F3F6F8; border: 1px solid #E3E8EC; border-radius: 12px; padding: 0.8rem 1rem;
}
[data-testid="stMetricValue"] { color: #0F3D56; font-weight: 700; }
.stDownloadButton button, .stButton button { border-radius: 10px; font-weight: 500; }
[data-testid="stImage"] img, [data-testid="stPyplot"] img { border-radius: 10px; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: #F3F6F8; }
.psa-side-brand { font-weight: 700; font-size: 1.1rem; color: #0F3D56; margin-bottom: 0.1rem; }
.psa-side-sub { font-size: 0.8rem; color: #5B6B78; margin-bottom: 0.6rem; }

/* ---------- footer ---------- */
.psa-footer {
    margin-top: 2.5rem; padding: 1.2rem 0 0.4rem 0; border-top: 1px solid #E3E8EC;
    font-size: 0.85rem; color: #5B6B78; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.6rem;
}
.psa-footer a { color: #1F6F8B; text-decoration: none; font-weight: 500; }
</style>
"""

HERO = """
<div class="psa-hero">
  <div class="psa-credit"><small>Created and developed by</small><b>Dr. Rafael dos Reis</b></div>
  <div class="psa-badge">Open-source · ⁸⁷Sr/⁸⁶Sr – ¹⁴³Nd/¹⁴⁴Nd provenance</div>
  <h1>Bayesian Model for PSA</h1>
  <p class="lead">Estimate the probability that each sample comes from each Potential Source Area.
  Upload your own data, choose the source areas and samples, and download publication-ready figures.</p>
  <div class="psa-steps">
    <div class="psa-step"><b><span class="num">1</span>Source areas</b>
      <span>Upload PSA reference data and pick the regions to model</span></div>
    <div class="psa-step"><b><span class="num">2</span>Samples</b>
      <span>Upload your samples and select which ones to analyse</span></div>
    <div class="psa-step"><b><span class="num">3</span>Results</b>
      <span>Isotope diagram, probability heatmap and tables</span></div>
  </div>
</div>
"""

FOOTER = f"""
<div class="psa-footer">
  <div>Bayesian Model for PSA · Created and developed by Dr. Rafael dos Reis · MIT License · Uploaded data are processed in memory for your session only and are never stored.</div>
  <div><a href="{REPO_URL}" target="_blank">GitHub</a> ·
       <a href="{REPO_URL}#the-model" target="_blank">Method</a> ·
       <a href="{REPO_URL}/issues" target="_blank">Report an issue</a></div>
</div>
"""


def apply_style() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero() -> None:
    st.markdown(HERO, unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.markdown('<div class="psa-side-brand">🌍 Bayesian Model for PSA</div>'
                '<div class="psa-side-sub">Potential Source Area provenance</div>', unsafe_allow_html=True)


def footer() -> None:
    st.markdown(FOOTER, unsafe_allow_html=True)
