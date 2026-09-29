"""Robust reading and cleaning of user-supplied CSV / TXT tables."""

from __future__ import annotations

import csv
import io
import re
from typing import IO, Iterable, Optional, Union

import numpy as np
import pandas as pd

MISSING_TOKENS = {"", "NA", "N/A", "NaN", "nan", "None", "null", "-", "--"}
DELIMITERS = [",", ";", "\t", "|"]


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def sniff_separator(text: str, nbytes: int = 10000) -> str:
    """Guess the column delimiter of a delimited text sample."""
    sample = text[:nbytes]
    try:
        return csv.Sniffer().sniff(sample, delimiters=DELIMITERS).delimiter
    except Exception:
        counts = {d: sample.count(d) for d in DELIMITERS}
        return max(counts, key=counts.get)


def read_table(
    source: Union[str, bytes, IO],
    sep: Optional[str] = None,
) -> pd.DataFrame:
    """Read a CSV/TXT table (path, bytes or file-like) as strings.

    The delimiter is detected automatically unless ``sep`` is given.
    Every value is kept as text so that numbers written with decimal
    commas (e.g. ``0,7123``) can be converted later by
    :func:`smart_to_float`.
    """
    if isinstance(source, (bytes, bytearray)):
        text = _decode(bytes(source))
    elif isinstance(source, str):
        with open(source, "rb") as f:
            text = _decode(f.read())
    else:
        raw = source.read()
        text = _decode(raw) if isinstance(raw, (bytes, bytearray)) else raw

    if sep is None:
        sep = sniff_separator(text)

    df = pd.read_csv(
        io.StringIO(text),
        sep=sep,
        engine="python",
        on_bad_lines="skip",
        dtype=str,
        skipinitialspace=True,
    )
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")
    return df.loc[:, ~df.columns.str.startswith("Unnamed")] if len(df.columns) else df


def clean_text(x):
    """Strip whitespace; map empty / missing tokens to NaN."""
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    return np.nan if x in MISSING_TOKENS else x


def smart_to_float(x) -> float:
    """Convert a string to float, handling decimal commas and thousand marks."""
    if pd.isna(x):
        return np.nan
    x = str(x).strip().replace(" ", "").replace("\xa0", "")
    if x in MISSING_TOKENS:
        return np.nan
    # Unicode minus signs
    x = x.replace("−", "-").replace("–", "-")
    if "," in x and "." in x:
        if x.rfind(",") > x.rfind("."):
            x = x.replace(".", "").replace(",", ".")
        else:
            x = x.replace(",", "")
    elif "," in x:
        x = x.replace(",", ".")
    x = re.sub(r"[^0-9eE\+\-\.]", "", x)
    try:
        return float(x)
    except ValueError:
        return np.nan


def to_numeric(series: pd.Series) -> pd.Series:
    return series.apply(smart_to_float).astype(float)


def normalize_colname(c) -> str:
    c = str(c).strip().lower().replace("\xa0", " ")
    return re.sub(r"\s+", " ", c)


def guess_column(
    columns: Iterable[str],
    candidates: Iterable[str],
    exclude: Iterable[str] = (),
) -> Optional[str]:
    """Return the first column whose normalised name contains a candidate.

    Columns whose name contains any of ``exclude`` are skipped. Used only
    to pre-fill the column selectors; the user can always change them.
    """
    columns = list(columns)
    exclude = [normalize_colname(e) for e in exclude]
    norm = {col: normalize_colname(col) for col in columns}
    for cand in candidates:
        cn = normalize_colname(cand)
        for orig, n in norm.items():
            if cn in n and not any(e in n for e in exclude):
                return orig
    return None
