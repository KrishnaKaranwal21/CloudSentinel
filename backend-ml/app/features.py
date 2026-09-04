"""
Single source of truth for turning a raw weather observation into the
model-ready feature vector. Used by BOTH train_model.py (on historical rows)
and model_service.py (on live/manual rows), so training and inference can
never disagree about column order or encoding.
"""
from __future__ import annotations

import datetime as dt
import math

import pandas as pd

from app.config import MONTH_DUMMY_COLUMNS, MONTHS, NUMERIC_FEATURES


def month_dummies(month_abbr: str) -> dict[str, int]:
    """One-hot encode a 3-letter month ('jan'..'dec'), 'jan' as the
    dropped reference category — mirrors the training-time encoding."""
    month_abbr = month_abbr.lower()[:3]
    if month_abbr not in MONTHS:
        raise ValueError(f"Unrecognised month abbreviation: {month_abbr!r}")
    return {col: int(col == f"month_{month_abbr}") for col in MONTH_DUMMY_COLUMNS}


def build_feature_row(values: dict, observed_at: dt.datetime) -> pd.DataFrame:
    """
    Build a single-row DataFrame with columns in the exact order the
    scaler/model expect: NUMERIC_FEATURES + MONTH_DUMMY_COLUMNS.

    `values` must contain every key in NUMERIC_FEATURES.
    """
    missing = [f for f in NUMERIC_FEATURES if f not in values]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    row = {f: float(values[f]) for f in NUMERIC_FEATURES}
    month_abbr = observed_at.strftime("%b").lower()
    row.update(month_dummies(month_abbr))

    ordered_columns = NUMERIC_FEATURES + MONTH_DUMMY_COLUMNS
    return pd.DataFrame([row], columns=ordered_columns)


def clean_clht(value: float) -> float:
    """999 ('no cloud ceiling recorded') is remapped to -1, matching the
    original script — keeps that observation numerically close to other
    'clear sky' rows instead of an outlier 999 far from everything else."""
    return -1.0 if value == 999 else value


def vapour_pressure_hpa(temp_c: float, rhum_pct: float) -> float:
    """Actual vapour pressure (hPa) from temperature + relative humidity,
    via the Magnus-Tetens approximation. Used to derive `vappr` when a
    data source (e.g. Open-Meteo) doesn't provide it directly."""
    es = 6.112 * math.exp((17.62 * temp_c) / (243.12 + temp_c))
    return es * (rhum_pct / 100.0)


def wet_bulb_c(temp_c: float, rhum_pct: float) -> float:
    """Wet bulb temperature (\u00b0C) via Stull's (2011) empirical
    approximation, valid for rhum in [5, 99]% and temp in [-20, 50]\u00b0C.
    Used to derive `wetb` when a data source doesn't provide it directly."""
    rh = max(5.0, min(99.0, rhum_pct))
    t = temp_c
    return (
        t * math.atan(0.151977 * math.sqrt(rh + 8.313659))
        + math.atan(t + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * (rh ** 1.5) * math.atan(0.023101 * rh)
        - 4.686035
    )
