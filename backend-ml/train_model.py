"""
Offline training script for SkyGuardAI's anomaly detector.

Run this ONCE (and again any time dublin.csv is updated) to (re)produce the
artifacts the FastAPI backend loads at startup:

    python train_model.py

This mirrors the cleaning + modelling steps of the original
anomalyDetection.py notebook (StandardScaler + IsolationForest on the 13
numeric weather features + one-hot month dummies), but:
  - handles missing values generically (not via hardcoded row indices),
  - persists the fitted scaler/model instead of refitting on every run,
  - records the training anomaly-score distribution so the API can turn a
    live score into a severity band (watch / warning / severe).
"""
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.config import (
    ARTIFACTS_DIR, BACKGROUND_PATH, DATA_PATH, FEATURE_META_PATH,
    ISOLATION_FOREST_PARAMS, MODEL_PATH, MONTH_DUMMY_COLUMNS,
    NUMERIC_FEATURES, SCALER_PATH, SHAP_BACKGROUND_SIZE, STATS_PATH,
)
from app.features import clean_clht, month_dummies


def load_and_clean(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, skiprows=23)

    # The raw file has 5 duplicate 'ind' quality-flag columns pandas
    # renames to ind, ind.1 .. ind.4 — not used by the model.
    ind_cols = [c for c in df.columns if c == "ind" or c.startswith("ind.")]
    df = df.drop(columns=ind_cols)

    # Some numeric columns contain literal " " (whitespace) for missing
    # readings. Generic fix: blank -> NaN -> forward-fill from the previous
    # hour (weather is highly autocorrelated hour-to-hour, so this is a
    # reasonable, if simple, imputation).
    for col in df.columns:
        if col == "date":
            continue
        df[col] = df[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[NUMERIC_FEATURES + ["ww", "w"]] = (
        df[NUMERIC_FEATURES + ["ww", "w"]].ffill().bfill()
    )

    df["clht"] = df["clht"].apply(clean_clht)

    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y %H:%M")
    df["month_abbr"] = df["date"].dt.strftime("%b").str.lower()

    return df


def build_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df[NUMERIC_FEATURES].reset_index(drop=True)
    dummies = pd.DataFrame(
        [month_dummies(m) for m in df["month_abbr"]],
        columns=MONTH_DUMMY_COLUMNS,
    )
    return pd.concat([numeric, dummies], axis=1)


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATA_PATH} ...")
    df = load_and_clean(DATA_PATH)
    print(f"  {len(df):,} rows after cleaning")

    X_raw = build_model_matrix(df)
    feature_columns = list(X_raw.columns)

    print("Fitting StandardScaler on the 13 numeric features ...")
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_raw[NUMERIC_FEATURES])
    X_scaled = pd.DataFrame(X_numeric_scaled, columns=NUMERIC_FEATURES)
    X_scaled = pd.concat([X_scaled, X_raw[MONTH_DUMMY_COLUMNS].reset_index(drop=True)], axis=1)
    X_scaled = X_scaled[feature_columns]  # enforce column order

    print("Fitting IsolationForest ...")
    clf = IsolationForest(**ISOLATION_FOREST_PARAMS)
    clf.fit(X_scaled)

    # decision_function: higher = more normal, lower/negative = more
    # anomalous. We flip sign so "anomaly_score": higher = more anomalous,
    # which reads more naturally in the API response.
    raw_scores = clf.decision_function(X_scaled)
    anomaly_scores = -raw_scores

    percentiles = {
        "p95": float(np.percentile(anomaly_scores, 95)),
        "p99": float(np.percentile(anomaly_scores, 99)),
        "max": float(np.max(anomaly_scores)),
        "mean": float(np.mean(anomaly_scores)),
        "std": float(np.std(anomaly_scores)),
    }

    feature_stats = {
        col: {"mean": float(X_raw[col].mean()), "std": float(X_raw[col].std())}
        for col in NUMERIC_FEATURES
    }

    # Small representative background sample for SHAP (keeps explanations
    # fast; SHAP only needs a reference distribution, not the full dataset).
    background = X_scaled.sample(
        n=min(SHAP_BACKGROUND_SIZE, len(X_scaled)), random_state=42
    ).reset_index(drop=True)

    print("Saving artifacts ...")
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(BACKGROUND_PATH, "wb") as f:
        pickle.dump(background, f)
    with open(FEATURE_META_PATH, "w") as f:
        json.dump({"feature_columns": feature_columns}, f, indent=2)
    with open(STATS_PATH, "w") as f:
        json.dump(
            {"anomaly_score_percentiles": percentiles, "feature_stats": feature_stats},
            f, indent=2,
        )

    n_anomalies = int((clf.predict(X_scaled) == -1).sum())
    print(f"Done. {n_anomalies:,} / {len(X_scaled):,} training rows flagged anomalous "
          f"({n_anomalies / len(X_scaled):.2%}).")
    print(f"Artifacts written to {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()
