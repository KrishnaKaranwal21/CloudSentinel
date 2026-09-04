"""
Loads the trained scaler/model/background once (at process startup) and
exposes a single `explain(values, observed_at)` call that FastAPI endpoints
use. Keeping this as one long-lived object avoids re-loading pickles or
re-building the SHAP explainer on every request.
"""
from __future__ import annotations

import datetime as dt
import json
import pickle

import numpy as np
import pandas as pd
import shap

from app.config import (
    BACKGROUND_PATH, FEATURE_META_PATH, MODEL_PATH, NUMERIC_FEATURES,
    SCALER_PATH, SEVERITY_PERCENTILE_BANDS, STATS_PATH,
    TOP_K_CONTRIBUTING_FEATURES,
)
from app.features import build_feature_row


class ModelNotTrainedError(RuntimeError):
    """Raised when artifacts/ is empty — run train_model.py first."""


class AnomalyModelService:
    def __init__(self):
        for path in (SCALER_PATH, MODEL_PATH, FEATURE_META_PATH, STATS_PATH, BACKGROUND_PATH):
            if not path.exists():
                raise ModelNotTrainedError(
                    f"Missing artifact: {path}. Run `python train_model.py` first."
                )

        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(BACKGROUND_PATH, "rb") as f:
            self.background = pickle.load(f)
        with open(FEATURE_META_PATH) as f:
            self.feature_columns = json.load(f)["feature_columns"]
        with open(STATS_PATH) as f:
            stats = json.load(f)
        self.score_percentiles = stats["anomaly_score_percentiles"]
        self.feature_stats = stats["feature_stats"]

        # TreeExplainer natively understands sklearn's IsolationForest and
        # is fast/exact. If a different sklearn/shap version combination
        # doesn't support it, fall back to a model-agnostic explainer built
        # on the anomaly score function directly (slower, still correct).
        try:
            self.explainer = shap.TreeExplainer(self.model)
            self._explainer_kind = "tree"
        except Exception:
            self.explainer = shap.Explainer(self._score_fn, self.background)
            self._explainer_kind = "permutation"

    def _score_fn(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score where HIGHER = more anomalous (flipped sign vs
        sklearn's decision_function, which is higher = more normal)."""
        df = pd.DataFrame(X, columns=self.feature_columns)
        return -self.model.decision_function(df)

    def _scale(self, row_df: pd.DataFrame) -> pd.DataFrame:
        scaled_numeric = self.scaler.transform(row_df[NUMERIC_FEATURES])
        scaled = pd.DataFrame(scaled_numeric, columns=NUMERIC_FEATURES)
        month_cols = [c for c in self.feature_columns if c not in NUMERIC_FEATURES]
        scaled = pd.concat([scaled, row_df[month_cols].reset_index(drop=True)], axis=1)
        return scaled[self.feature_columns]

    def severity_for_score(self, score: float, is_flagged_anomaly: bool) -> str:
        p = self.score_percentiles
        if score >= p["max"] or is_flagged_anomaly and score >= p["p99"]:
            return "severe"
        if score >= p["p99"]:
            return "warning"
        if score >= p["p95"]:
            return "watch"
        return "normal"

    def explain(self, raw_values: dict, observed_at: dt.datetime | None = None) -> dict:
        """
        raw_values: dict with the 13 NUMERIC_FEATURES (see app.config).
        Returns a fully structured result: prediction, score, severity,
        and a ranked list of contributing features with plain-language text.
        """
        observed_at = observed_at or dt.datetime.utcnow()
        row_df = build_feature_row(raw_values, observed_at)
        scaled_row = self._scale(row_df)

        raw_decision = float(self.model.decision_function(scaled_row)[0])
        anomaly_score = -raw_decision
        is_anomaly = bool(self.model.predict(scaled_row)[0] == -1)
        severity = self.severity_for_score(anomaly_score, is_anomaly)

        if self._explainer_kind == "tree":
            # TreeExplainer explains decision_function directly, where
            # HIGHER = more normal. Negate so shap_value follows the same
            # convention as anomaly_score: positive = pushes toward anomaly.
            shap_values = -np.asarray(self.explainer.shap_values(scaled_row)[0])
        else:
            # self._score_fn is already defined as higher = more anomalous,
            # so no sign flip needed here.
            shap_values = self.explainer(scaled_row.values)[0].values

        contributions = []
        for col, val in zip(self.feature_columns, shap_values):
            if col not in NUMERIC_FEATURES:
                continue  # month dummies aren't reported as "causes" to the user
            contributions.append({
                "feature": col,
                "raw_value": float(raw_values[col]),
                "shap_value": float(val),
            })
        contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
        top_contributions = contributions[:TOP_K_CONTRIBUTING_FEATURES]

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 4),
            "severity": severity,
            "raw_values": raw_values,
            "top_contributions": top_contributions,
            "all_shap_values": {c["feature"]: round(c["shap_value"], 5) for c in contributions},
        }


_service: AnomalyModelService | None = None


def get_model_service() -> AnomalyModelService:
    """Lazy singleton so the (slow-ish) SHAP explainer setup happens once."""
    global _service
    if _service is None:
        _service = AnomalyModelService()
    return _service
