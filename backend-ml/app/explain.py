"""
Turns the numeric output of AnomalyModelService.explain() into plain-English
text: WHAT the anomaly is, its likely ROOT CAUSE (from the SHAP-ranked
features), and HOW BAD it is (severity/warning level).
"""
from __future__ import annotations

from app.config import FEATURE_INFO

SEVERITY_TEXT = {
    "normal":  "No anomaly detected. Conditions are consistent with typical patterns for this station.",
    "watch":   "Conditions are somewhat unusual but still within the model's normal range. Worth keeping an eye on.",
    "warning": "This observation is a statistical outlier compared to 30 years of station history. Review recommended.",
    "severe":  "This observation is among the most extreme the model has seen. Immediate review recommended.",
}

SEVERITY_LEVEL_LABEL = {
    "normal": "NORMAL",
    "watch": "WATCH",
    "warning": "WARNING",
    "severe": "SEVERE",
}

# Lightweight, human-readable pattern hints. These are heuristic and are
# only ever presented as *possible* patterns, never asserted as fact — the
# actual "why" is always the SHAP-ranked feature list, this just adds a
# familiar name for readers when the pattern is recognisable.
_PATTERN_RULES = [
    ({"vis", "rhum", "dewpt"}, "conditions consistent with fog or very low cloud"),
    ({"wdsp", "msl"}, "conditions consistent with a low-pressure / storm system"),
    ({"rain", "rhum"}, "conditions consistent with heavy or sustained precipitation"),
    ({"temp", "wetb", "dewpt"}, "conditions consistent with an unusual temperature swing"),
    ({"clamt", "sun"}, "conditions consistent with an atypical cloud/sunshine pattern"),
]


def _direction(z: float) -> str:
    if z >= 2:
        return "unusually high"
    if z >= 1:
        return "somewhat high"
    if z <= -2:
        return "unusually low"
    if z <= -1:
        return "somewhat low"
    return "close to typical"


def _feature_sentence(contribution: dict, feature_stats: dict) -> str:
    col = contribution["feature"]
    info = FEATURE_INFO[col]
    stats = feature_stats[col]
    value = contribution["raw_value"]
    z = (value - stats["mean"]) / stats["std"] if stats["std"] else 0.0
    direction = _direction(z)
    return (
        f"{info['label']} is {direction} at {value:g}{info['unit']} "
        f"(typical average \u2248 {stats['mean']:.1f}{info['unit']}, "
        f"{abs(z):.1f}\u03c3 from normal)."
    )


def _likely_pattern(top_features: set[str]) -> str | None:
    for trigger_set, description in _PATTERN_RULES:
        if trigger_set.issubset(top_features) or len(trigger_set & top_features) >= 2:
            return description
    return None


def build_explanation(result: dict, feature_stats: dict) -> dict:
    """
    result: the dict returned by AnomalyModelService.explain()
    Returns a dict with 'what', 'root_cause', 'severity_text', 'severity_level'.
    """
    severity = result["severity"]
    top = result["top_contributions"]

    if not result["is_anomaly"] and severity == "normal":
        what = "This observation looks like normal weather for this station and time of year."
        root_cause = "No feature stood out as a significant driver \u2014 nothing here deviates meaningfully from the historical pattern."
    else:
        feature_sentences = [_feature_sentence(c, feature_stats) for c in top if c["shap_value"] > 0] or \
                             [_feature_sentence(c, feature_stats) for c in top]
        what = (
            "This observation is flagged as an anomaly." if result["is_anomaly"]
            else "This observation is unusual, though not extreme enough to be flagged as a full anomaly."
        )
        root_cause = " ".join(feature_sentences)
        pattern = _likely_pattern({c["feature"] for c in top})
        if pattern:
            root_cause += f" Overall pattern: {pattern} (heuristic guess based on the combination of drivers above, not a certainty)."

    return {
        "what": what,
        "root_cause": root_cause,
        "severity_level": SEVERITY_LEVEL_LABEL[severity],
        "severity_text": SEVERITY_TEXT[severity],
    }
