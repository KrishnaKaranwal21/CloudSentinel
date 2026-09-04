"""
Central configuration for SkyGuardAI backend-ml.

Keeping every "magic value" (feature order, station coordinates, artifact
paths) in one place so training and serving can never silently drift apart.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_PATH = BASE_DIR / "dublin.csv"

SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
FEATURE_META_PATH = ARTIFACTS_DIR / "feature_meta.json"
BACKGROUND_PATH = ARTIFACTS_DIR / "shap_background.pkl"
STATS_PATH = ARTIFACTS_DIR / "training_stats.json"

# ---------------------------------------------------------------------------
# Station / Open-Meteo location
# Dublin Airport — matches the training data (dublin.csv) exactly.
# Override via query params on the /predict/live endpoint if needed.
# ---------------------------------------------------------------------------
STATION_NAME = "Dublin Airport"
STATION_LAT = 53.428
STATION_LON = -6.241

# ---------------------------------------------------------------------------
# The 13 numeric weather features the model was trained on, in this exact
# order (mirrors `to_model_columns` in the original anomalyDetection.py).
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "rain", "temp", "wetb", "dewpt", "vappr", "rhum",
    "msl", "wdsp", "wddir", "sun", "vis", "clht", "clamt",
]

# Human-readable labels + units, sourced from KeyHourly.txt
FEATURE_INFO = {
    "rain":  {"label": "Precipitation amount", "unit": "mm"},
    "temp":  {"label": "Air temperature", "unit": "\u00b0C"},
    "wetb":  {"label": "Wet bulb temperature", "unit": "\u00b0C"},
    "dewpt": {"label": "Dew point temperature", "unit": "\u00b0C"},
    "vappr": {"label": "Vapour pressure", "unit": "hPa"},
    "rhum":  {"label": "Relative humidity", "unit": "%"},
    "msl":   {"label": "Mean sea level pressure", "unit": "hPa"},
    "wdsp":  {"label": "Mean wind speed", "unit": "kt"},
    "wddir": {"label": "Wind direction", "unit": "\u00b0"},
    "sun":   {"label": "Sunshine duration", "unit": "hours"},
    "vis":   {"label": "Visibility", "unit": "m"},
    "clht":  {"label": "Cloud ceiling height", "unit": "100s ft (-1 = no clouds)"},
    "clamt": {"label": "Cloud amount", "unit": "okta"},
}

# Months used as one-hot dummies, "jan" dropped as the reference category —
# mirrors `pd.get_dummies(...).drop('jan', axis=1)` in the original script.
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]
MONTH_DUMMY_COLUMNS = [f"month_{m}" for m in MONTHS if m != "jan"]

# IsolationForest hyperparameters — identical to the original script.
# (`behaviour="new"` was removed from modern scikit-learn; "new" has been
# the only behaviour since sklearn 0.22, so dropping the kwarg is safe.)
ISOLATION_FOREST_PARAMS = dict(
    contamination=0.01,
    n_jobs=-1,
    random_state=42,
)

# Number of representative rows kept from training data as the SHAP
# background/reference distribution (keeps SHAP explanations fast).
SHAP_BACKGROUND_SIZE = 100

# Severity is assigned from the anomaly score's percentile within the
# training distribution (computed once in train_model.py, stored in
# training_stats.json). Anything below the model's own decision boundary
# (score <= 0) is already flagged "normal" by IsolationForest itself.
SEVERITY_PERCENTILE_BANDS = [
    ("watch", 0.95),     # top 5% most anomalous-looking, still "normal" by clf
    ("warning", 0.99),   # top 1%
    ("severe", 1.01),    # beyond max observed in training / clf flags anomaly
]

TOP_K_CONTRIBUTING_FEATURES = 4
