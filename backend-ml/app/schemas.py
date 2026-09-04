from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field


class WeatherInput(BaseModel):
    """Manual observation input — matches the 13 features the model was
    trained on (see KeyHourly.txt). Fill these in via /docs to test the
    anomaly detector with a custom scenario."""

    rain: float = Field(0.0, description="Precipitation amount (mm)", examples=[0.0])
    temp: float = Field(..., description="Air temperature (\u00b0C)", examples=[11.3])
    wetb: float = Field(..., description="Wet bulb temperature (\u00b0C)", examples=[11.1])
    dewpt: float = Field(..., description="Dew point temperature (\u00b0C)", examples=[10.9])
    vappr: float = Field(..., description="Vapour pressure (hPa)", examples=[13.0])
    rhum: float = Field(..., description="Relative humidity (%)", examples=[97.0])
    msl: float = Field(..., description="Mean sea level pressure (hPa)", examples=[1009.2])
    wdsp: float = Field(..., description="Mean wind speed (kt)", examples=[6.0])
    wddir: float = Field(..., description="Wind direction (\u00b0)", examples=[150.0])
    sun: float = Field(0.0, description="Sunshine duration (hours)", examples=[0.0])
    vis: float = Field(..., description="Visibility (m)", examples=[7000.0])
    clht: float = Field(999.0, description="Cloud ceiling height (100s ft); 999 = none", examples=[999.0])
    clamt: float = Field(..., description="Cloud amount (okta, 0-8)", examples=[8.0])
    observed_at: dt.datetime | None = Field(
        None, description="Timestamp of the observation (defaults to now). "
                           "Used only to determine the month for seasonal features."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "rain": 0.5, "temp": 11.3, "wetb": 11.1, "dewpt": 10.9,
                "vappr": 13.0, "rhum": 97.0, "msl": 1009.2, "wdsp": 6.0,
                "wddir": 150.0, "sun": 0.0, "vis": 7000.0, "clht": 35.0,
                "clamt": 8.0,
            }]
        }
    }


class FeatureContribution(BaseModel):
    feature: str
    raw_value: float
    shap_value: float = Field(..., description="Positive = pushed toward anomaly, negative = pushed toward normal")


class AnomalyResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float = Field(..., description="Higher = more anomalous. 0 is roughly the model's own decision boundary.")
    severity_level: str = Field(..., description="NORMAL | WATCH | WARNING | SEVERE")
    severity_text: str
    what: str = Field(..., description="Plain-language description of the anomaly")
    root_cause: str = Field(..., description="Plain-language likely cause, derived from SHAP feature attribution")
    top_contributions: list[FeatureContribution]
    observed_at: dt.datetime
    raw_values: dict[str, float]
    estimated_fields: list[str] = Field(
        default_factory=list,
        description="Feature names that were derived/estimated rather than measured directly (only set for /predict/live)",
    )
    source: str = Field("manual-input", description="'open-meteo.com' for live data, 'manual-input' for /predict/manual")
