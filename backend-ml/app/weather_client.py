"""
Fetches current conditions from the Open-Meteo API (no key required) and
maps them onto the 13 features the model was trained on.

Open-Meteo doesn't publish every field the Dublin Airport station records
(wet bulb temperature, vapour pressure, and cloud ceiling height in
particular), so those are derived/estimated rather than fetched directly.
Every estimated field is listed in `estimated_fields` on the returned dict
so the API response can be transparent about it instead of pretending it's
a direct sensor reading.
"""
from __future__ import annotations

import datetime as dt

import httpx

from app.config import STATION_LAT, STATION_LON
from app.features import vapour_pressure_hpa, wet_bulb_c

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Everything we can get directly from Open-Meteo's hourly block.
HOURLY_VARS = (
    "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,"
    "pressure_msl,wind_speed_10m,wind_direction_10m,cloud_cover,"
    "visibility,sunshine_duration"
)


def _closest_hour_index(times: list[str], now: dt.datetime) -> int:
    parsed = [dt.datetime.fromisoformat(t) for t in times]
    return min(range(len(parsed)), key=lambda i: abs((parsed[i] - now).total_seconds()))


async def fetch_live_weather(lat: float = STATION_LAT, lon: float = STATION_LON) -> dict:
    """Returns {"observed_at": datetime, "values": {...13 features...},
    "estimated_fields": [...], "source": {...raw fields used...}}"""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_VARS,
        "wind_speed_unit": "kn",
        "timezone": "UTC",
        "forecast_days": 1,
        "past_days": 1,  # so "now" is never past the end of the array
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(OPEN_METEO_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    hourly = data["hourly"]
    now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    idx = _closest_hour_index(hourly["time"], now)
    observed_at = dt.datetime.fromisoformat(hourly["time"][idx])

    temp = hourly["temperature_2m"][idx]
    rhum = hourly["relative_humidity_2m"][idx]
    dewpt = hourly["dew_point_2m"][idx]
    rain = hourly["precipitation"][idx] or 0.0
    msl = hourly["pressure_msl"][idx]
    wdsp = hourly["wind_speed_10m"][idx]
    wddir = hourly["wind_direction_10m"][idx]
    cloud_cover_pct = hourly["cloud_cover"][idx]
    vis = hourly["visibility"][idx]
    sun_seconds = hourly["sunshine_duration"][idx] or 0.0

    estimated_fields = ["wetb", "vappr", "clht"]
    clamt = round((cloud_cover_pct or 0) / 100 * 8)
    # Open-Meteo's forecast API doesn't expose cloud ceiling height; -1
    # ("no ceiling") is a reasonable estimate when skies are near-clear,
    # otherwise we fall back to a neutral placeholder the model was trained
    # to see often (documented as an estimate, not a real ceiling reading).
    clht = -1.0 if (cloud_cover_pct or 0) < 10 else 1500.0

    values = {
        "rain": round(rain, 1),
        "temp": round(temp, 1),
        "wetb": round(wet_bulb_c(temp, rhum), 1),
        "dewpt": round(dewpt, 1),
        "vappr": round(vapour_pressure_hpa(temp, rhum), 1),
        "rhum": round(rhum, 0),
        "msl": round(msl, 1),
        "wdsp": round(wdsp, 1),
        "wddir": round(wddir, 0),
        "sun": round(sun_seconds / 3600, 2),
        "vis": round(vis, 0) if vis is not None else 10000.0,
        "clht": clht,
        "clamt": clamt,
    }

    return {
        "observed_at": observed_at,
        "values": values,
        "estimated_fields": estimated_fields,
        "source": "open-meteo.com",
    }
