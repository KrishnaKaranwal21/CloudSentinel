# 🌦️ Hazard Wizard

**Real-time weather hazard monitoring for India, powered by a live weather feed and an AI anomaly-detection model trained across 50 cities.**

Hazard Wizard combines two independent layers of hazard detection: fast, transparent **rule-based thresholds** for known danger signs, and a **PyTorch autoencoder** that learns what "normal" weather looks like for any location and season in India — and flags conditions that don't fit, even when no single rule catches them.

---

## Table of Contents

- [Overview](#overview)
- [How it works](#how-it-works)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Frontend](#frontend-setup)
  - [ML Backend](#ml-backend-setup)
- [Training the Model](#training-the-model)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Roadmap](#roadmap)
- [Contributors](#contributors)

---

## Overview

Type or search any location in India and Hazard Wizard shows:

- **Live weather telemetry** — temperature, humidity, wind, pressure, precipitation, and cloud cover, sourced from [Open-Meteo](https://open-meteo.com).
- **Rule-based hazard signals** — clear, explainable alerts for known danger patterns (extreme heat, high wind gusts, heavy precipitation, etc.).
- **AI anomaly detection** — a trained autoencoder scores how unusual the current conditions are for that location *and* time of year, with SHAP-based explanations for *why* something was flagged.

The two systems are intentionally complementary rather than redundant: rules catch known, well-understood danger thresholds; the model catches statistically unusual combinations of conditions that don't necessarily trip any single rule — including for locations the model was never explicitly trained on.

## How it works

1. You pick or search a location. The frontend fetches live conditions directly from Open-Meteo.
2. Rule-based thresholds run **client-side**, instantly, against that reading.
3. In parallel, the same reading (plus the location's coordinates and the current month) is sent server-side to a FastAPI service wrapping the trained PyTorch autoencoder.
4. The autoencoder tries to reconstruct the input. A reading that reconstructs poorly — i.e. doesn't look like anything the model learned as "normal" — gets flagged as an anomaly, with SHAP identifying which features contributed most.
5. Because the model was trained on latitude, longitude, and a cyclical month encoding *alongside* the weather variables, it doesn't need a fixed threshold per city — it learns a continuous sense of what's typical for a given place and season, and generalizes to locations outside its original 50-city training set by nearest-neighbor similarity in the learned space.

## Architecture

```text
┌─────────────────────────┐         ┌───────────────────────────┐
│   Browser (React SPA)   │         │   Open-Meteo (public API) │
│                         │◄────────┤     live weather data     │
└────────────┬────────────┘         └───────────────────────────┘
             │
             │  (same-origin request)
             ▼
┌─────────────────────────┐
│  TanStack Start server  │   Server function (createServerFn)
│   (Vercel serverless)   │   proxies the reading to the ML API —
└────────────┬────────────┘   keeps the ML API URL/key off the client
             │
             │  server-to-server, API-key authenticated
             ▼
┌─────────────────────────┐
│   FastAPI ML service    │   Loads trained weights once at startup
│   (Render/Railway)      │   Autoencoder + SHAP inference
└─────────────────────────┘
```

This split matters: the Python ML stack (PyTorch, SHAP, scikit-learn) is too heavy for a serverless function, so it runs as its own always-on service, while the frontend stays fully on Vercel and never talks to it directly from the browser.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend framework | React + TanStack Start (Vite, file-based routing, server functions) |
| Data fetching (client) | Open-Meteo REST API |
| Data fetching (server) | React Query |
| ML runtime | PyTorch (autoencoder), scikit-learn (preprocessing), SHAP (explainability) |
| ML API | FastAPI + Uvicorn |
| Frontend hosting | Vercel |
| ML API hosting | Render |

## Project Structure

```text
Hazard-Wizard/
├── backend-ml/
│   ├── raw_data/
│   ├── weights/
│   ├── .python-version
│   ├── final_ml_engine.py
│   ├── main.py
│   ├── model_experiments.py
│   ├── requirements.txt
│   └── train.py
├── public/
│   ├── favicon.ico
│   └── robots.txt
├── src/
│   ├── lib/
│   │   ├── error-capture.ts
│   │   ├── error-page.ts
│   │   ├── hazard-ml.ts
│   │   └── weather.ts
│   ├── routes/
│   │   ├── __root.tsx
│   │   └── index.tsx
│   ├── routeTree.gen.ts
│   ├── router.tsx
│   ├── server.ts
│   ├── start.ts
│   └── styles.css
└── README.md
```

## Getting Started

### Frontend Setup

**Prerequisites:** Node.js 18+

```bash
npm install
npm run dev
```

The dashboard runs at `http://localhost:5173` (actual port shown in your terminal). The weather display and rule-based hazard signals work immediately — no backend required for those.

### ML Backend Setup

**Prerequisites:** Python 3.10.13 (newer versions may fail to install pinned dependencies like Pandas and PyTorch).

```bash
cd backend-ml
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

You'll need trained model weights before the API can start — see [Training the Model](#training-the-model) below.

```bash
uvicorn main:app --reload --port 8000
```

Confirm it's running:
```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

## Training the Model

The model is trained on a full year of pooled hourly historical weather data from **50 Indian cities**, chosen to span every major climate zone — coastal, arid desert, Himalayan alpine, tropical wet, continental plains, plateau, and island. Each training row includes:

- The 7 core weather variables (temperature, humidity, pressure, wind speed, wind gusts, precipitation, cloud cover)
- The location's latitude and longitude
- A cyclical (sine/cosine) encoding of the month, so the model understands seasonality without treating December and January as far apart

```bash
cd backend-ml
python train.py
```

This fetches historical data for each city (cached locally to `raw_data/` so re-runs don't re-hit the API), trains the autoencoder, and writes four files to `weights/`:

| File | Purpose |
|---|---|
| `autoencoder.pt` | Trained PyTorch model weights |
| `scaler.pkl` | Fitted feature scaler (must match what the model was trained on) |
| `threshold.txt` | Calibrated anomaly threshold |
| `background.npy` | Background sample used by SHAP for explanations |

The full training run takes several minutes, most of it in the historical data fetch. Once complete, commit `weights/` so your deployed API has the artifacts it needs at startup.

## Environment Variables

### Frontend (set in Vercel project settings, or `.env.local` for local dev)

| Variable | Description |
|---|---|
| `ML_API_URL` | Base URL of the deployed FastAPI ML service |
| `ML_API_KEY` | Shared secret sent with each request to the ML service |

These are intentionally **not** prefixed with `VITE_` — they're read inside a server function that never ships to the browser bundle, so the ML service's URL and key stay server-side only.

### ML Backend (set in Render/Railway service settings)

| Variable | Description |
|---|---|
| `ML_API_KEY` | Must match the frontend's `ML_API_KEY` exactly |
| `WEIGHTS_DIR` | Optional — defaults to `weights` |
| `PYTHON_VERSION` | Set to `3.10.13` — Several ML dependencies do not have prebuilt wheels for newer versions (like 3.11 or 3.14) and will crash the cloud build. |

## API Reference

### `POST /predict`

**Headers:** `X-API-Key: <your key>` (if `ML_API_KEY` is set)

**Body:**
```json
{
  "temperature_2m": 25.3,
  "relative_humidity_2m": 65,
  "pressure_msl": 1013.25,
  "wind_speed_10m": 12.5,
  "wind_gusts_10m": 18.0,
  "precipitation": 0.0,
  "cloud_cover": 40,
  "latitude": 19.0760,
  "longitude": 72.8777
}
```

**Response:**
```json
{
  "status": "Normal",
  "anomaly_score": 0.14,
  "trigger_factors": [],
  "reference_city": "Mumbai"
}
```

`reference_city` is the nearest of the 50 training cities to the given coordinates — shown for transparency, not used by the model itself (the model takes raw latitude/longitude directly and generalizes continuously across any location in India).

### `GET /health`

Returns `{"status": "ok"}` — used for uptime checks and confirming the service started successfully (i.e. weights loaded without error).

## Deployment

**Frontend → Vercel.** `vercel.json` already declares the TanStack Start framework preset, so Vercel auto-detects the build. Set `ML_API_URL` and `ML_API_KEY` in the project's environment variables before deploying.

**ML API → Render.** Create a Web Service pointed at the `backend-ml/` folder as the root directory, with:
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- `PYTHON_VERSION=3.10.13` and `ML_API_KEY` set in the environment

The ML API must be deployed and its URL added to Vercel *before* the "AI anomaly detection" panel will show real results — the rest of the dashboard works independently of it.

## Anomaly panel (SHAP explainability)

`src/components/anomaly/AnomalyPanel.tsx` is the single place that renders the
model's output, driven entirely by `getHazardAssessment` (`src/lib/hazard-ml.ts`).
It always resolves to one of four states — loading skeleton, endpoint
unavailable, no anomaly, or anomaly detected — and the anomaly-detected state
always shows the three things a reviewer needs:

- **What** — a headline built from the top SHAP-ranked `trigger_factors`.
- **How it might occur** — a plain-language explanation of the autoencoder's
  reconstruction-error logic, referencing `reference_city` for context.
- **How bad it is** — a `Moderate` / `High` / `Critical` badge plus an animated
  meter, derived from `anomaly_score` by `deriveSeverity()` in `hazard-ml.ts`
  (thresholds at 0.7 / 0.85 — tune these in one place if the model's own score
  distribution differs).

The request payload now also sends `latitude`/`longitude`, which the trained
model requires but the previous integration omitted.

---
