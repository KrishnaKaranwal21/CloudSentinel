import { createServerFn } from "@tanstack/react-start";

import type { WeatherSnapshot } from "./weather";

/**
 * Anomaly detection contract for the FastAPI ML service (`backend-ml/main.py`).
 * See README.md → "API Reference → POST /predict" for the authoritative schema.
 */
export type Severity = "Moderate" | "High" | "Critical";

export type HazardAssessment = {
  status: "Normal" | "Hazard";
  /** Autoencoder reconstruction-error score, 0–1. Higher = more unusual. */
  anomalyScore: number;
  /** SHAP-ranked features that most influenced the score, most important first. */
  triggerFactors: TriggerFactor[];
  /** Nearest of the model's 50 training cities, for transparency only. */
  referenceCity: string | null;
  /** Only meaningful when status === "Hazard". */
  severity: Severity | null;
  observedAt: number;
};

export type TriggerFactor = {
  /** Raw key as returned by the model, e.g. "wind_gusts_10m". */
  key: string;
  /** Human-readable label, e.g. "Wind gusts". */
  label: string;
};

type MlApiResponse = {
  status?: unknown;
  anomaly_score?: unknown;
  trigger_factors?: unknown;
  reference_city?: unknown;
};

// Tunable locally without touching call sites — adjust if the model's own
// "Hazard" threshold or score distribution changes.
const SEVERITY_THRESHOLDS: { min: number; severity: Severity }[] = [
  { min: 0.85, severity: "Critical" },
  { min: 0.7, severity: "High" },
  { min: 0, severity: "Moderate" },
];

export function deriveSeverity(anomalyScore: number): Severity {
  const match = SEVERITY_THRESHOLDS.find((tier) => anomalyScore >= tier.min);
  return match?.severity ?? "Moderate";
}

// Fills in friendly labels for the feature keys the model is known to emit;
// anything unrecognised still renders sensibly via humanizeKey().
const KNOWN_FEATURE_LABELS: Record<string, string> = {
  temperature_2m: "Temperature",
  relative_humidity_2m: "Humidity",
  pressure_msl: "Sea-level pressure",
  wind_speed_10m: "Wind speed",
  wind_gusts_10m: "Wind gusts",
  precipitation: "Precipitation",
  cloud_cover: "Cloud cover",
};

function humanizeKey(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
    .replace(/(\d)m\b/, "$1 m");
}

function parseTriggerFactors(value: unknown): TriggerFactor[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string" && item.length > 0).map((key) => ({
    key,
    label: KNOWN_FEATURE_LABELS[key] ?? humanizeKey(key),
  }));
}

export const getHazardAssessment = createServerFn({ method: "POST" })
  .validator((snapshot: WeatherSnapshot) => snapshot)
  .handler(async ({ data }): Promise<HazardAssessment> => {
    const apiUrl = process.env.ML_API_URL;
    if (!apiUrl) {
      throw new Error("ML_API_URL is not configured on the server.");
    }

    const response = await fetch(`${apiUrl}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(process.env.ML_API_KEY ? { "X-API-Key": process.env.ML_API_KEY } : {}),
      },
      body: JSON.stringify({
        temperature_2m: data.temperature,
        relative_humidity_2m: data.humidity,
        pressure_msl: data.pressureMsl,
        wind_speed_10m: data.windSpeed,
        wind_gusts_10m: data.windGusts,
        precipitation: data.precipitation,
        cloud_cover: data.cloudCover,
        // Required by the trained model (it conditions on location) — omitting
        // these silently degrades every prediction, so they are not optional.
        latitude: data.location.latitude,
        longitude: data.location.longitude,
      }),
      signal: AbortSignal.timeout(8000),
    });

    if (!response.ok) {
      throw new Error(`ML service returned ${response.status}`);
    }

    const payload = (await response.json()) as MlApiResponse;
    if (payload.status !== "Normal" && payload.status !== "Hazard") {
      throw new Error("ML service returned an unrecognised status.");
    }
    if (typeof payload.anomaly_score !== "number" || !Number.isFinite(payload.anomaly_score)) {
      throw new Error("ML service returned an invalid anomaly score.");
    }

    const anomalyScore = payload.anomaly_score;
    return {
      status: payload.status,
      anomalyScore,
      triggerFactors: parseTriggerFactors(payload.trigger_factors),
      referenceCity: typeof payload.reference_city === "string" ? payload.reference_city : null,
      severity: payload.status === "Hazard" ? deriveSeverity(anomalyScore) : null,
      observedAt: data.observedAt,
    };
  });
