export type Location = {
  name: string;
  country?: string;
  admin1?: string;
  latitude: number;
  longitude: number;
};

export type HourlyReading = {
  time: number;
  precipitation: number;
  precipitationProbability: number;
  temperature: number;
  windSpeed: number;
};

export type WeatherSnapshot = {
  location: Location;
  observedAt: number;
  timezone: string;
  timezoneAbbreviation: string;
  temperature: number;
  apparentTemperature: number;
  humidity: number;
  precipitation: number;
  rain: number;
  windSpeed: number;
  windGusts: number;
  windDirection: number;
  pressure: number;
  pressureMsl: number;
  cloudCover: number;
  weatherCode: number;
  isDay: boolean;
  hourly: HourlyReading[];
};

type ApiRecord = Record<string, unknown>;

const forecastEndpoint = "https://api.open-meteo.com/v1/forecast";
const geocodingEndpoint = "https://geocoding-api.open-meteo.com/v1/search";

function numberValue(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`Weather service returned an invalid ${name}.`);
  }
  return value;
}

function recordValue(value: unknown, name: string): ApiRecord {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Weather service returned no ${name}.`);
  }
  return value as ApiRecord;
}

function numberArray(value: unknown): number[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is number => typeof item === "number" && Number.isFinite(item));
}

function providerMessage(response: Response): string {
  if (response.status === 429) return "The weather service is busy. Please try again in a moment.";
  return `The weather service could not load this location (${response.status}).`;
}

export async function fetchWeather(
  location: Location,
  signal?: AbortSignal,
): Promise<WeatherSnapshot> {
  const query = new URLSearchParams({
    latitude: String(location.latitude),
    longitude: String(location.longitude),
    current:
      "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,weather_code,surface_pressure,pressure_msl,cloud_cover,wind_speed_10m,wind_direction_10m,wind_gusts_10m,is_day",
    hourly: "temperature_2m,precipitation,precipitation_probability,wind_speed_10m",
    forecast_hours: "24",
    timezone: "auto",
    timeformat: "unixtime",
    wind_speed_unit: "kmh",
  });
  const response = await fetch(`${forecastEndpoint}?${query}`, signal ? { signal } : undefined);
  if (!response.ok) throw new Error(providerMessage(response));

  const payload = recordValue(await response.json(), "forecast payload");
  const current = recordValue(payload["current"], "current conditions");
  const hourly = recordValue(payload["hourly"], "hourly forecast");
  const times = numberArray(hourly["time"]);
  const precipitation = numberArray(hourly["precipitation"]);
  const probabilities = numberArray(hourly["precipitation_probability"]);
  const temperatures = numberArray(hourly["temperature_2m"]);
  const winds = numberArray(hourly["wind_speed_10m"]);

  const readings = times.slice(0, 24).map((time, index) => ({
    time,
    precipitation: precipitation[index] ?? 0,
    precipitationProbability: probabilities[index] ?? 0,
    temperature: temperatures[index] ?? 0,
    windSpeed: winds[index] ?? 0,
  }));

  return {
    location,
    observedAt: numberValue(current["time"], "observation time") * 1000,
    timezone: typeof payload["timezone"] === "string" ? payload["timezone"] : "UTC",
    timezoneAbbreviation:
      typeof payload["timezone_abbreviation"] === "string"
        ? payload["timezone_abbreviation"]
        : "UTC",
    temperature: numberValue(current["temperature_2m"], "temperature"),
    apparentTemperature: numberValue(current["apparent_temperature"], "apparent temperature"),
    humidity: numberValue(current["relative_humidity_2m"], "humidity"),
    precipitation: numberValue(current["precipitation"], "precipitation"),
    rain: numberValue(current["rain"], "rain"),
    windSpeed: numberValue(current["wind_speed_10m"], "wind speed"),
    windGusts: numberValue(current["wind_gusts_10m"], "wind gusts"),
    windDirection: numberValue(current["wind_direction_10m"], "wind direction"),
    pressure: numberValue(current["surface_pressure"], "surface pressure"),
    pressureMsl: numberValue(current["pressure_msl"], "sea-level pressure"),
    cloudCover: numberValue(current["cloud_cover"], "cloud cover"),
    weatherCode: numberValue(current["weather_code"], "weather code"),
    isDay: numberValue(current["is_day"], "day indicator") === 1,
    hourly: readings,
  };
}

export async function searchLocations(
  queryText: string,
  signal?: AbortSignal,
): Promise<Location[]> {
  const query = new URLSearchParams({
    name: queryText,
    count: "5",
    language: "en",
    format: "json",
  });
  const response = await fetch(`${geocodingEndpoint}?${query}`, signal ? { signal } : undefined);
  if (!response.ok) throw new Error(providerMessage(response));
  const payload = recordValue(await response.json(), "location search");
  if (!Array.isArray(payload["results"])) return [];

  return payload["results"].flatMap((result) => {
    if (!result || typeof result !== "object" || Array.isArray(result)) return [];
    const item = result as ApiRecord;
    if (
      typeof item["name"] !== "string" ||
      typeof item["latitude"] !== "number" ||
      typeof item["longitude"] !== "number"
    )
      return [];
    const location: Location = {
      name: item["name"],
      latitude: item["latitude"],
      longitude: item["longitude"],
    };
    if (typeof item["country"] === "string") location.country = item["country"];
    if (typeof item["admin1"] === "string") location.admin1 = item["admin1"];
    return [location];
  });
}

export function describeWeather(code: number): string {
  const labels: Record<number, string> = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Icy fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Heavy drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    80: "Rain showers",
    81: "Rain showers",
    82: "Violent showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with hail",
  };
  return labels[code] ?? "Weather conditions";
}
