import { createFileRoute } from "@tanstack/react-router";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Check,
  ChevronDown,
  CloudRain,
  Crosshair,
  Gauge,
  LocateFixed,
  MapPin,
  RefreshCw,
  Search,
  ShieldCheck,
  Siren,
  ThermometerSun,
  Wind,
  type LucideIcon,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import {
  describeWeather,
  fetchWeather,
  searchLocations,
  type Location,
  type WeatherSnapshot,
} from "@/lib/weather";
import { getHazardAssessment } from "@/lib/hazard-ml";
import { assessmentFromLive, demoScenarios, type CloudSentinelAssessment } from "@/lib/cloudsentinel";
import { AnomalyPanel } from "@/components/anomaly/AnomalyPanel";

// Live weather refresh cadence — the anomaly query key below is derived from
// the snapshot, so it refetches on the same cadence automatically.
const REFRESH_INTERVAL_MS = 60_000;

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "CloudSentinel — Predictive Weather Hazard Intelligence" },
      {
        name: "description",
        content:
          "Forecast-driven early warning, AWS anomaly detection, sensor correction, explainability and impact awareness.",
      },
      { property: "og:title", content: "CloudSentinel — Predictive Weather Hazard Intelligence" },
      {
        property: "og:description",
        content: "Predict the hazard. Detect the anomaly. Correct the sensor. Explain the decision.",
      },
    ],
  }),
  component: Dashboard,
});

const defaultLocation: Location = {
  name: "Bengaluru",
  country: "India",
  latitude: 12.9716,
  longitude: 77.5946,
};

type Hazard = {
  title: string;
  detail: string;
  severity: "critical" | "watch";
  source: string;
};

type PillTone = "teal" | "amber" | "red" | "blue" | "violet";

function Dashboard() {
  const [location, setLocation] = useState<Location>(defaultLocation);
  const [placeQuery, setPlaceQuery] = useState(defaultLocation.name);
  const [latitude, setLatitude] = useState(String(defaultLocation.latitude));
  const [longitude, setLongitude] = useState(String(defaultLocation.longitude));
  const [snapshot, setSnapshot] = useState<WeatherSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(true);
  const [isSearching, setIsSearching] = useState(false);
  const [clock, setClock] = useState(() => new Date());
  const [demoScenario, setDemoScenario] = useState("live");

  const refresh = useCallback(
    async (target: Location = location) => {
      setIsRefreshing(true);
      setError(null);
      try {
        setSnapshot(await fetchWeather(target));
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "Live weather data could not be loaded.");
      } finally {
        setIsRefreshing(false);
      }
    },
    [location],
  );

  useEffect(() => {
    void refresh(location);
    const id = window.setInterval(() => void refresh(location), REFRESH_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [location, refresh]);

  useEffect(() => {
    const id = window.setInterval(() => setClock(new Date()), 1000);
    return () => window.clearInterval(id);
  }, []);

  const updateLocation = (next: Location) => {
    setSnapshot(null);
    setLocation(next);
    setPlaceQuery(next.name);
    setLatitude(next.latitude.toFixed(4));
    setLongitude(next.longitude.toFixed(4));
    setDemoScenario("live");
  };

  const searchPlace = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const term = placeQuery.trim();
    if (!term) return setError("Enter a place name to search.");
    setIsSearching(true);
    setError(null);
    try {
      const match = (await searchLocations(term))[0];
      if (!match) return setError(`No matching place was found for “${term}”.`);
      updateLocation(match);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Location search failed.");
    } finally {
      setIsSearching(false);
    }
  };

  const submitCoordinates = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const lat = Number(latitude);
    const lon = Number(longitude);
    if (!Number.isFinite(lat) || !Number.isFinite(lon) || Math.abs(lat) > 90 || Math.abs(lon) > 180) {
      return setError("Use a latitude from −90 to 90 and a longitude from −180 to 180.");
    }
    updateLocation({ name: `${lat.toFixed(4)}°, ${lon.toFixed(4)}°`, latitude: lat, longitude: lon });
  };

  const useDeviceLocation = () => {
    if (!navigator.geolocation) return setError("This browser does not provide device location.");
    setError(null);
    navigator.geolocation.getCurrentPosition(
      (position) => updateLocation({ name: "Your current location", latitude: position.coords.latitude, longitude: position.coords.longitude }),
      () => setError("Location access was not granted. Use coordinates instead."),
      { enableHighAccuracy: false, timeout: 10_000 },
    );
  };

  const hazards = useMemo(() => (snapshot ? getHazards(snapshot) : []), [snapshot]);
  const mlAssessment = useQuery({
    queryKey: ["hazard-ml", snapshot?.location.latitude, snapshot?.location.longitude, snapshot?.observedAt],
    queryFn: () => getHazardAssessment({ data: snapshot! }),
    enabled: snapshot !== null,
    retry: 1,
    staleTime: REFRESH_INTERVAL_MS - 5_000,
    // Belt-and-braces: the query key already changes every refresh() cycle,
    // but this keeps scoring alive even if the weather poll stalls.
    refetchInterval: REFRESH_INTERVAL_MS,
  });

  const cloudSentinel = useMemo<CloudSentinelAssessment | null>(() => {
    if (!snapshot) return null;
    return demoScenario === "live" ? assessmentFromLive(snapshot) : demoScenarios[demoScenario] ?? assessmentFromLive(snapshot);
  }, [snapshot, demoScenario]);

  const forecastRain = snapshot?.hourly.reduce((sum, hour) => sum + hour.precipitation, 0) ?? 0;
  const nextWetHour = snapshot?.hourly.find((hour) => hour.precipitation > 1 || hour.precipitationProbability >= 50);
  const topHazard = cloudSentinel?.forecast;
  const systemState = topHazard?.severity === "Critical" ? "critical" : topHazard?.severity === "High" ? "high" : topHazard?.severity === "Moderate" ? "moderate" : "normal";

  return (
    <main className="cloud-app">
      <div className="noise-layer" />
      <div className="glow glow-a" />
      <div className="glow glow-b" />

      <section className="app-shell">
        <header className="nav-bar">
          <div className="brand">
            <div className="brand-icon"><ShieldCheck size={22} /></div>
            <div><strong>CloudSentinel</strong><span>WEATHER HAZARD INTELLIGENCE</span></div>
          </div>
          <div className="nav-meta">
            <span className="live-chip"><i /> {isRefreshing ? "SYNCING" : "LIVE"}</span>
            <span>{formatClock(clock, snapshot?.timezone)}</span>
            <span className="nav-location"><MapPin size={13} /> {location.name}</span>
          </div>
        </header>

        <section className="hero">
          <div className="hero-copy-block">
            <div className="section-kicker"><span className="kicker-line" /> PREDICTIVE WEATHER HAZARD INTELLIGENCE</div>
            <h1>Predict the hazard.<br /><em>Before it becomes a problem.</em></h1>
            <p>CloudSentinel combines forecast risk, automatic weather-station validation, explainable anomaly detection and exposure awareness in one operational view.</p>
            <div className="flow-strip">
              <FlowStep index="01" label="Predict" detail="forecast risk" tone="teal" />
              <ArrowRight size={15} />
              <FlowStep index="02" label="Detect" detail="AWS anomalies" tone="blue" />
              <ArrowRight size={15} />
              <FlowStep index="03" label="Correct" detail="faulty readings" tone="violet" />
              <ArrowRight size={15} />
              <FlowStep index="04" label="Explain" detail="XAI factors" tone="amber" />
              <ArrowRight size={15} />
              <FlowStep index="05" label="Assess" detail="asset exposure" tone="red" />
            </div>
          </div>
          <div className={`command-card state-${systemState}`}>
            <div className="command-head"><span>COMMAND STATUS</span><span>{locationLabel(snapshot?.location ?? location)}</span></div>
            <div className="command-icon"><Siren size={24} /></div>
            <span className="command-status">{topHazard?.severity ?? "—"}</span>
            <h2>{topHazard?.hazard ?? "Loading forecast intelligence"}</h2>
            <p>{topHazard?.explanation ?? "Connecting to the live forecast feed."}</p>
            <div className="command-stats">
              <div><span>WINDOW</span><strong>{topHazard?.period ?? "—"}</strong></div>
              <div><span>ONSET</span><strong>{topHazard?.onset ?? "—"}</strong></div>
              <div><span>SCREENING</span><strong>{topHazard?.confidence ?? "—"}%</strong></div>
            </div>
          </div>
        </section>

        <section className="control-bar">
          <form onSubmit={searchPlace} className="search-control"><Search size={16} /><input value={placeQuery} onChange={(e) => setPlaceQuery(e.target.value)} placeholder="Search city / region" /><button type="submit" disabled={isSearching}>{isSearching ? "Finding" : "Monitor"}</button></form>
          <form onSubmit={submitCoordinates} className="coordinate-control"><input aria-label="Latitude" value={latitude} onChange={(e) => setLatitude(e.target.value)} /><input aria-label="Longitude" value={longitude} onChange={(e) => setLongitude(e.target.value)} /><button type="button" onClick={useDeviceLocation} title="Use current location"><LocateFixed size={16} /></button><button type="submit">Apply</button></form>
          <div className="source-note"><span>LIVE SOURCE</span> Open-Meteo · refresh 60 s</div>
        </section>

        {error && <div className="error-strip"><AlertTriangle size={16} /> <span>{error}</span><button onClick={() => void refresh()}>Retry</button></div>}

        {snapshot && cloudSentinel ? (
          <>
            <section className="section-block early-section">
              <div className="section-head"><div><div className="section-kicker">01 / EARLY WARNING</div><h2>What may happen next?</h2><p>Forecast conditions are screened separately from live AWS observations.</p></div><ScenarioPicker value={demoScenario} onChange={setDemoScenario} /></div>
              <div className="warning-grid">
                <article className="warning-card primary-warning">
                  <div className="warning-label"><span className={`status-dot severity-${topHazard?.severity?.toLowerCase()}`} /> EARLY WARNING · {topHazard?.source === "Demo simulation" ? "DEMO" : "LIVE FORECAST"}</div>
                  <h3>{topHazard?.hazard}</h3>
                  <p>{topHazard?.explanation}</p>
                  <div className="warning-metrics">{topHazard?.measurements.map((m) => <span key={m}>{m}</span>)}</div>
                </article>
                <article className="warning-card timeline-card"><div className="mini-label">NEXT SIGNAL</div><strong>{nextWetHour ? formatHour(nextWetHour.time, snapshot.timezone) : topHazard?.onset ?? "—"}</strong><p>{nextWetHour ? `${nextWetHour.precipitationProbability.toFixed(0)}% precipitation probability` : "No elevated precipitation window identified."}</p><div className="timeline"><i /><i /><i /><i className="active" /><i /></div></article>
                <article className="warning-card rain-total"><div className="mini-label">24H PRECIPITATION</div><strong>{forecastRain.toFixed(1)}<small> mm</small></strong><p>Forecast accumulation across the monitored window.</p></article>
              </div>
            </section>

            <section className="section-block">
              <div className="section-head"><div><div className="section-kicker">02 / AWS SENSOR INTELLIGENCE</div><h2>Is the station reporting reality?</h2><p>Live observations are validated independently from the forecast layer.</p></div><span className="live-badge"><i /> {cloudSentinel.isDemo ? "SIMULATION" : "LIVE OBSERVATION"}</span></div>
              <div className="sensor-topline"><Metric label="Temperature" value={`${cloudSentinel.observation.temperature.toFixed(1)} °C`} note="2 m air temperature" icon={ThermometerSun} tone="teal" /><Metric label="Humidity" value={`${cloudSentinel.observation.humidity.toFixed(0)} %`} note="Relative humidity" icon={Activity} tone="blue" /><Metric label="Pressure" value={`${cloudSentinel.observation.pressureMsl.toFixed(0)} hPa`} note="Mean sea level" icon={Gauge} tone="violet" /><Metric label="Wind" value={`${cloudSentinel.observation.windSpeed.toFixed(0)} km/h`} note={`Gusts ${cloudSentinel.observation.windGusts.toFixed(0)} km/h`} icon={Wind} tone="amber" /><Metric label="Precipitation" value={`${cloudSentinel.observation.precipitation.toFixed(1)} mm`} note="Current interval" icon={CloudRain} tone="red" /></div>
              <div className="sensor-grid">
                <article className="panel sensor-panel"><div className="panel-title"><div><span>STATION HEALTH</span><h3>Signal validation</h3></div><ShieldCheck size={18} /></div><div className="health-list">{cloudSentinel.sensors.map((sensor) => <div key={sensor.name}><span className={`health-dot ${sensor.quality.toLowerCase()}`} /><div><strong>{sensor.name}</strong><small>{sensor.value}</small></div><em>{sensor.status}</em></div>)}</div><div className="pipeline"><span>INGEST</span><ArrowRight size={13}/><span>VALIDATE</span><ArrowRight size={13}/><span>DETECT</span><ArrowRight size={13}/><span>ASSESS</span></div></article>
                <AnomalyPanel
                  assessment={mlAssessment.data}
                  isLoading={mlAssessment.isLoading}
                  isError={mlAssessment.isError}
                  isFetching={mlAssessment.isFetching}
                />
              </div>
            </section>

            <section className="section-block impact-section">
              <div className="section-head"><div><div className="section-kicker">03 / POTENTIAL IMPACT</div><h2>What could be affected nearby?</h2><p>Exposure screening helps teams decide what to review next — it does not guarantee damage.</p></div><span className="xai-chip">ILLUSTRATIVE ASSET LAYER</span></div>
              {cloudSentinel.assets.length ? <div className="impact-grid"><article className="impact-map panel"><div className="map-grid" /><div className="map-center"><Crosshair size={18}/><span>MONITORED LOCATION</span></div>{cloudSentinel.assets.map((asset, index) => <div className={`asset-pin pin-${index}`} key={asset.name}><i /><span>{asset.name}</span></div>)}<div className="map-caption">Potential exposure zones · relative positions for demonstration</div></article><article className="impact-list">{cloudSentinel.assets.map((asset) => <div className="asset-row" key={asset.name}><div className="asset-icon"><MapPin size={16}/></div><div><strong>{asset.name}</strong><span>{asset.type} · {asset.distanceKm} km</span><p>{asset.exposure}</p></div><b className={`risk risk-${asset.risk.toLowerCase()}`}>{asset.risk}</b></div>)}</article></div> : <div className="no-impact panel"><ShieldCheck size={22}/><div><strong>No elevated exposure layer shown.</strong><span>The selected assessment does not currently trigger the illustrative asset layer.</span></div></div>}
            </section>

            <section className="section-block final-strip">
              <div><div className="section-kicker">04 / DECISION VIEW</div><h2>One screen. One operational story.</h2><p>Forecast risk, station integrity, explainability and exposure stay visibly separate so it's clear what is live, what is simulated and what still needs backend integration.</p></div><div className="decision-badges"><span><Check size={14}/> Forecast layer</span><span><Check size={14}/> AWS validation</span><span><Check size={14}/> XAI pathway</span><span><Check size={14}/> Impact screening</span></div>
            </section>
          </>
        ) : (
          <div className="loading-state"><RefreshCw className="spin" size={20}/><div><strong>Connecting to live weather intelligence</strong><span>Loading forecast and current observation data for {locationLabel(location)}.</span></div></div>
        )}

        <footer className="footer"><span>CloudSentinel · SIH 2026 · Disaster Management</span><span>Weather-awareness and screening tool — not an official emergency alert system.</span></footer>
      </section>
    </main>
  );
}

function ScenarioPicker({ value, onChange }: { value: string; onChange: (value: string) => void }) {
  return <label className="scenario-picker"><span>DEMO / SIMULATION</span><select value={value} onChange={(event) => onChange(event.target.value)}><option value="live">Live forecast assessment</option><option value="normal">Normal weather</option><option value="rain">Heavy rainfall warning</option><option value="wind">High wind warning</option><option value="sensor">Temperature sensor spike</option><option value="multiple">Multiple abnormal conditions</option></select><ChevronDown size={14}/></label>;
}

function FlowStep({ index, label, detail, tone }: { index: string; label: string; detail: string; tone: PillTone }) {
  return <div className={`flow-step tone-${tone}`}><small>{index}</small><div><strong>{label}</strong><span>{detail}</span></div></div>;
}

function Metric({ label, value, note, icon: Icon, tone }: { label: string; value: string; note: string; icon: LucideIcon; tone: PillTone }) {
  return <article className={`metric-card tone-${tone}`}><div className="metric-icon"><Icon size={16}/></div><span>{label}</span><strong>{value}</strong><small>{note}</small></article>;
}

function getHazards(snapshot: WeatherSnapshot): Hazard[] {
  const alerts: Hazard[] = [];
  if (snapshot.precipitation >= 7) alerts.push({ title: "Heavy precipitation", detail: `${snapshot.precipitation.toFixed(1)} mm recorded in the current interval. Review local drainage and flood guidance.`, severity: "critical", source: "RAIN" });
  else if (snapshot.precipitation >= 2) alerts.push({ title: "Steady precipitation", detail: `${snapshot.precipitation.toFixed(1)} mm recorded in the current interval. Monitor surface conditions.`, severity: "watch", source: "RAIN" });
  if (snapshot.windGusts >= 70) alerts.push({ title: "Strong wind gusts", detail: `Peak gusts are ${snapshot.windGusts.toFixed(0)} km/h. Review exposed equipment and wind guidance.`, severity: "critical", source: "WIND" });
  else if (snapshot.windGusts >= 50) alerts.push({ title: "Gusty conditions", detail: `Peak gusts are ${snapshot.windGusts.toFixed(0)} km/h. Outdoor operations may need extra care.`, severity: "watch", source: "WIND" });
  if (snapshot.apparentTemperature >= 42) alerts.push({ title: "Extreme heat stress", detail: `Apparent temperature is ${snapshot.apparentTemperature.toFixed(1)}°C. Apply heat-safety procedures.`, severity: "critical", source: "HEAT" });
  else if (snapshot.apparentTemperature >= 36) alerts.push({ title: "Elevated heat stress", detail: `Apparent temperature is ${snapshot.apparentTemperature.toFixed(1)}°C. Plan hydration and rest.`, severity: "watch", source: "HEAT" });
  if (snapshot.pressure < 985) alerts.push({ title: "Low-pressure pattern", detail: `Surface pressure is ${snapshot.pressure.toFixed(0)} hPa. Watch for rapidly evolving conditions.`, severity: "watch", source: "PRESSURE" });
  return alerts;
}

function locationLabel(location: Location): string { return [location.name, location.admin1, location.country].filter(Boolean).join(", "); }
function formatClock(date: Date, timezone = "UTC"): string { return new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false, timeZone: timezone }).format(date); }
function formatHour(timestamp: number, timezone: string): string { return new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit", hour12: false, timeZone: timezone }).format(new Date(timestamp)); }
