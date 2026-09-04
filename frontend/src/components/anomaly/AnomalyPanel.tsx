import { AlertTriangle, Gauge, Radar, ShieldCheck, WifiOff } from "lucide-react";

import type { HazardAssessment, Severity } from "@/lib/hazard-ml";

type AnomalyPanelProps = {
  assessment: HazardAssessment | undefined;
  isLoading: boolean;
  isError: boolean;
  isFetching: boolean;
};

const SEVERITY_COPY: Record<Severity, { label: string; blurb: string }> = {
  Moderate: {
    label: "Moderate",
    blurb: "Worth a look — conditions are drifting outside the model's learned normal range.",
  },
  High: {
    label: "High",
    blurb: "Reconstruction error is high — this reading looks meaningfully out of pattern.",
  },
  Critical: {
    label: "Critical",
    blurb: "Severe deviation from the learned normal — treat this as a priority to review.",
  },
};

/**
 * Anomaly explainability panel for the FastAPI/SHAP anomaly-detection endpoint.
 * Always shows three things once an anomaly is flagged: WHAT was detected,
 * HOW it likely arose (in model terms), and HOW BAD it is (severity + score).
 */
export function AnomalyPanel({ assessment, isLoading, isError, isFetching }: AnomalyPanelProps) {
  if (isLoading) return <AnomalySkeleton />;

  if (isError) {
    return (
      <article className="panel anomaly-panel anomaly-state-unavailable anim-fade-in-up">
        <div className="panel-title">
          <div>
            <span>ANOMALY DETECTION</span>
            <h3>Model endpoint unavailable</h3>
          </div>
          <WifiOff size={20} />
        </div>
        <p className="panel-body-copy">
          The reading could not be scored right now. Check that <code>ML_API_URL</code> (and{" "}
          <code>ML_API_KEY</code> if required) are configured, and that the ML service is running.
        </p>
      </article>
    );
  }

  if (!assessment) {
    return (
      <article className="panel anomaly-panel anomaly-state-unavailable">
        <div className="panel-title">
          <div>
            <span>ANOMALY DETECTION</span>
            <h3>Waiting for a reading</h3>
          </div>
          <Radar size={20} />
        </div>
        <p className="panel-body-copy">Scoring will start as soon as live weather data arrives.</p>
      </article>
    );
  }

  if (assessment.status === "Normal") {
    return (
      <article
        key={assessment.observedAt}
        className="panel anomaly-panel anomaly-state-normal anim-fade-in-up"
      >
        <div className="panel-title">
          <div>
            <span>ANOMALY DETECTION{isFetching ? " · UPDATING" : ""}</span>
            <h3>No anomaly detected</h3>
          </div>
          <ShieldCheck size={20} />
        </div>
        <p className="panel-body-copy">
          This reading reconstructs cleanly against the model's learned normal pattern
          {assessment.referenceCity ? ` for the ${assessment.referenceCity} area` : ""} — nothing
          for the anomaly model to flag right now.
        </p>
        <div className="anomaly-score-footnote">
          <span>ANOMALY SCORE</span>
          <strong>{assessment.anomalyScore.toFixed(2)}</strong>
        </div>
      </article>
    );
  }

  const severity = assessment.severity ?? "Moderate";
  const copy = SEVERITY_COPY[severity];

  return (
    <article
      key={assessment.observedAt}
      className={`panel anomaly-panel anomaly-state-hazard severity-${severity.toLowerCase()} anim-fade-in-up`}
    >
      <div className="panel-title">
        <div>
          <span>ANOMALY DETECTED{isFetching ? " · UPDATING" : ""}</span>
          <h3>{describeWhat(assessment)}</h3>
        </div>
        <AlertTriangle size={20} />
      </div>

      <div className="anomaly-block">
        <span className="anomaly-block-label">HOW IT MIGHT OCCUR</span>
        <p>{describeHow(assessment)}</p>
      </div>

      <div className="anomaly-block">
        <span className="anomaly-block-label">HOW BAD IT IS</span>
        <div className="severity-row">
          <span className={`severity-badge severity-${severity.toLowerCase()}`}>{copy.label}</span>
          <span className="severity-score">{Math.round(assessment.anomalyScore * 100)}% anomaly score</span>
        </div>
        <div className="severity-meter" aria-hidden="true">
          <div
            className={`severity-meter-fill severity-${severity.toLowerCase()}`}
            style={{ width: `${Math.min(100, Math.max(4, assessment.anomalyScore * 100))}%` }}
          />
        </div>
        <p className="severity-blurb">{copy.blurb}</p>
      </div>

      {assessment.triggerFactors.length > 0 && (
        <div className="anomaly-block">
          <span className="anomaly-block-label">TOP CONTRIBUTING FACTORS</span>
          <div className="trigger-chip-row">
            {assessment.triggerFactors.map((factor, index) => (
              <span className="trigger-chip" key={factor.key}>
                <em>{index + 1}</em>
                {factor.label}
              </span>
            ))}
          </div>
        </div>
      )}

      {assessment.referenceCity && (
        <small className="demo-disclaimer">
          Nearest reference city used by the model: {assessment.referenceCity}
        </small>
      )}
    </article>
  );
}

function describeWhat(assessment: HazardAssessment): string {
  const [first, second] = assessment.triggerFactors;
  if (!first) return "Unusual combination of conditions";
  if (!second) return `Unusual ${first.label.toLowerCase()} reading`;
  return `Unusual ${first.label.toLowerCase()} and ${second.label.toLowerCase()}`;
}

function describeHow(assessment: HazardAssessment): string {
  const place = assessment.referenceCity ? `conditions typical of ${assessment.referenceCity} and nearby areas` : "typical conditions for this location and season";
  const factors = assessment.triggerFactors.map((factor) => factor.label.toLowerCase());
  const factorClause =
    factors.length > 0
      ? ` It's most influenced by ${joinWithAnd(factors)}, which the model weighs most heavily for this reading.`
      : "";
  return `The autoencoder could not reconstruct this reading from the patterns it learned for ${place} — that mismatch, rather than any single fixed threshold, is what triggers the flag.${factorClause}`;
}

function joinWithAnd(items: string[]): string {
  if (items.length <= 1) return items[0] ?? "";
  if (items.length === 2) return `${items[0]} and ${items[1]}`;
  return `${items.slice(0, -1).join(", ")}, and ${items[items.length - 1]}`;
}

function AnomalySkeleton() {
  return (
    <article className="panel anomaly-panel anomaly-skeleton" aria-busy="true" aria-label="Loading anomaly assessment">
      <div className="panel-title">
        <div>
          <span>ANOMALY DETECTION</span>
          <div className="skeleton-line skeleton-line-title" />
        </div>
        <Gauge size={20} className="skeleton-icon" />
      </div>
      <div className="skeleton-line skeleton-line-body" />
      <div className="skeleton-line skeleton-line-body short" />
      <div className="skeleton-meter" />
    </article>
  );
}
