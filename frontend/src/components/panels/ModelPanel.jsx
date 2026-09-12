import { DATA, HAZARDS, STEPS, timeAt } from '../../data/nowcastData';
import './ModelPanel.css';

/*
 * ModelPanel — model diagnostics.
 *
 * What produced this forecast and how much to trust it: model identity and
 * version, when inference last ran, which inputs fed it and how fresh they
 * are, and how well calibrated the model has been at this lead time.
 *
 * PLACEHOLDER DATA. nowcastData.js carries no model telemetry, so MODEL_STATUS
 * below is hand-written stand-in data. It is shaped to match the response a
 * live /model/status endpoint would return, so wiring it up is the single
 * assignment marked "swap here" inside the component — nothing else changes.
 *
 * Contract expected of a live response:
 *   model       { name, version, family, trainedThrough }
 *   run         { inferenceAt, cycle, gridKm, runtimeSec }
 *   sources[]   { name, provider, latest, ageMin, cadenceMin|null }
 *                 cadenceMin null marks a static layer; ageMin > cadenceMin
 *                 is reported as late.
 *   skill[hazardId] { brier[4], reliability[4], cases, window }
 *                 the four entries index by STEPS, same as everything else.
 */
export const MODEL_STATUS = {
  model: {
    name: 'Convective nowcast ensemble',
    version: '0.4.2-rc1',
    family: 'ConvLSTM + gradient-boosted post-processing',
    trainedThrough: '2026-06-30',
    // ISO timestamp backing the "Last trained" relative-time chip; the date
    // above stays the human-authored label so the two can never drift apart.
    trainedAt: '2026-06-30T00:00:00+05:30',
  },
  run: {
    inferenceAt: '07:42 IST',
    cycle: '2026-09-10 06:00 UTC',
    gridKm: 4,
    runtimeSec: 4.2,
    // Same idea as trainedAt: the canonical timestamp for "when did this
    // model last actually run inference", independent of the two display
    // strings (inferenceAt, cycle) already used elsewhere in this panel.
    lastUsedAt: '2026-09-10T07:42:00+05:30',
    lastUsedDisplay: '10 Sep 2026, 07:42 IST',
  },
  sources: [
    { name: 'INSAT-3D/3DR rapid scan', provider: 'ISRO MOSDAC', latest: '07:36 IST', ageMin: 6,   cadenceMin: 15 },
    { name: 'Doppler radar mosaic',    provider: 'IMD DWR',      latest: '07:40 IST', ageMin: 2,   cadenceMin: 10 },
    { name: 'Surface AWS / ARG',       provider: 'IMD',          latest: '07:27 IST', ageMin: 15,  cadenceMin: 15 },
    { name: 'GFS 0.25° boundary',      provider: 'NOAA NCEP',    latest: '06:00 UTC', ageMin: 102, cadenceMin: 360 },
    { name: 'CartoDEM terrain, flow routing', provider: 'ISRO Bhuvan', latest: 'v3 R1', ageMin: 0, cadenceMin: null },
  ],
  skill: {
    thunderstorm: { brier: [0.092, 0.108, 0.131, 0.157], reliability: [0.94, 0.91, 0.86, 0.80], cases: 2412, window: 'rolling 90 days' },
    cloudburst:   { brier: [0.121, 0.146, 0.178, 0.209], reliability: [0.89, 0.84, 0.77, 0.71], cases: 611,  window: 'rolling 90 days' },
    flashflood:   { brier: [0.104, 0.126, 0.152, 0.181], reliability: [0.92, 0.88, 0.82, 0.75], cases: 874,  window: 'rolling 90 days' },
  },
};

/* Reliability is a 0..1 calibration score; these cuts set the wording. */
function confidenceFor(reliability) {
  if (reliability >= 0.9) return 'High';
  if (reliability >= 0.78) return 'Moderate';
  return 'Limited';
}

function formatAge(minutes) {
  if (minutes < 60) return `${minutes} min`;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return m === 0 ? `${h} h` : `${h} h ${m} min`;
}

// Human "X ago" phrasing for the lifecycle stat cards, computed against the
// viewer's real clock -- this is display polish over static demo timestamps,
// not a claim that the underlying MODEL_STATUS values are live.
function relativeFrom(iso) {
  const diffMs = Date.now() - new Date(iso).getTime();
  if (diffMs < 60000) return 'just now';
  const min = Math.round(diffMs / 60000);
  if (min < 60) return `${min} min ago`;
  const hrs = Math.round(min / 60);
  if (hrs < 48) return `${hrs} h ago`;
  const days = Math.round(hrs / 24);
  if (days < 60) return `${days} d ago`;
  return `${Math.round(days / 30)} mo ago`;
}

function freshnessOf(source) {
  if (source.cadenceMin === null) return { text: 'static layer', late: false };
  if (source.ageMin > source.cadenceMin) {
    return { text: `${formatAge(source.ageMin - source.cadenceMin)} late`, late: true };
  }
  return { text: `${formatAge(source.ageMin)} old`, late: false };
}

function Row({ term, children }) {
  return (
    <div className="mp-row">
      <span className="mp-term">{term}</span>
      <span className="mp-val">{children}</span>
    </div>
  );
}

export default function ModelPanel({ region, hazard, step }) {
  // swap here: replace with the live response, e.g. useModelStatus(region, hazard)
  const status = MODEL_STATUS;

  const regionData = DATA[region];
  if (!regionData) return null;

  const hazardName = HAZARDS.find((h) => h.id === hazard)?.name ?? hazard;
  const stepLabel = STEPS[step] ?? STEPS[0];

  const skill = status.skill[hazard] ?? status.skill.thunderstorm;
  const brier = skill.brier[step] ?? skill.brier[skill.brier.length - 1];
  const reliability = skill.reliability[step] ?? skill.reliability[skill.reliability.length - 1];
  const confidence = confidenceFor(reliability);

  const lateCount = status.sources.filter((s) => freshnessOf(s).late).length;

  return (
    <section className="panel model-panel" aria-labelledby="mp-title">
      <div className="panel-head">
        <h2 id="mp-title">Model status</h2>
        <span className="mp-head-version mono">{status.model.version}</span>
      </div>

      <div className="panel-body">
        <p className="mp-context">
          {status.model.name}, run for {regionData.title} at{' '}
          <span className="mono">{status.run.inferenceAt}</span>.
        </p>

        <div className="mp-lifecycle">
          <div className="mp-stat">
            <span className="mp-stat-label">Last trained</span>
            <span className="mp-stat-relative">{relativeFrom(status.model.trainedAt)}</span>
            <span className="mp-stat-abs mono">{status.model.trainedThrough}</span>
          </div>
          <div className="mp-stat">
            <span className="mp-stat-label">Last used</span>
            <span className="mp-stat-relative">{relativeFrom(status.run.lastUsedAt)}</span>
            <span className="mp-stat-abs mono">{status.run.lastUsedDisplay}</span>
          </div>
        </div>

        <div className="mp-section">
          <h3 className="mp-section-title">This run</h3>
          <Row term="Architecture">{status.model.family}</Row>
          <Row term="Driving cycle"><span className="mono">{status.run.cycle}</span></Row>
          <Row term="Grid spacing"><span className="mono">{status.run.gridKm}</span> km</Row>
          <Row term="Inference time"><span className="mono">{status.run.runtimeSec.toFixed(1)}</span> s</Row>
          <Row term="Valid at"><span className="mono">{timeAt(step)}</span> ({stepLabel})</Row>
        </div>

        <div className="mp-section">
          <h3 className="mp-section-title">
            Inputs
            <span className="mp-section-note">
              {lateCount === 0
                ? 'all feeds current'
                : `${lateCount} feed${lateCount > 1 ? 's' : ''} behind cadence`}
            </span>
          </h3>
          <ul className="mp-sources">
            {status.sources.map((s) => {
              const fresh = freshnessOf(s);
              return (
                <li className="mp-source" key={s.name} data-late={fresh.late ? 'true' : 'false'}>
                  <span className="mp-source-main">
                    <span className="mp-source-name">{s.name}</span>
                    <span className="mp-source-provider">{s.provider}</span>
                  </span>
                  <span className="mp-source-age">
                    <span className="mono">{s.latest}</span>
                    <span className="mp-source-lag">{fresh.text}</span>
                  </span>
                </li>
              );
            })}
          </ul>
        </div>

        <div className="mp-section">
          <h3 className="mp-section-title">Calibration</h3>
          <p className="mp-calib-scope">
            {hazardName}, verified at {stepLabel} lead time.
          </p>

          <div className="mp-confidence">
            <span className="mp-confidence-word">{confidence} confidence</span>
            <span className="mp-confidence-track" aria-hidden="true">
              <span
                className="mp-confidence-fill"
                style={{ width: `${Math.round(reliability * 100)}%` }}
              />
            </span>
            <span className="mp-confidence-num mono">{reliability.toFixed(2)}</span>
          </div>

          <Row term="Reliability, 0 to 1"><span className="mono">{reliability.toFixed(2)}</span></Row>
          <Row term="Brier score"><span className="mono">{brier.toFixed(3)}</span></Row>
          <Row term="Verified cases">
            <span className="mono">{skill.cases.toLocaleString('en-IN')}</span>, {skill.window}
          </Row>
        </div>

        <p className="mp-note">
          Lower Brier is better. Reliability compares forecast probability against
          observed frequency. Skill falls with lead time, so read the +6h column
          with more caution than Now.
        </p>
        <p className="mp-placeholder">
          Diagnostics are placeholder values. Wire to live model telemetry before
          operational use.
        </p>
      </div>
    </section>
  );
}
