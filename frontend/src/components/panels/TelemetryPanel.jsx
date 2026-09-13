import { DATA, HAZARDS, STEPS, timeAt, regionPeak } from '../../data/nowcastData';
import { SEV, sevFor, radiusMeters } from '../../lib/severity';
import { MODEL_STATUS } from './ModelPanel.jsx';
import './TelemetryPanel.css';

/*
 * TelemetryPanel — the pasted AEROCAST mock's "Technical Telemetry & Impact
 * Grid": three headline stat cards plus a per-station table. Every number
 * here is derived from the same DATA/MODEL_STATUS the other panels already
 * read, not re-invented, so this view stays consistent with Drivers/Model/
 * Points if any of them change.
 */

function confidenceWord(reliability) {
  if (reliability >= 0.9) return 'High';
  if (reliability >= 0.78) return 'Moderate';
  return 'Limited';
}

export default function TelemetryPanel({ region, hazard, step }) {
  const regionData = DATA[region];
  const hazardData = regionData?.hazards?.[hazard];
  if (!regionData || !hazardData) return null;

  const hazardName = HAZARDS.find((h) => h.id === hazard)?.name ?? hazard;
  const stepLabel = STEPS[step] ?? STEPS[0];

  const skill = MODEL_STATUS.skill[hazard] ?? MODEL_STATUS.skill.thunderstorm;
  const reliability = skill.reliability[step] ?? skill.reliability[skill.reliability.length - 1];
  const brier = skill.brier[step] ?? skill.brier[skill.brier.length - 1];

  const peak = regionPeak(region, hazard, step);
  const areaKm2 = (Math.PI * (radiusMeters(peak) / 1000) ** 2);

  const rows = Object.keys(regionData.stations)
    .map((id) => {
      const station = regionData.stations[id];
      const pct = hazardData.vals[id][step];
      const topDriver = [...(hazardData.drivers || [])].sort((a, b) => b.w[step] - a.w[step])[0];
      return { id, station, pct, topDriver };
    })
    .sort((a, b) => b.pct - a.pct);

  const peakRow = rows[0];

  return (
    <div className="telemetry-panel">
      <p className="telemetry-context">
        {hazardName} over {regionData.title}, at <span className="mono">{timeAt(step)}</span> ({stepLabel.toLowerCase()}).
      </p>

      <div className="telemetry-cards">
        <div className="detail-card">
          <div className="detail-card-header">
            <span>Selected Lead Time</span>
            <i className="fa-regular fa-clock" aria-hidden="true"></i>
          </div>
          <div className="detail-value mono">{stepLabel}</div>
          <div className="detail-subtext">Forecast issued {timeAt(0)}, valid at {timeAt(step)} for the selected frame.</div>
        </div>

        <div className="detail-card">
          <div className="detail-card-header">
            <span>Confidence Index</span>
            <i className="fa-solid fa-shield-halved" aria-hidden="true"></i>
          </div>
          <div className="detail-value mono">{Math.round(reliability * 100)}%</div>
          <div className="detail-subtext">{confidenceWord(reliability)} confidence &middot; Brier {brier.toFixed(3)} over {skill.cases.toLocaleString('en-IN')} verified cases.</div>
        </div>

        <div className="detail-card">
          <div className="detail-card-header">
            <span>Impact Area</span>
            <i className="fa-solid fa-expand" aria-hidden="true"></i>
          </div>
          <div className="detail-value mono">{areaKm2.toFixed(1)} km&sup2;</div>
          <div className="detail-subtext">Peak-probability grid cell{peakRow ? `, ${peakRow.station.name}` : ''} &mdash; broadcast radius at {peak}%.</div>
        </div>
      </div>

      <div className="detail-table-card">
        <table>
          <caption className="sr-only">
            Per-station probability, leading driver and status for {hazardName} at {timeAt(step)}
          </caption>
          <thead>
            <tr>
              <th scope="col">Station</th>
              <th scope="col">Location Grid</th>
              <th scope="col">Probability</th>
              <th scope="col">Leading Driver</th>
              <th scope="col">Status</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const sev = SEV[sevFor(r.pct)];
              return (
                <tr key={r.id}>
                  <td className="mono">{r.id}</td>
                  <td>{r.station.name}</td>
                  <td className="mono">{r.pct}%</td>
                  <td>{r.topDriver ? `${r.topDriver.label}: ${r.topDriver.vals[step]} ${r.topDriver.unit}` : '—'}</td>
                  <td>
                    <span className="status-pill" style={{ background: sev.hex }}>{sev.label.toUpperCase()}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
