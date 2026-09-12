import { DATA, STEPS } from '../../data/nowcastData.js';
import { SEV, sevFor } from '../../lib/severity.js';
import './PointsPanel.css';

/* Monitored points for the selected region, ranked by current probability so
   the point that needs attention is always first. Moved off the persistent
   rail into the dock -- it is reference data, not something to keep on screen. */
export default function PointsPanel({ region, hazard, step }) {
  const regionData = DATA[region];
  const vals = regionData.hazards[hazard].vals;

  const rows = Object.keys(regionData.stations)
    .map((id) => ({
      id,
      name: regionData.stations[id].name,
      lat: regionData.stations[id].lat,
      lng: regionData.stations[id].lng,
      pct: vals[id][step],
      series: vals[id],
    }))
    .sort((a, b) => b.pct - a.pct);

  return (
    <div className="points-panel">
      <p className="points-intro">
        Probability at <span className="mono">{STEPS[step].toLowerCase()}</span>, highest first.
      </p>

      <ul className="points-list">
        {rows.map((r) => {
          const sev = SEV[sevFor(r.pct)];
          return (
            <li className="point-row" key={r.id} style={{ borderLeftColor: sev.hex }}>
              <div className="point-main">
                <span className="point-name">{r.name}</span>
                <span className="point-pct mono">{r.pct}%</span>
              </div>
              <div className="point-sub">
                <span>{sev.label}</span>
                <span className="mono">
                  {r.lat.toFixed(3)}, {r.lng.toFixed(3)}
                </span>
              </div>
              <div className="point-track" aria-hidden="true">
                {r.series.map((v, i) => (
                  <span
                    key={i}
                    className={'point-tick' + (i === step ? ' is-current' : '')}
                    style={{ height: `${Math.max(2, Math.round(v * 0.22))}px` }}
                  />
                ))}
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
