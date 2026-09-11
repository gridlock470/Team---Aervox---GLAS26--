import { HAZARDS, DATA, regionPeak } from '../data/nowcastData.js'
import { SEV, sevFor } from '../lib/severity.js'
import './NavRail.css'

export default function NavRail({ region, hazard, step, onHazardChange }) {
  const regionData = DATA[region]

  return (
    <nav className="nav-rail">
      <div className="nav-section">
        <h3>Hazard type</h3>
        <div className="hazard-list" role="group" aria-label="Hazard type">
          {HAZARDS.map((hz) => {
            const peak = regionPeak(region, hz.id, step)
            const sev = SEV[sevFor(peak)]
            return (
              <button
                key={hz.id}
                type="button"
                className="hazard-btn"
                aria-pressed={hazard === hz.id}
                style={{ borderLeftColor: sev.hex }}
                onClick={() => onHazardChange(hz.id)}
              >
                <span className="name">{hz.name}</span>
                <span className="reading">
                  <span className="sev-dot" style={{ background: sev.hex }}></span>
                  <span className="pct mono">{peak}%</span>
                </span>
              </button>
            )
          })}
        </div>
      </div>
      <p className="nav-note">
        {regionData.title} &mdash; {regionData.subtitle}
      </p>
    </nav>
  )
}
