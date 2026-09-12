import { HAZARDS, DATA, regionPeak } from '../data/nowcastData.js'
import { SEV, sevFor } from '../lib/severity.js'
import './NavRail.css'

const HAZARD_ICON = {
  thunderstorm: 'fa-solid fa-bolt',
  cloudburst: 'fa-solid fa-cloud-showers-heavy',
  flashflood: 'fa-solid fa-water',
}

export default function NavRail({ region, hazard, step, onHazardChange }) {
  const regionData = DATA[region]
  const hazardData = regionData.hazards[hazard]
  const alertCount = regionData.alerts.length

  const topDrivers = [...(hazardData.drivers || [])]
    .sort((a, b) => b.w[step] - a.w[step])
    .slice(0, 4)

  return (
    <nav className="nav-rail">
      <div className="nav-heading">
        <i className="fa-solid fa-triangle-exclamation" aria-hidden="true"></i> Active Hazards &amp; Parameters
      </div>

      <div className="nav-section">
        <h3>Hazard Categories</h3>
        <div className="hazard-card-list" role="group" aria-label="Hazard type">
          {HAZARDS.map((hz) => {
            const peak = regionPeak(region, hz.id, step)
            const sev = SEV[sevFor(peak)]
            const active = hazard === hz.id
            return (
              <button
                key={hz.id}
                type="button"
                className={active ? 'hazard-card active' : 'hazard-card'}
                aria-pressed={active}
                onClick={() => onHazardChange(hz.id)}
              >
                <span className="hazard-info">
                  <span className="hazard-icon-wrapper">
                    <i className={HAZARD_ICON[hz.id]} style={{ color: sev.hex }} aria-hidden="true"></i>
                  </span>
                  <span className="hazard-name">{hz.name}</span>
                </span>
                <span className="hazard-badge" style={{ background: sev.hex }}>
                  {sev.label.toUpperCase()} &middot; <span className="mono">{peak}%</span>
                </span>
              </button>
            )
          })}
        </div>
      </div>

      <div className="nav-section">
        <h3>Atmospheric Parameters</h3>
        <div className="hazard-detail-box">
          {topDrivers.map((d) => (
            <div className="metric-row" key={d.label}>
              <span><i className="fa-solid fa-gauge-high" aria-hidden="true"></i> {d.label}:</span>
              <strong className="mono">{d.vals[step]} {d.unit}</strong>
            </div>
          ))}
        </div>
      </div>

      <p className="nav-note">
        {regionData.title} &mdash; {regionData.subtitle}
        <span className="nav-alert-count">
          <i className="fa-solid fa-bell" aria-hidden="true"></i> {alertCount} active CAP alert{alertCount === 1 ? '' : 's'}
        </span>
      </p>
    </nav>
  )
}
