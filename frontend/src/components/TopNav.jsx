import { HAZARDS, regionPeak } from '../data/nowcastData.js'
import { SEV, sevFor } from '../lib/severity.js'
import './TopNav.css'

const HAZARD_ICON = {
  thunderstorm: 'fa-solid fa-bolt',
  cloudburst: 'fa-solid fa-cloud-showers-heavy',
  flashflood: 'fa-solid fa-water',
}

/**
 * Slim top bar replacing the old left rail (hazard picker) and bottom dock
 * (panel tabs). Both used to be always-visible real estate; here they are
 * a single row of pills/buttons and the panels they open are on-demand
 * popups (see InsightModal), so the map keeps the rest of the screen.
 */
export default function TopNav({ region, hazard, step, onHazardChange, tabs, onOpenPanel }) {
  return (
    <nav className="top-nav">
      <div className="top-nav-hazards" role="group" aria-label="Hazard type">
        {HAZARDS.map((hz) => {
          const peak = regionPeak(region, hz.id, step)
          const sev = SEV[sevFor(peak)]
          const active = hazard === hz.id
          return (
            <button
              key={hz.id}
              type="button"
              className={active ? 'hazard-pill active' : 'hazard-pill'}
              aria-pressed={active}
              onClick={() => onHazardChange(hz.id)}
            >
              <i className={HAZARD_ICON[hz.id]} style={{ color: sev.hex }} aria-hidden="true"></i>
              <span className="hazard-pill-name">{hz.name}</span>
              <span className="hazard-pill-badge" style={{ background: sev.hex }}>{peak}%</span>
            </button>
          )
        })}
      </div>

      <div className="top-nav-insights" role="group" aria-label="Insights">
        {tabs.map((tab) => {
          const badge = Number(tab.badge)
          const showBadge = Number.isFinite(badge) && badge > 0
          return (
            <button
              key={tab.id}
              type="button"
              className="insight-btn"
              onClick={() => onOpenPanel(tab.id)}
            >
              {tab.icon && <i className={tab.icon} aria-hidden="true"></i>}
              <span>{tab.label}</span>
              {showBadge && <span className="insight-btn-badge mono">{badge}</span>}
            </button>
          )
        })}
      </div>
    </nav>
  )
}
