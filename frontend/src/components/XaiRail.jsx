import { SEV } from '../lib/severity';
import { DATA } from '../data/nowcastData';
import './XaiRail.css';

export default function XaiRail({ region, hazard, step }) {
  const regionData = DATA[region];
  const drivers = regionData.hazards[hazard].drivers;
  const alerts = regionData.alerts;

  return (
    <aside className="xai-rail">
      <div className="panel">
        <div className="panel-head"><h2>What’s driving this forecast</h2></div>
        <div className="panel-body">
          {drivers.map((d) => (
            <div className="driver-row" key={d.label}>
              <div className="driver-top">
                <span className="label">{d.label}</span>
                <span className="value mono">{d.vals[step]} {d.unit}</span>
              </div>
              <div className="driver-track">
                <div className="driver-fill" style={{ width: `${d.w[step] * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className="panel">
        <div className="panel-head"><h2>Alerts issued</h2></div>
        <div className="panel-body">
          {alerts.map((a) => {
            const sev = SEV[a.sev];
            return (
              <div className="alert-card" style={{ borderLeftColor: sev.hex }} key={a.id}>
                <div className="alert-top">
                  <span className="alert-sev">
                    <span className="sev-dot" style={{ background: sev.hex }} />
                    {sev.label}
                  </span>
                  <span className="alert-id mono">{a.id}</span>
                </div>
                <div className="alert-headline">{a.headline}</div>
                <div className="alert-meta">
                  {a.area}
                  <br />
                  Effective <span className="mono">{a.window}</span> · sent <span className="mono">{a.sent}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </aside>
  );
}
