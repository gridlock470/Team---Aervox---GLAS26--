import { DATA, timeAt } from '../../data/nowcastData';
import { SEV } from '../../lib/severity';
import './CapPanel.css';

function minutesOf(text) {
  const m = /(\d{1,2}):(\d{2})/.exec(String(text || ''));
  return m ? Number(m[1]) * 60 + Number(m[2]) : null;
}

function windowBounds(text) {
  const found = String(text || '').match(/\d{1,2}:\d{2}/g) || [];
  return { start: minutesOf(found[0]), end: minutesOf(found[1]) };
}

function clock(mins) {
  if (mins == null) return '';
  const h = Math.floor(mins / 60) % 24;
  const m = mins % 60;
  return `${h < 10 ? '0' : ''}${h}:${m < 10 ? '0' : ''}${m} IST`;
}

/* Status of a broadcast message against the console clock. */
function status(windowText, nowText) {
  const { start, end } = windowBounds(windowText);
  const now = minutesOf(nowText);
  if (start == null || now == null) return { lead: 'Broadcast', at: '' };
  if (now < start) return { lead: 'In force from', at: clock(start) };
  if (end == null) return { lead: 'In force', at: '' };
  if (now <= end) return { lead: 'In force until', at: clock(end) };
  return { lead: 'Closed', at: clock(end) };
}

export default function CapPanel({ region }) {
  const regionData = DATA[region];
  if (!regionData) return null;

  const now = timeAt(0);

  /* Latest broadcast first. Copy before sorting. */
  const rows = [...(regionData.alerts || [])].sort((a, b) =>
    String(b.sent).localeCompare(String(a.sent))
  );

  return (
    <section className="cap-panel" aria-label="CAP message log">
      <header className="cap-head">
        <h2>CAP message log</h2>
        <span className="cap-note">
          Latest first, clock <span className="mono">{now}</span>
        </span>
      </header>

      {rows.length === 0 ? (
        <p className="cap-empty">No CAP messages issued for {regionData.title}.</p>
      ) : (
        <div className="cap-table-wrap">
          <table className="cap-table">
            <thead>
              <tr>
                <th scope="col">Sent</th>
                <th scope="col">Message</th>
                <th scope="col" className="cap-sev-col">Severity</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((a) => {
                const sev = SEV[a.sev];
                const s = status(a.window, now);

                return (
                  <tr key={a.id} style={sev ? { '--sev': sev.hex } : undefined}>
                    <td className="cap-time mono">{a.sent}</td>
                    <td className="cap-msg">
                      <span className="cap-headline">{a.headline}</span>
                      <span className="cap-meta">
                        <span className="mono">{a.id}</span>
                        {' · '}
                        {s.lead}
                        {s.at ? (
                          <>
                            {' '}
                            <span className="mono">{s.at}</span>
                          </>
                        ) : null}
                      </span>
                      <span className="cap-area">{a.area}</span>
                    </td>
                    <td className="cap-sev">{sev ? sev.label : a.sev}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
