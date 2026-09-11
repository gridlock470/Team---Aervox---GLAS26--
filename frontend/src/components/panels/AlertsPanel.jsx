import { useState } from 'react';
import { DATA, HAZARDS, STEPS, timeAt } from '../../data/nowcastData';
import { SEV, SEV_ORDER } from '../../lib/severity';
import './AlertsPanel.css';

/* Response guidance is keyed off alert severity. nowcastData carries no
   action field, so the wording lives here rather than being faked in data. */
const ACTION = {
  red: 'Notify the district EOC now. Move response teams to staging points and hold traffic on the affected routes.',
  orange: 'Brief the district control room. Pre-position response teams and warn transport and pilgrimage operators.',
  yellow: 'Watch the next two frames. Put drainage and traffic crews on call and brief field units.',
  green: 'No action. Keep routine monitoring.',
};

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

function gapLabel(mins) {
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  if (h && m) return `${h}h ${m}m`;
  if (h) return `${h}h`;
  return `${m}m`;
}

/* Where the effective window sits relative to the frame on screen. */
function timing(windowText, frameText) {
  const { start, end } = windowBounds(windowText);
  const now = minutesOf(frameText);
  if (start == null || now == null) return { kind: 'unknown' };
  if (now < start) return { kind: 'pending', gap: gapLabel(start - now), at: clock(start) };
  if (end == null) return { kind: 'active', gap: null, at: null };
  if (now <= end) return { kind: 'active', gap: gapLabel(end - now), at: clock(end) };
  return { kind: 'closed', at: clock(end) };
}

export default function AlertsPanel({ region, hazard, step }) {
  const [openIds, setOpenIds] = useState([]);

  const regionData = DATA[region];
  if (!regionData) return null;

  const frame = timeAt(step);
  const hazardName = (HAZARDS.find((h) => h.id === hazard)?.name || String(hazard)).toLowerCase();
  const stepLabel = STEPS[step] || STEPS[0];
  const stepPhrase = stepLabel === 'Now' ? 'now' : `at ${stepLabel}`;

  /* Most severe first, then most recently sent. Copy before sorting. */
  const alerts = [...(regionData.alerts || [])].sort(
    (a, b) =>
      SEV_ORDER.indexOf(b.sev) - SEV_ORDER.indexOf(a.sev) ||
      String(b.sent).localeCompare(String(a.sent))
  );

  const toggle = (id) =>
    setOpenIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));

  return (
    <section className="alerts-panel" aria-label="Active alerts">
      <header className="ap-head">
        <h2>Active alerts</h2>
        <span className="ap-count mono">{alerts.length}</span>
      </header>

      <p className="ap-note">
        All hazards listed. Map shows {hazardName} {stepPhrase}.
      </p>

      {alerts.length === 0 ? (
        <p className="ap-empty">No alerts in force for {regionData.title}.</p>
      ) : (
        <ul className="ap-list">
          {alerts.map((a) => {
            const sev = SEV[a.sev];
            const open = openIds.includes(a.id);
            const detailId = `ap-detail-${a.id}`;
            const t = timing(a.window, frame);

            return (
              <li
                className="ap-item"
                key={a.id}
                style={sev ? { '--sev': sev.hex } : undefined}
              >
                <button
                  type="button"
                  className="ap-row"
                  aria-expanded={open}
                  aria-controls={detailId}
                  onClick={() => toggle(a.id)}
                >
                  <span className="ap-line-top">
                    <span className="ap-sev">{sev ? sev.label : a.sev}</span>
                    <span className="ap-when mono">{a.window}</span>
                  </span>
                  <span className="ap-headline">{a.headline}</span>
                  <span className="ap-where">{a.area}</span>
                  <span className="ap-chev" aria-hidden="true" />
                </button>

                {open && (
                  <div className="ap-detail" id={detailId}>
                    <dl className="ap-facts">
                      <dt>Area</dt>
                      <dd>{a.area}</dd>

                      <dt>Effective</dt>
                      <dd>
                        <span className="mono">{a.window}</span>
                        {t.kind === 'pending' && (
                          <span className="ap-timing">
                            Starts in <span className="mono">{t.gap}</span> from the{' '}
                            <span className="mono">{frame}</span> frame.
                          </span>
                        )}
                        {t.kind === 'active' && (
                          <span className="ap-timing">
                            In force at the <span className="mono">{frame}</span> frame
                            {t.gap ? (
                              <>
                                , <span className="mono">{t.gap}</span> left
                              </>
                            ) : null}
                            .
                          </span>
                        )}
                        {t.kind === 'closed' && (
                          <span className="ap-timing">
                            Closed at <span className="mono">{t.at}</span>.
                          </span>
                        )}
                      </dd>

                      <dt>Issued</dt>
                      <dd className="mono">{a.sent}</dd>

                      <dt>Message id</dt>
                      <dd className="mono">{a.id}</dd>
                    </dl>

                    <p className="ap-action">
                      <span className="ap-action-label">Action</span>
                      {ACTION[a.sev] || ACTION.green}
                    </p>
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
