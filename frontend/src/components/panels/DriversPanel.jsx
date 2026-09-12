import { useMemo } from 'react';
import { DATA, HAZARDS, STEPS, timeAt } from '../../data/nowcastData';
import './DriversPanel.css';

/*
 * DriversPanel — explainability view.
 *
 * Answers "why does the model think this?" for the selected region, hazard
 * and lead-time step. Drivers are ranked by their contribution weight at the
 * selected step, strongest first, so the dominant signal is always on top.
 *
 * Data comes straight from DATA[region].hazards[hazard].drivers, where each
 * driver is { label, unit, vals: [4 strings], w: [4 numbers 0..1] } and the
 * four entries line up with STEPS = ['Now','+2h','+4h','+6h'].
 *
 * Bars and sparklines use the neutral sequential tokens only. A driver is not
 * a hazard level, so severity colour is deliberately absent from this panel.
 */

const SPARK_W = 56;
const SPARK_H = 18;
const SPARK_PAD = 2.5;

/* Decimal places used by the source strings, so deltas read like the values. */
function decimalsOf(str) {
  const dot = String(str).indexOf('.');
  return dot === -1 ? 0 : String(str).length - dot - 1;
}

/* Signed change between two raw value strings, formatted to match them. */
function formatDelta(fromStr, toStr) {
  const from = Number(fromStr);
  const to = Number(toStr);
  if (!Number.isFinite(from) || !Number.isFinite(to)) return null;
  const diff = to - from;
  const dp = Math.max(decimalsOf(fromStr), decimalsOf(toStr));
  const magnitude = Math.abs(diff).toFixed(dp);
  if (Number(magnitude) === 0) return '0';
  return `${diff > 0 ? '+' : '−'}${magnitude}`;
}

/* Map a 4-point series onto the sparkline box, min at the bottom. */
function sparkPoints(series) {
  const min = Math.min(...series);
  const max = Math.max(...series);
  const span = max - min;
  const usableW = SPARK_W - SPARK_PAD * 2;
  const usableH = SPARK_H - SPARK_PAD * 2;
  const gap = series.length > 1 ? usableW / (series.length - 1) : 0;
  return series.map((v, i) => {
    const x = SPARK_PAD + i * gap;
    const y = span === 0
      ? SPARK_H / 2
      : SPARK_H - SPARK_PAD - ((v - min) / span) * usableH;
    return [Number(x.toFixed(2)), Number(y.toFixed(2))];
  });
}

function Sparkline({ driver, step }) {
  const series = driver.vals.map(Number);
  if (!series.every(Number.isFinite)) return <span className="dp-spark-empty" />;

  const pts = sparkPoints(series);
  const active = pts[step] ?? pts[pts.length - 1];
  const readout = driver.vals
    .map((v, i) => `${STEPS[i]} ${v}`)
    .join(', ');

  return (
    <svg
      className="dp-spark"
      viewBox={`0 0 ${SPARK_W} ${SPARK_H}`}
      width={SPARK_W}
      height={SPARK_H}
      role="img"
      aria-label={`${driver.label} in ${driver.unit}: ${readout}`}
    >
      <polyline
        className="dp-spark-line"
        points={pts.map(([x, y]) => `${x},${y}`).join(' ')}
      />
      {pts.map(([x, y], i) => (
        <circle key={STEPS[i]} className="dp-spark-dot" cx={x} cy={y} r="1.3" />
      ))}
      <line
        className="dp-spark-stem"
        x1={active[0]}
        y1={SPARK_PAD}
        x2={active[0]}
        y2={SPARK_H - SPARK_PAD}
      />
      <circle className="dp-spark-now" cx={active[0]} cy={active[1]} r="2.2" />
    </svg>
  );
}

export default function DriversPanel({ region, hazard, step }) {
  const regionData = DATA[region];
  const hazardData = regionData?.hazards?.[hazard];
  const drivers = hazardData?.drivers;

  const ranked = useMemo(
    () => (drivers ? [...drivers].sort((a, b) => b.w[step] - a.w[step]) : []),
    [drivers, step],
  );

  if (!ranked.length) return null;

  const hazardName = HAZARDS.find((h) => h.id === hazard)?.name ?? hazard;
  const stepLabel = STEPS[step] ?? STEPS[0];

  return (
    <section className="panel drivers-panel" aria-labelledby="dp-title">
      <div className="panel-head">
        <h2 id="dp-title">Why the model says this</h2>
        <span className="dp-head-step mono">{stepLabel}</span>
      </div>

      <div className="panel-body">
        <p className="dp-context">
          {hazardName} over {regionData.title}, ranked by contribution at{' '}
          <span className="mono">{timeAt(step)}</span>.
        </p>

        <div className="dp-legend">
          <span className="dp-legend-label">Lead time</span>
          <span className="dp-legend-steps">
            {STEPS.map((s, i) => (
              <span
                key={s}
                className="dp-legend-step mono"
                data-active={i === step ? 'true' : 'false'}
              >
                {s}
              </span>
            ))}
          </span>
        </div>

        <ol className="dp-list">
          {ranked.map((d, i) => {
            const weight = d.w[step];
            const delta = step > 0 ? formatDelta(d.vals[0], d.vals[step]) : null;

            return (
              <li className="dp-row" key={d.label} data-lead={i === 0 ? 'true' : 'false'}>
                <div className="dp-row-top">
                  <span className="dp-rank mono">{i + 1}</span>
                  <span className="dp-label">{d.label}</span>
                  <span className="dp-value mono">
                    {d.vals[step]}
                    <span className="dp-unit"> {d.unit}</span>
                  </span>
                </div>

                <div className="dp-row-bottom">
                  <span className="dp-track" aria-hidden="true">
                    <span
                      className="dp-fill"
                      style={{ width: `${Math.round(weight * 100)}%` }}
                    />
                  </span>
                  <span className="dp-weight mono" title="Contribution weight, 0 to 1">
                    {weight.toFixed(2)}
                  </span>
                  <Sparkline driver={d} step={step} />
                </div>

                {delta && (
                  <div className="dp-row-delta">
                    <span className="mono">{delta}</span> {d.unit} since Now
                  </div>
                )}
              </li>
            );
          })}
        </ol>

        <p className="dp-note">
          Weight is each driver&rsquo;s contribution at this step, 0 to 1. Weights
          are independent and do not sum to 1. The sparkline traces the
          driver&rsquo;s own value across all four steps; the marker is the step
          you have selected.
        </p>
      </div>
    </section>
  );
}
