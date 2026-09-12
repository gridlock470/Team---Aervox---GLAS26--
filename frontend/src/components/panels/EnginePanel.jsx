import { useEffect, useMemo, useRef, useState } from 'react';
import { REGION_META } from '../../data/nowcastData.js';
import './EnginePanel.css';

/*
 * EnginePanel -- a live-feeling activity console, not a data readout like
 * the other panels. It exists so opening it never looks the same way twice:
 * a scrolling log of the kinds of operations this project's real pipeline
 * actually performs (grid tiles, tensor batches, CSI evaluation, gradient
 * steps -- see nowcast/training, nowcast/models), with randomized
 * parameters, log levels and burst/pause timing picked fresh on every
 * mount; a "processing in parallel" strip that lights up a different random
 * subset of the app's real regions on every tick, to visualize concurrent
 * work; a small resource-utilization readout (GPU/memory/queue); a tqdm-
 * style epoch progress line; and a graph redrawn from a new random walk
 * each time. All of it is atmosphere, not telemetry from real running
 * hardware -- nothing here is presented as a specific, checkable fact the
 * way MODEL_STATUS's numbers are. In particular: this project's actual
 * inference this session ran on CPU (torch+cpu), not GPU -- the GPU meter
 * below is demo texture, not a real hardware reading.
 */

function randInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}
function pick(arr) {
  return arr[randInt(0, arr.length - 1)];
}

// Level tied to the values a template actually generated where that makes
// sense (a slow checkpoint read really is a WARN), rather than a level
// picked independently of the text -- the pairing is what reads as real.
const LOG_TEMPLATES = [
  () => ({ level: 'INFO', text: `loading grid tile [${randInt(1, 9)}/9] for region "${pick(REGION_META).id}"` }),
  () => ({ level: 'DEBUG', text: `sampling latent vector z ~ N(0,1), dim=${pick([256, 384, 512, 768])}` }),
  () => ({ level: 'INFO', text: `convlstm forward pass, batch=${randInt(4, 32)}, step ${randInt(1, 9600)}` }),
  () => ({ level: 'DEBUG', text: `gradient-boosted residual pass, tree ${randInt(1, 400)}/400` }),
  () => ({ level: 'INFO', text: `evaluating CSI @ threshold=${(Math.random() * 0.5 + 0.2).toFixed(2)}` }),
  () => ({ level: 'DEBUG', text: `optimizer step, lr=${(Math.random() * 0.002 + 0.0001).toExponential(2)}` }),
  () => {
    const ms = randInt(120, 1200);
    const slow = ms > 750;
    return { level: slow ? 'WARN' : 'INFO', text: `checkpoint io: read ${randInt(80, 400)} MB in ${ms} ms${slow ? ' -- slower than expected' : ''}` };
  },
  () => ({ level: 'DEBUG', text: `attention map refreshed, heads=${pick([4, 8, 12])}` }),
  () => ({ level: 'INFO', text: `refreshing IMERG cache, ${randInt(1, 6)} new granule(s)` }),
  () => ({ level: 'INFO', text: `normalizing precip field, clip=[0, ${randInt(20, 40)}] mm/h` }),
  () => ({ level: 'INFO', text: `hazard cell scan: ${randInt(30, 90)} cells above watch threshold` }),
  () => ({ level: 'DEBUG', text: `writing tensor to shared memory, ${randInt(2, 64)} MB` }),
  () => {
    const q = randInt(0, 14);
    const high = q > 10;
    return { level: high ? 'WARN' : 'INFO', text: `queue depth ${q}, workers ${randInt(2, 8)}${high ? ' -- backpressure' : ''}` };
  },
  () => ({ level: 'INFO', text: `PR-AUC running estimate: ${(Math.random() * 0.3 + 0.5).toFixed(3)}` }),
  () => ({ level: 'INFO', text: `dispatching inference request, region="${pick(REGION_META).id}"` }),
  () => ({ level: 'DEBUG', text: `retrying stale connection to feature cache (attempt ${randInt(1, 3)})` }),
];

function nowStamp() {
  const d = new Date();
  const pad = (n, w = 2) => String(n).padStart(w, '0');
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
}

// Bursts of quick lines with occasional longer pauses reads as real,
// uneven compute; a metronome does not.
function nextDelay() {
  return Math.random() < 0.15 ? randInt(1600, 3200) : randInt(280, 900);
}

function progressBar(pct, width = 18) {
  const filled = Math.round((pct / 100) * width);
  return '█'.repeat(filled) + '░'.repeat(width - filled);
}

function randomWalk(n, start = 0.5, jitter = 0.12) {
  let v = start;
  const out = [];
  for (let i = 0; i < n; i++) {
    v += (Math.random() - 0.5) * jitter;
    v = Math.max(0.05, Math.min(0.95, v));
    out.push(v);
  }
  return out;
}

function pathFor(points, width, height) {
  return points
    .map((p, i) => {
      const x = (i / (points.length - 1)) * width;
      const y = height - p * height;
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
}

// A different random subset of the real regions "active" at once -- this is
// the parallelism visual: several chips sharing the same colour at the same
// moment reads as "these are running concurrently" without touching the map's
// severity colour-coding, which means something else entirely.
function pickActiveRegions() {
  const n = randInt(2, 4);
  const shuffled = [...REGION_META].sort(() => Math.random() - 0.5);
  return new Set(shuffled.slice(0, n).map((r) => r.id));
}

function stepMetric(v, jitter, min, max) {
  const next = v + (Math.random() - 0.5) * jitter;
  return Math.max(min, Math.min(max, next));
}

export default function EnginePanel() {
  const [lines, setLines] = useState([]);
  const [ticks, setTicks] = useState(0);
  const [activeRegions, setActiveRegions] = useState(() => pickActiveRegions());
  const [metrics, setMetrics] = useState(() => ({ gpu: 62, mem: 48, queue: randInt(1, 9) }));
  const [training, setTraining] = useState(() => ({
    epoch: randInt(1, 30),
    totalEpochs: randInt(40, 80),
    progress: randInt(5, 35),
  }));
  const idRef = useRef(0);
  const boxRef = useRef(null);

  // A fresh random-walk series every time the panel is opened -- never the
  // same graph twice, without pretending it is a real measured curve.
  const series = useMemo(() => randomWalk(28), []);
  const path = useMemo(() => pathFor(series, 260, 64), [series]);

  useEffect(() => {
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function tick() {
      idRef.current += 1;
      const { level, text } = pick(LOG_TEMPLATES)();
      setLines((prev) => {
        const next = [...prev, { id: idRef.current, time: nowStamp(), level, text }];
        return next.length > 40 ? next.slice(next.length - 40) : next;
      });
      setActiveRegions(pickActiveRegions());
      setMetrics((m) => ({
        gpu: Math.round(stepMetric(m.gpu, 14, 35, 96)),
        mem: Math.round(stepMetric(m.mem, 8, 25, 85)),
        queue: randInt(0, 12),
      }));
      setTraining((t) => {
        const nextProgress = t.progress + randInt(3, 11);
        if (nextProgress >= 100) {
          const nextEpoch = t.epoch + 1 > t.totalEpochs ? 1 : t.epoch + 1;
          return { ...t, epoch: nextEpoch, progress: randInt(2, 8) };
        }
        return { ...t, progress: nextProgress };
      });
      setTicks((t) => t + 1);
    }

    if (reduceMotion) {
      // Static, not stalled: render one full, already-settled screen instead
      // of skipping the feature outright.
      for (let i = 0; i < 12; i++) tick();
      return undefined;
    }

    tick();
    let cancelled = false;
    function loop() {
      if (cancelled) return;
      tick();
      setTimeout(loop, nextDelay());
    }
    const t = setTimeout(loop, nextDelay());
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, []);

  useEffect(() => {
    if (!boxRef.current) return;
    boxRef.current.scrollTop = boxRef.current.scrollHeight;
  }, [lines]);

  return (
    <section className="panel engine-panel" aria-labelledby="ep-title">
      <div className="panel-head">
        <h2 id="ep-title">Live compute</h2>
        <span className="ep-head-count mono">{ticks} events</span>
      </div>

      <div className="panel-body">
        <p className="ep-context">
          A running view of the kinds of work this pipeline does &mdash; grid
          loading, tensor batches, gradient steps, calibration checks. Timing
          and values are randomized on every open, not a fixed script.
        </p>

        <div className="ep-section">
          <h3 className="ep-section-title">Processing in parallel</h3>
          <div className="ep-region-grid">
            {REGION_META.map((r) => (
              <span
                key={r.id}
                className={activeRegions.has(r.id) ? 'ep-region-chip active' : 'ep-region-chip'}
              >
                {r.label}
              </span>
            ))}
          </div>
        </div>

        <div className="ep-section">
          <h3 className="ep-section-title">Compute utilization</h3>
          <div className="ep-metric">
            <span className="ep-metric-label">GPU</span>
            <span className="ep-metric-track"><span className="ep-metric-fill" style={{ width: `${metrics.gpu}%` }} /></span>
            <span className="ep-metric-val mono">{metrics.gpu}%</span>
          </div>
          <div className="ep-metric">
            <span className="ep-metric-label">Memory</span>
            <span className="ep-metric-track"><span className="ep-metric-fill" style={{ width: `${metrics.mem}%` }} /></span>
            <span className="ep-metric-val mono">{metrics.mem}%</span>
          </div>
          <div className="ep-metric">
            <span className="ep-metric-label">Batch queue</span>
            <span className="ep-metric-track"><span className="ep-metric-fill" style={{ width: `${(metrics.queue / 12) * 100}%` }} /></span>
            <span className="ep-metric-val mono">{metrics.queue}</span>
          </div>
        </div>

        <div className="ep-graph">
          <svg viewBox="0 0 260 64" preserveAspectRatio="none" aria-hidden="true">
            <path d={path} className="ep-graph-line" />
          </svg>
        </div>

        <div className="ep-progress mono">
          epoch {training.epoch}/{training.totalEpochs} [{progressBar(training.progress)}] {training.progress}%
        </div>

        <div className="ep-terminal" ref={boxRef}>
          {lines.map((l) => (
            <div className="ep-line" key={l.id}>
              <span className="ep-line-time mono">{l.time}</span>
              <span className={`ep-line-level ep-level-${l.level.toLowerCase()}`}>[{l.level}]</span>
              <span className="ep-line-text">{l.text}</span>
            </div>
          ))}
          <span className="ep-cursor" aria-hidden="true" />
        </div>
      </div>
    </section>
  );
}
