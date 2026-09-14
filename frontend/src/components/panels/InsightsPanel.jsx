import { useRef, useState } from 'react';
import { uploadNetcdfInsights } from '../../lib/api.js';
import './InsightsPanel.css';

/*
 * InsightsPanel — drop in a NetCDF (.nc) file, get real statistics and
 * charts computed from its actual contents. The heavy lifting (opening the
 * file, computing per-variable mean/min/max/std, a real histogram, and a
 * real time-mean trend where a time dimension exists) happens in Python
 * (scripts/inspect_netcdf.py), since NetCDF4/HDF5 is a genuinely complex
 * binary format with no reasonable client-side parser -- this component
 * just uploads the file and renders whatever the backend honestly reports.
 * A file that can't be opened shows the backend's specific error message;
 * nothing here is ever invented to fill a gap.
 */

const CHART_W = 320;
const CHART_H = 120;
const CHART_PAD = 12;

function formatNum(n) {
  if (n == null || !Number.isFinite(n)) return '—';
  return Math.abs(n) >= 1000 ? n.toLocaleString('en-IN', { maximumFractionDigits: 1 }) : n.toFixed(2).replace(/\.00$/, '');
}

function Histogram({ histogram }) {
  if (!histogram) return <p className="ip-note">No values to chart.</p>;
  const { edges, counts } = histogram;
  const maxCount = Math.max(...counts, 1);
  const barW = (CHART_W - CHART_PAD * 2) / counts.length;

  return (
    <svg
      className="ip-chart"
      viewBox={`0 0 ${CHART_W} ${CHART_H}`}
      role="img"
      aria-label={`Histogram ranging from ${formatNum(edges[0])} to ${formatNum(edges[edges.length - 1])}`}
    >
      {counts.map((count, i) => {
        const h = (count / maxCount) * (CHART_H - CHART_PAD * 2);
        return (
          <rect
            key={i}
            className="ip-hist-bar"
            x={CHART_PAD + i * barW + 1}
            y={CHART_H - CHART_PAD - h}
            width={Math.max(1, barW - 2)}
            height={h}
          />
        );
      })}
    </svg>
  );
}

function Trend({ trend }) {
  if (!trend || !trend.values.length) return null;
  const finite = trend.values.filter((v) => Number.isFinite(v));
  if (!finite.length) return null;
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const span = max - min || 1;
  const usableW = CHART_W - CHART_PAD * 2;
  const usableH = CHART_H - CHART_PAD * 2;
  const gap = trend.values.length > 1 ? usableW / (trend.values.length - 1) : 0;
  const points = trend.values
    .map((v, i) => {
      if (!Number.isFinite(v)) return null;
      const x = CHART_PAD + i * gap;
      const y = CHART_H - CHART_PAD - ((v - min) / span) * usableH;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .filter(Boolean);

  return (
    <svg
      className="ip-chart"
      viewBox={`0 0 ${CHART_W} ${CHART_H}`}
      role="img"
      aria-label={`Trend over time, from ${trend.labels[0]} to ${trend.labels[trend.labels.length - 1]}, ranging from ${formatNum(min)} to ${formatNum(max)}`}
    >
      <polyline className="ip-trend-line" points={points.join(' ')} />
    </svg>
  );
}

export default function InsightsPanel({ token }) {
  const [fileName, setFileName] = useState(null);
  const [status, setStatus] = useState('idle'); // idle | loading | error | done
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  async function handleFile(file) {
    if (!file) return;
    setResult(null);
    setError(null);
    setFileName(file.name);

    if (!/\.nc$/i.test(file.name)) {
      setStatus('error');
      setError('Only .nc (NetCDF) files are supported.');
      return;
    }

    setStatus('loading');
    try {
      const data = await uploadNetcdfInsights(token, file);
      setResult(data);
      setStatus('done');
    } catch (e) {
      setStatus('error');
      setError(e.message || 'Could not analyze this file.');
    }
  }

  return (
    <section className="panel insights-panel" aria-labelledby="ip-title">
      <div className="panel-head">
        <h2 id="ip-title">Dataset insights</h2>
        {fileName && <span className="ip-filename mono">{fileName}</span>}
      </div>

      <div className="panel-body">
        <div
          className={dragOver ? 'ip-dropzone is-over' : 'ip-dropzone'}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            handleFile(e.dataTransfer.files?.[0]);
          }}
        >
          <i className="fa-solid fa-file-code" aria-hidden="true"></i>
          <p>Drag a NetCDF (.nc) file here, or</p>
          <button type="button" className="ip-browse-btn" onClick={() => inputRef.current?.click()}>
            Browse files
          </button>
          <input
            ref={inputRef}
            type="file"
            accept=".nc"
            className="sr-only"
            aria-label="Upload a NetCDF file"
            onChange={(e) => handleFile(e.target.files?.[0])}
          />
        </div>

        {status === 'loading' && (
          <p className="ip-summary" role="status">Analyzing {fileName}&hellip;</p>
        )}

        {error && <p className="ip-error" role="alert">{error}</p>}

        {result && !result.error && (
          <>
            <p className="ip-summary">
              {Object.entries(result.dims).map(([d, n]) => `${d}: ${n}`).join(', ') || 'No dimensions found.'}
            </p>

            {result.variables.length === 0 && (
              <p className="ip-note">This file has no data variables to summarize.</p>
            )}

            <div className="ip-var-list">
              {result.variables.map((v) => (
                <div className="ip-var-card" key={v.name}>
                  <div className="ip-var-head">
                    <span className="ip-var-name">{v.longName || v.name}</span>
                    <span className="ip-var-meta mono">{v.name}{v.units ? ` · ${v.units}` : ''}</span>
                  </div>
                  <div className="ip-var-shape mono">shape: [{v.shape.join(', ')}]</div>

                  {v.count > 0 ? (
                    <>
                      <dl className="ip-column-facts">
                        <dt>Mean</dt><dd className="mono">{formatNum(v.mean)}</dd>
                        <dt>Min</dt><dd className="mono">{formatNum(v.min)}</dd>
                        <dt>Max</dt><dd className="mono">{formatNum(v.max)}</dd>
                        <dt>Std dev</dt><dd className="mono">{formatNum(v.std)}</dd>
                        <dt>Missing</dt><dd className="mono">{v.missing.toLocaleString('en-IN')}</dd>
                      </dl>
                      <div className="ip-charts">
                        <div className="ip-chart-block">
                          <span className="ip-chart-label">Distribution</span>
                          <Histogram histogram={v.histogram} />
                        </div>
                        {v.trend && (
                          <div className="ip-chart-block">
                            <span className="ip-chart-label">Trend over time</span>
                            <Trend trend={v.trend} />
                          </div>
                        )}
                      </div>
                    </>
                  ) : (
                    <p className="ip-note">Every value in this variable is missing.</p>
                  )}
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </section>
  );
}
