import { Fragment } from 'react';
import { STEPS, timeAt } from '../data/nowcastData';
import './TimelineStrip.css';

export default function TimelineStrip({ step, onStepChange }) {
  return (
    <div className="timeline-strip">
      <div className="timeline-controls">
        <div className="timeline-track" role="group" aria-label="Lead time">
          {STEPS.map((label, i) => (
            <Fragment key={label}>
              {i > 0 && <span className="step-connector" />}
              <button
                type="button"
                className="step-btn"
                aria-pressed={step === i}
                onClick={() => onStepChange(i)}
              >
                {label}
              </button>
            </Fragment>
          ))}
        </div>
        <div className="timeline-readout">
          Forecast issued <span className="mono">{timeAt(0)}</span>, lead time <span className="mono">{STEPS[step].toLowerCase()}</span>
        </div>
      </div>
      <p className="provenance">Model inputs: IMDAA reanalysis (NCMRWF), INSAT-3D/3DR water vapour, thermal-IR and rainfall retrievals (MOSDAC), and CartoDEM/SRTM elevation. Figures shown are illustrative for this prototype walkthrough.</p>
    </div>
  );
}
