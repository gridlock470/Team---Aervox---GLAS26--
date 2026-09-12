import { useEffect, useRef } from 'react'
import './InsightModal.css'

/**
 * Popup that a top-nav insight button opens. Replaces the old always-docked
 * bottom panel: the panel content only exists on screen while an operator
 * has actually asked to see it, so the map owns the rest of the viewport.
 */
export default function InsightModal({ open, title, icon, onClose, children }) {
  const dialogRef = useRef(null)

  useEffect(() => {
    if (!open) return undefined
    function onKey(e) {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    dialogRef.current?.focus()
    return () => document.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

  return (
    <div className="insight-modal-backdrop" onMouseDown={onClose}>
      <div
        className="insight-modal"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        ref={dialogRef}
        tabIndex={-1}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="insight-modal-head">
          <h2>{icon && <i className={icon} aria-hidden="true"></i>} {title}</h2>
          <button type="button" className="insight-modal-close" aria-label="Close" onClick={onClose}>
            <i className="fa-solid fa-xmark" aria-hidden="true"></i>
          </button>
        </div>
        <div className="insight-modal-body">{children}</div>
      </div>
    </div>
  )
}
