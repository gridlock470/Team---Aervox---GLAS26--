import { useEffect, useRef } from 'react'
import './InsightModal.css'

/**
 * Popup that a top-nav insight button opens. Replaces the old always-docked
 * bottom panel: the panel content only exists on screen while an operator
 * has actually asked to see it, so the map owns the rest of the viewport.
 *
 * Carries its own tab strip (reusing the same `tabs` array App.jsx already
 * builds for TopNav) so switching between the 7 insight panels no longer
 * means closing this modal and reopening a different one from TopNav --
 * standard W3C ARIA APG tabs pattern, one tabpanel whose content swaps.
 */
export default function InsightModal({ open, tabs, activeId, onSelect, onClose, children }) {
  const dialogRef = useRef(null)
  const tabRefs = useRef({})
  const activeTab = tabs?.find((t) => t.id === activeId) ?? null

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

  // Roving-tabindex arrow-key navigation (automatic activation): Left/Right
  // move and select the adjacent tab, Home/End jump to the ends.
  function onTabKeyDown(e, index) {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return
    e.preventDefault()
    let nextIndex = index
    if (e.key === 'ArrowLeft') nextIndex = (index - 1 + tabs.length) % tabs.length
    if (e.key === 'ArrowRight') nextIndex = (index + 1) % tabs.length
    if (e.key === 'Home') nextIndex = 0
    if (e.key === 'End') nextIndex = tabs.length - 1
    const next = tabs[nextIndex]
    onSelect(next.id)
    tabRefs.current[next.id]?.focus()
  }

  return (
    <div className="insight-modal-backdrop" onMouseDown={onClose}>
      <div
        className="insight-modal"
        role="dialog"
        aria-modal="true"
        aria-label={activeTab?.label}
        ref={dialogRef}
        tabIndex={-1}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="insight-modal-head">
          <h2>{activeTab?.icon && <i className={activeTab.icon} aria-hidden="true"></i>} {activeTab?.label}</h2>
          <button type="button" className="insight-modal-close" aria-label="Close" onClick={onClose}>
            <i className="fa-solid fa-xmark" aria-hidden="true"></i>
          </button>
        </div>
        <div className="insight-modal-tabs" role="tablist" aria-label="Insight panels">
          {tabs?.map((tab, index) => {
            const isActive = tab.id === activeId
            const badge = Number(tab.badge)
            const showBadge = Number.isFinite(badge) && badge > 0
            return (
              <button
                key={tab.id}
                type="button"
                role="tab"
                id={`insight-tab-${tab.id}`}
                aria-selected={isActive}
                aria-controls="insight-panel"
                tabIndex={isActive ? 0 : -1}
                className={isActive ? 'insight-tab active' : 'insight-tab'}
                title={tab.label}
                ref={(el) => { tabRefs.current[tab.id] = el }}
                onClick={() => onSelect(tab.id)}
                onKeyDown={(e) => onTabKeyDown(e, index)}
              >
                {tab.icon && <i className={tab.icon} aria-hidden="true"></i>}
                <span className="sr-only">{tab.label}</span>
                {showBadge && <span className="insight-tab-badge mono">{badge}</span>}
              </button>
            )
          })}
        </div>
        <div
          className="insight-modal-body"
          role="tabpanel"
          id="insight-panel"
          aria-labelledby={activeTab ? `insight-tab-${activeTab.id}` : undefined}
        >
          {children}
        </div>
      </div>
    </div>
  )
}
