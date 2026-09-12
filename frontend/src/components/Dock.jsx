import { useEffect, useId, useRef, useState } from 'react'
import './Dock.css'

/**
 * Right-hand tabbed dock for the console's data panels.
 *
 * Props:
 *   tabs            [{ id, label, badge? }]  badge renders only when > 0
 *   active          id of the selected tab
 *   onActiveChange  (id) => void
 *   collapsed       bool — shrinks the dock to a 44px vertical strip
 *   onCollapsedChange (bool) => void
 *   children        the already-selected panel element
 */
export default function Dock({
  tabs = [],
  active,
  onActiveChange,
  collapsed = false,
  onCollapsedChange,
  children,
}) {
  const uid = useId()
  const tabRefs = useRef({})
  const [focusedId, setFocusedId] = useState(active)

  // Re-anchor the roving tabindex whenever the owner changes the selection.
  useEffect(() => {
    setFocusedId(active)
  }, [active])

  const hasTabs = tabs.length > 0
  const activeTab = tabs.find((t) => t.id === active) || (hasTabs ? tabs[0] : null)
  const activeId = activeTab ? activeTab.id : null
  const rovingId = tabs.some((t) => t.id === focusedId) ? focusedId : activeId

  const panelDomId = `${uid}-panel`
  const tabDomId = (id) => `${uid}-tab-${id}`

  function select(id) {
    if (collapsed && onCollapsedChange) onCollapsedChange(false)
    if (id !== active && onActiveChange) onActiveChange(id)
  }

  function moveFocus(index) {
    if (!hasTabs) return
    const wrapped = (index + tabs.length) % tabs.length
    const id = tabs[wrapped].id
    setFocusedId(id)
    const el = tabRefs.current[id]
    if (el) el.focus()
  }

  function handleKeyDown(event) {
    if (!hasTabs) return
    const found = tabs.findIndex((t) => t.id === rovingId)
    const current = found < 0 ? 0 : found

    switch (event.key) {
      case 'ArrowRight':
      case 'ArrowDown':
        moveFocus(current + 1)
        break
      case 'ArrowLeft':
      case 'ArrowUp':
        moveFocus(current - 1)
        break
      case 'Home':
        moveFocus(0)
        break
      case 'End':
        moveFocus(tabs.length - 1)
        break
      default:
        // Enter / Space fall through to the native button click.
        return
    }
    event.preventDefault()
  }

  return (
    <aside className={collapsed ? 'dock is-collapsed' : 'dock'} data-collapsed={collapsed}>
      <div className="dock-strip">
        <div
          className="dock-tabs"
          role="tablist"
          aria-label="Console panels"
          aria-orientation={collapsed ? 'vertical' : 'horizontal'}
          onKeyDown={handleKeyDown}
        >
          {tabs.map((tab) => {
            const isActive = tab.id === activeId
            const badge = Number(tab.badge)
            const showBadge = Number.isFinite(badge) && badge > 0
            return (
              <button
                key={tab.id}
                type="button"
                role="tab"
                id={tabDomId(tab.id)}
                className="dock-tab"
                aria-selected={isActive}
                aria-controls={panelDomId}
                tabIndex={tab.id === rovingId ? 0 : -1}
                ref={(el) => {
                  if (el) tabRefs.current[tab.id] = el
                  else delete tabRefs.current[tab.id]
                }}
                onFocus={() => setFocusedId(tab.id)}
                onClick={() => select(tab.id)}
              >
                {tab.icon && <i className={tab.icon} aria-hidden="true"></i>}
                <span className="dock-tab-label">{tab.label}</span>
                {showBadge && (
                  <>
                    <span className="dock-badge mono" aria-hidden="true">
                      {badge}
                    </span>
                    <span className="dock-vh">, {badge} active</span>
                  </>
                )}
              </button>
            )
          })}
        </div>

        <button
          type="button"
          className="dock-collapse"
          aria-label={collapsed ? 'Expand panel dock' : 'Collapse panel dock'}
          aria-expanded={!collapsed}
          onClick={() => onCollapsedChange && onCollapsedChange(!collapsed)}
        >
          <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false">
            <path
              d={collapsed ? 'M10 3 L5 8 L10 13' : 'M6 3 L11 8 L6 13'}
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </div>

      <div
        className="dock-body"
        role="tabpanel"
        id={panelDomId}
        aria-labelledby={activeId ? tabDomId(activeId) : undefined}
        tabIndex={0}
        hidden={collapsed}
      >
        {children}
      </div>
    </aside>
  )
}
