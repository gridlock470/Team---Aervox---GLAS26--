import { useEffect, useRef, useState } from 'react'
import 'maplibre-gl/dist/maplibre-gl.css'
import maplibregl from 'maplibre-gl'
import { MapboxOverlay } from '@deck.gl/mapbox'
import { ScatterplotLayer, BitmapLayer } from '@deck.gl/layers'
import './MapPanel.css'
import { DATA, HAZARDS, timeAt } from '../data/nowcastData.js'
import { SEV, sevFor, sevRgba, radiusMeters } from '../lib/severity.js'
import dgmrNowcast from '../data/dgmrNowcast.json'

// Matches the transform duration driven on .stage in App.jsx -- kept in one
// place so the post-transition map.resize() timeout cannot drift out of sync
// with the CSS transition it is waiting on.
export const FULLSCREEN_TRANSITION_MS = 320

export default function MapPanel({ region, hazard, step, fullscreen, onToggleFullscreen }) {
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const overlayRef = useRef(null)
  const prevRegionRef = useRef(null)
  const [hoverInfo, setHoverInfo] = useState(null)
  const [mapLoaded, setMapLoaded] = useState(false)
  const [showAiForecast, setShowAiForecast] = useState(false)

  // Only generated for one region so far (see scripts/make_dgmr_nowcast.py) --
  // the toggle only appears where there is actually a frame to show.
  const aiAvailable = dgmrNowcast?.region === region
  const aiFrame = aiAvailable ? dgmrNowcast.frames[Math.min(step, dgmrNowcast.frames.length - 1)] : null

  // Mount the map once.
  useEffect(() => {
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: 'https://tiles.openfreemap.org/styles/liberty',
      center: [78.5, 30.1],
      zoom: 7,
    })
    mapRef.current = map

    map.addControl(new maplibregl.NavigationControl(), 'top-right')

    map.on('load', () => {
      // deck.gl's MapboxOverlay must be added after the base map has finished
      // loading its style/sources — adding it earlier (or with interleaved:true
      // before load) stalls MapLibre's internal tile-update cycle entirely.
      const overlay = new MapboxOverlay({ layers: [] })
      map.addControl(overlay)
      overlayRef.current = overlay
      setMapLoaded(true)
    })
    map.on('error', (e) => {
      console.error('MapLibre error:', e && e.error ? e.error.message : e)
    })

    return () => {
      map.remove()
      mapRef.current = null
      overlayRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Wire the fullscreen toggle to MapLibre's own click event, not a raw DOM
  // listener on the container -- MapLibre only fires `click` for a genuine
  // click, never mid-drag, so this can't be triggered by someone panning the
  // map. Bound in its own effect (rather than inside the mount effect) so it
  // always closes over the latest onToggleFullscreen without re-mounting the
  // map.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !onToggleFullscreen) return
    const handler = () => onToggleFullscreen()
    map.on('click', handler)
    return () => map.off('click', handler)
  }, [mapLoaded, onToggleFullscreen])

  // The container's box changes size when the fullscreen transform-transition
  // (driven by the parent on .stage) finishes -- MapLibre renders into a
  // canvas sized at mount/last-resize, so it has to be told to re-measure or
  // it stays cropped to the old box. A timeout matching the CSS transition
  // duration is the simplest reliable hook for "transition finished" here,
  // since the transition lives on an ancestor this component doesn't own.
  useEffect(() => {
    if (!mapRef.current) return
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const delay = reduceMotion ? 0 : FULLSCREEN_TRANSITION_MS + 20
    const t = setTimeout(() => mapRef.current?.resize(), delay)
    return () => clearTimeout(t)
  }, [fullscreen])

  // Fits the view to the currently selected region's stations -- the same
  // region the header's search box is showing. Used both to auto-recenter
  // on a region change and by the recenter button, so a viewer who has
  // panned/zoomed away can always get back to "where the search box says
  // I am" without re-picking the region.
  function recenterToRegion() {
    const map = mapRef.current
    if (!map) return
    const stations = Object.values(DATA[region].stations)
    if (!stations.length) return
    const bounds = stations.reduce(
      (b, st) => b.extend([st.lng, st.lat]),
      new maplibregl.LngLatBounds([stations[0].lng, stations[0].lat], [stations[0].lng, stations[0].lat])
    )
    map.fitBounds(bounds, { padding: 60, duration: 800 })
  }

  // Rebuild deck.gl layers + fly-to-region whenever region/hazard/step change
  // (and once the map finishes its initial load).
  useEffect(() => {
    if (!mapLoaded || !overlayRef.current) return

    const regionData = DATA[region]
    const hazardVals = regionData.hazards[hazard].vals

    const points = Object.entries(regionData.stations).map(([id, st]) => ({
      id,
      ...st,
      pct: hazardVals[id][step],
    }))

    // A single flat filled circle with a solid dark ring reads as a map-pin
    // sticker -- wrong register for a control-room console whose whole visual
    // language is glow, not outline (the header's live-dot, the severity
    // badges' box-shadows). Two layers instead: a soft, larger, low-opacity
    // halo underneath for a glow that survives zooming out, and a smaller
    // core on top with a faint light rim (not black) for edge definition --
    // the same "coloured light against near-black" read as everywhere else
    // in the theme.
    const haloLayer = new ScatterplotLayer({
      id: 'hazard-cells-halo',
      data: points,
      getPosition: (d) => [d.lng, d.lat],
      getFillColor: (d) => sevRgba(d.pct, 0.16),
      getRadius: (d) => radiusMeters(d.pct) * 1.9,
      radiusMinPixels: 12,
      stroked: false,
      pickable: false,
      updateTriggers: {
        getFillColor: [region, hazard, step],
        getRadius: [region, hazard, step],
      },
    })

    const coreLayer = new ScatterplotLayer({
      id: 'hazard-cells',
      data: points,
      getPosition: (d) => [d.lng, d.lat],
      getFillColor: (d) => sevRgba(d.pct, 0.82),
      getRadius: (d) => radiusMeters(d.pct),
      // radiusMeters is a real-world radius, so a cell shrinks to a few
      // screen pixels (or less) at a zoomed-out view -- radiusMinPixels
      // keeps the core legible on its own even where the halo has faded out.
      radiusMinPixels: 5,
      stroked: true,
      getLineColor: [249, 250, 251, 90],
      lineWidthMinPixels: 1,
      pickable: true,
      updateTriggers: {
        getFillColor: [region, hazard, step],
        getRadius: [region, hazard, step],
      },
      onHover: (info) => {
        setHoverInfo(info && info.object ? info : null)
      },
    })

    // Demo overlay from a real pretrained model (scripts/make_dgmr_nowcast.py),
    // not a placeholder -- but drawn under the hazard dots and off by default,
    // since it is an integration demo rather than a validated forecast (see
    // dgmrNowcast.json's own disclaimer, surfaced in the caption below).
    const aiLayer = showAiForecast && aiFrame
      ? new BitmapLayer({
          id: 'ai-forecast-bitmap',
          image: aiFrame.file,
          bounds: dgmrNowcast.bbox,
          opacity: 0.55,
        })
      : null

    overlayRef.current.setProps({ layers: [aiLayer, haloLayer, coreLayer].filter(Boolean) })

    if (prevRegionRef.current !== region) {
      prevRegionRef.current = region
      recenterToRegion()
    }
  }, [mapLoaded, region, hazard, step, showAiForecast])

  const regionData = DATA[region]
  const hazardName = HAZARDS.find((h) => h.id === hazard)?.name ?? hazard

  return (
    <section className="hero panel">
      <div className="panel-head">
        <div>
          <h2>{hazardName} — {regionData.title}</h2>
          <p>{regionData.subtitle}</p>
        </div>
        <div className="valid-at">Valid at<br /><span className="mono">{timeAt(step)}</span></div>
      </div>

      <div className="map-wrap">
        <div className="map-overlay-title">
          <span className="pulse-dot"></span>
          <span>LIVE RADAR &amp; HAZARD ZONE MATRIX</span>
        </div>
        {fullscreen && (
          <button
            type="button"
            className="map-fullscreen-exit"
            onClick={(e) => { e.stopPropagation(); onToggleFullscreen?.() }}
            title="Exit fullscreen (Esc)"
          >
            <i className="fa-solid fa-compress" aria-hidden="true"></i> Exit fullscreen
          </button>
        )}
        <button
          type="button"
          className="map-recenter"
          onClick={(e) => { e.stopPropagation(); recenterToRegion() }}
          title={`Recenter on ${regionData.title}`}
        >
          <i className="fa-solid fa-location-crosshairs" aria-hidden="true"></i>
        </button>
        {aiAvailable && (
          <button
            type="button"
            className={showAiForecast ? 'map-ai-toggle active' : 'map-ai-toggle'}
            onClick={(e) => { e.stopPropagation(); setShowAiForecast((v) => !v) }}
            title="Toggle AI-generated forecast overlay (demo, experimental)"
          >
            <i className="fa-solid fa-wand-magic-sparkles" aria-hidden="true"></i> AI Forecast (beta)
          </button>
        )}
        <div
          className="map-canvas"
          ref={containerRef}
          role="img"
          aria-label={`${hazardName} probability map — ${regionData.title}${fullscreen ? ' (fullscreen)' : ''}`}
        />
        {hoverInfo && hoverInfo.object && (
          <div
            className="map-tooltip"
            style={{ left: hoverInfo.x, top: hoverInfo.y }}
          >
            <div className="map-tooltip-name">{hoverInfo.object.name}</div>
            <div className="map-tooltip-pct mono">
              {hoverInfo.object.pct}% &middot; {SEV[sevFor(hoverInfo.object.pct)].label}
            </div>
          </div>
        )}
        {showAiForecast && aiFrame && (
          <div className="map-ai-caption">
            <strong>{aiFrame.label}</strong> &mdash; {dgmrNowcast.model.name}. {dgmrNowcast.disclaimer}
          </div>
        )}
      </div>

      <div className="map-legend">
        <span className="legend-item"><span className="legend-swatch" style={{ background: SEV.green.hex }}></span>Green: No warning</span>
        <span className="legend-item"><span className="legend-swatch" style={{ background: SEV.yellow.hex }}></span>Yellow: Watch</span>
        <span className="legend-item"><span className="legend-swatch" style={{ background: SEV.orange.hex }}></span>Orange: Alert</span>
        <span className="legend-item"><span className="legend-swatch" style={{ background: SEV.red.hex }}></span>Red: Warning</span>
      </div>
    </section>
  )
}
