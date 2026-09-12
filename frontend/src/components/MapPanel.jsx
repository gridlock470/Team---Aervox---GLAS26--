import { useEffect, useRef, useState } from 'react'
import 'maplibre-gl/dist/maplibre-gl.css'
import maplibregl from 'maplibre-gl'
import { MapboxOverlay } from '@deck.gl/mapbox'
import { ScatterplotLayer } from '@deck.gl/layers'
import './MapPanel.css'
import { DATA, HAZARDS, timeAt } from '../data/nowcastData.js'
import { SEV, sevFor, sevRgba, radiusMeters } from '../lib/severity.js'

export default function MapPanel({ region, hazard, step }) {
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const overlayRef = useRef(null)
  const prevRegionRef = useRef(null)
  const [hoverInfo, setHoverInfo] = useState(null)
  const [mapLoaded, setMapLoaded] = useState(false)

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

    const layer = new ScatterplotLayer({
      id: 'hazard-cells',
      data: points,
      getPosition: (d) => [d.lng, d.lat],
      getFillColor: (d) => sevRgba(d.pct),
      getRadius: (d) => radiusMeters(d.pct),
      pickable: true,
      stroked: false,
      updateTriggers: {
        getFillColor: [region, hazard, step],
        getRadius: [region, hazard, step],
      },
      onHover: (info) => {
        setHoverInfo(info && info.object ? info : null)
      },
    })

    overlayRef.current.setProps({ layers: [layer] })

    if (prevRegionRef.current !== region) {
      prevRegionRef.current = region
      const stations = Object.values(regionData.stations)
      if (stations.length && mapRef.current) {
        const bounds = stations.reduce(
          (b, st) => b.extend([st.lng, st.lat]),
          new maplibregl.LngLatBounds([stations[0].lng, stations[0].lat], [stations[0].lng, stations[0].lat])
        )
        mapRef.current.fitBounds(bounds, { padding: 60, duration: 800 })
      }
    }
  }, [mapLoaded, region, hazard, step])

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
        <div className="map-canvas" ref={containerRef} role="img" aria-label={`${hazardName} probability map — ${regionData.title}`} />
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
