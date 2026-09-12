import liveNowcast from './liveNowcast.json';

export const HAZARDS = [
  { id: 'thunderstorm', name: 'Severe thunderstorm' },
  { id: 'cloudburst',   name: 'Cloudburst' },
  { id: 'flashflood',   name: 'Flash flood' },
];

export const STEPS = ['Now', '+2h', '+4h', '+6h'];
const BASE_HOUR = 7, BASE_MIN = 42;

export function timeAt(stepIndex) {
  const total = BASE_HOUR * 60 + BASE_MIN + stepIndex * 120;
  const h = Math.floor(total / 60) % 24, m = total % 60;
  return `${h < 10 ? '0' : ''}${h}:${m < 10 ? '0' : ''}${m} IST`;
}

export const REGION_META = [
  { id: 'uk',  label: 'Uttarakhand' },
  { id: 'ncr', label: 'Delhi NCR' },
];

export const DATA = {
  uk: {
    title: 'Uttarakhand',
    subtitle: 'Rudraprayag–Dehradun corridor',
    stations: {
      dhauliganga: { name: 'Upper Dhauliganga basin', lat: 31.2000, lng: 79.4000 },
      kedarnath:   { name: 'Kedarnath approach', lat: 30.7346, lng: 79.0669 },
      rudraprayag: { name: 'Rudraprayag',        lat: 30.2844, lng: 78.9811 },
      dehradun:    { name: 'Dehradun',           lat: 30.3165, lng: 78.0322 },
      haridwar:    { name: 'Haridwar',           lat: 29.9457, lng: 78.1642 },
    },
    hazards: {
      thunderstorm: {
        vals: { kedarnath:[34,44,53,46], rudraprayag:[30,41,50,43], dehradun:[38,50,61,55], haridwar:[24,32,40,35] },
        drivers: [
          { label:'CAPE (surface-based)', unit:'J/kg', vals:['1120','1480','1810','1640'], w:[0.62,0.74,0.9,0.82] },
          { label:'Vertical wind shear, 0–6 km', unit:'m/s', vals:['11','14','16','15'], w:[0.5,0.62,0.72,0.66] },
          { label:'Cloud-top temperature drop-rate', unit:'°C/15min', vals:['-2.1','-3.4','-4.6','-3.9'], w:[0.4,0.58,0.76,0.62] },
          { label:'Low-level convergence', unit:'×10⁻⁵ s⁻¹', vals:['3.2','4.5','5.8','5.1'], w:[0.35,0.5,0.66,0.55] },
          { label:'Integrated water vapour, rate of change', unit:'kg/m²/hr', vals:['1.8','2.6','3.3','2.9'], w:[0.28,0.4,0.52,0.45] },
        ],
      },
      cloudburst: {
        vals: { kedarnath:[46,58,72,66], rudraprayag:[41,53,68,60], dehradun:[26,34,44,39], haridwar:[18,24,31,27] },
        drivers: [
          { label:'Integrated water vapour, rate of change', unit:'kg/m²/hr', vals:['3.4','4.6','6.1','5.4'], w:[0.55,0.7,0.92,0.8] },
          { label:'Cloud-top temperature drop-rate', unit:'°C/15min', vals:['-2.8','-4.1','-5.7','-4.9'], w:[0.5,0.66,0.88,0.75] },
          { label:'Low-level convergence', unit:'×10⁻⁵ s⁻¹', vals:['4.0','5.4','7.0','6.1'], w:[0.42,0.56,0.74,0.64] },
          { label:'CAPE (surface-based)', unit:'J/kg', vals:['980','1260','1520','1390'], w:[0.3,0.42,0.55,0.47] },
          { label:'Vertical wind shear, 0–6 km', unit:'m/s', vals:['7','8','10','9'], w:[0.18,0.24,0.32,0.27] },
        ],
      },
      flashflood: {
        vals: { kedarnath:[52,64,79,74], rudraprayag:[48,61,77,71], dehradun:[30,39,50,45], haridwar:[33,43,55,49] },
        drivers: [
          { label:'Integrated water vapour, rate of change', unit:'kg/m²/hr', vals:['3.1','4.3','5.8','5.2'], w:[0.5,0.66,0.88,0.78] },
          { label:'CartoDEM slope-routed runoff index', unit:'index', vals:['0.41','0.56','0.74','0.68'], w:[0.6,0.72,0.94,0.86] },
          { label:'Low-level convergence', unit:'×10⁻⁵ s⁻¹', vals:['3.8','5.1','6.7','5.9'], w:[0.4,0.54,0.7,0.61] },
          { label:'Cloud-top temperature drop-rate', unit:'°C/15min', vals:['-2.4','-3.6','-5.0','-4.3'], w:[0.32,0.46,0.62,0.53] },
          { label:'CAPE (surface-based)', unit:'J/kg', vals:['860','1040','1240','1150'], w:[0.2,0.28,0.37,0.32] },
        ],
      },
    },
    alerts: [
      { sev:'orange', id:'IN-UK-20260910-0742', headline:'Cloudburst risk building over Kedarnath approach', area:'Rudraprayag district, upper Alaknanda basin', window:'14:00–20:00 IST', sent:'07:42 IST' },
      { sev:'yellow', id:'IN-UK-20260910-0730', headline:'Isolated severe thunderstorm cells expected', area:'Dehradun valley', window:'13:00–17:00 IST', sent:'07:30 IST' },
    ],
  },
  ncr: {
    title: 'Delhi NCR',
    subtitle: 'Yamuna floodplain & western drainage basins',
    stations: {
      gurugram:   { name: 'Gurugram',                      lat: 28.4595, lng: 77.0266 },
      najafgarh:  { name: 'Najafgarh drain basin',          lat: 28.6092, lng: 76.9797 },
      southdelhi: { name: 'South Delhi, Yamuna floodplain', lat: 28.5355, lng: 77.2933 },
      noida:      { name: 'Noida',                          lat: 28.5709, lng: 77.3260 },
    },
    hazards: {
      thunderstorm: {
        vals: { gurugram:[30,40,49,43], najafgarh:[28,37,46,40], southdelhi:[33,44,54,47], noida:[27,36,45,39] },
        drivers: [
          { label:'CAPE (surface-based)', unit:'J/kg', vals:['1240','1560','1880','1720'], w:[0.6,0.72,0.9,0.8] },
          { label:'Vertical wind shear, 0–6 km', unit:'m/s', vals:['10','13','15','14'], w:[0.46,0.58,0.68,0.62] },
          { label:'Cloud-top temperature drop-rate', unit:'°C/15min', vals:['-1.9','-3.0','-4.2','-3.5'], w:[0.36,0.52,0.7,0.58] },
          { label:'Low-level convergence', unit:'×10⁻⁵ s⁻¹', vals:['2.8','3.9','5.0','4.4'], w:[0.3,0.44,0.58,0.5] },
          { label:'Integrated water vapour, rate of change', unit:'kg/m²/hr', vals:['1.5','2.2','2.9','2.5'], w:[0.24,0.35,0.46,0.39] },
        ],
      },
      cloudburst: {
        vals: { gurugram:[24,32,41,36], najafgarh:[35,46,58,51], southdelhi:[38,50,63,56], noida:[30,40,50,44] },
        drivers: [
          { label:'Integrated water vapour, rate of change', unit:'kg/m²/hr', vals:['2.6','3.7','4.9','4.3'], w:[0.46,0.6,0.8,0.7] },
          { label:'Cloud-top temperature drop-rate', unit:'°C/15min', vals:['-2.2','-3.3','-4.6','-3.9'], w:[0.4,0.54,0.72,0.62] },
          { label:'Low-level convergence', unit:'×10⁻⁵ s⁻¹', vals:['3.4','4.6','5.9','5.2'], w:[0.36,0.48,0.64,0.55] },
          { label:'CAPE (surface-based)', unit:'J/kg', vals:['1040','1290','1540','1410'], w:[0.28,0.38,0.5,0.43] },
          { label:'Vertical wind shear, 0–6 km', unit:'m/s', vals:['8','9','11','10'], w:[0.16,0.22,0.3,0.25] },
        ],
      },
      flashflood: {
        vals: { gurugram:[26,34,44,38], najafgarh:[44,56,70,64], southdelhi:[41,53,67,60], noida:[32,42,53,47] },
        drivers: [
          { label:'CartoDEM slope-routed runoff index', unit:'index', vals:['0.38','0.52','0.69','0.63'], w:[0.56,0.68,0.9,0.82] },
          { label:'Integrated water vapour, rate of change', unit:'kg/m²/hr', vals:['2.4','3.4','4.5','3.9'], w:[0.44,0.58,0.76,0.66] },
          { label:'Low-level convergence', unit:'×10⁻⁵ s⁻¹', vals:['3.1','4.2','5.5','4.8'], w:[0.34,0.46,0.6,0.52] },
          { label:'Cloud-top temperature drop-rate', unit:'°C/15min', vals:['-2.0','-3.0','-4.1','-3.5'], w:[0.28,0.4,0.54,0.46] },
          { label:'CAPE (surface-based)', unit:'J/kg', vals:['760','920','1090','1010'], w:[0.16,0.22,0.3,0.26] },
        ],
      },
    },
    alerts: [
      { sev:'orange', id:'IN-DL-20260910-0742', headline:'Urban flash flood risk rising, Najafgarh basin', area:'South West Delhi, Najafgarh drain catchment', window:'13:00–19:00 IST', sent:'07:42 IST' },
      { sev:'yellow', id:'IN-DL-20260910-0715', headline:'Waterlogging likely along Yamuna floodplain roads', area:'South Delhi', window:'12:00–16:00 IST', sent:'07:15 IST' },
    ],
  },
};

/* ---------- real model output ----------
   liveNowcast.json is written by scripts/make_live_nowcast.py: LightGBM
   boosters trained on 2018 ERA5 + IMERG + DEM, run over the feature cube at
   one timestamp. Probabilities are booster predictions at the grid cell
   nearest each station; driver values are the actual feature-channel values
   there; driver weights are the booster's own gain importances.

   The overlay is applied per hazard, so anything the export does not cover
   keeps the illustrative values it had -- and MODEL_META records which is
   which, rather than leaving the two indistinguishable on screen. */
let live = null;
try {
  live = liveNowcast;
} catch {
  live = null;
}

/* The pipeline names hazards with underscores (config.HAZARDS); the console
   uses the compact form. Mapping them here rather than renaming either side
   mid-demo. */
const HAZARD_ALIAS = { flash_flood: 'flashflood' };

if (live?.regions) {
  for (const [regionId, region] of Object.entries(live.regions)) {
    const target = DATA[regionId];
    if (!target) continue;
    for (const [rawId, hazard] of Object.entries(region.hazards ?? {})) {
      const slot = target.hazards[HAZARD_ALIAS[rawId] ?? rawId];
      if (!slot) continue;
      if (hazard.vals) slot.vals = { ...slot.vals, ...hazard.vals };
      if (hazard.drivers?.length) slot.drivers = hazard.drivers;
    }
  }

  /* Every station must have a series for every hazard. A station present in
     the map but missing from one hazard's vals would be read as undefined[step]
     and take the whole panel down. */
  for (const region of Object.values(DATA)) {
    const stationIds = Object.keys(region.stations);
    for (const hazard of Object.values(region.hazards)) {
      for (const id of stationIds) {
        if (!Array.isArray(hazard.vals[id])) {
          hazard.vals[id] = STEPS.map(() => 0);
        }
      }
    }
  }
}

export const MODEL_META = {
  live: Boolean(live?.regions),
  source: live?.generated_from ?? null,
  validAt: live?.valid_at ?? null,
  validAtLabel: live?.valid_at_label ?? null,
  heldOut: Boolean(live?.held_out),
};

export function regionPeak(regionId, hazardId, step) {
  const vals = DATA[regionId].hazards[hazardId].vals;
  return Object.keys(vals).reduce((max, k) => Math.max(max, vals[k][step]), 0);
}
