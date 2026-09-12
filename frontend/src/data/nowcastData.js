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
  { id: 'uk',   label: 'Uttarakhand' },
  { id: 'ncr',  label: 'Delhi NCR' },
  { id: 'konk', label: 'Mumbai & Konkan Coast' },
  { id: 'meg',  label: 'Meghalaya & Assam' },
  { id: 'hp',   label: 'Himachal Pradesh' },
  { id: 'chn',  label: 'Chennai Metro' },
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
  konk: {
    title: 'Mumbai & Konkan Coast',
    subtitle: 'Western Ghats escarpment & Arabian Sea landfall belt',
    stations: {
      mumbai:     { name: 'Mumbai, Mithi River basin', lat: 19.0760, lng: 72.8777 },
      thane:      { name: 'Thane creek catchment',      lat: 19.2183, lng: 72.9781 },
      panvel:     { name: 'Panvel, Ghats foothills',    lat: 18.9894, lng: 73.1175 },
      mahabaleshwar: { name: 'Mahabaleshwar escarpment', lat: 17.9307, lng: 73.6477 },
      ratnagiri:  { name: 'Ratnagiri coast',             lat: 16.9902, lng: 73.3120 },
    },
    hazards: {
      thunderstorm: {
        vals: { mumbai:[27,37,47,41], thane:[31,42,53,46], panvel:[35,47,59,52], mahabaleshwar:[40,53,66,58], ratnagiri:[29,39,49,43] },
        drivers: [
          { label:'CAPE (surface-based)', unit:'J/kg', vals:['980','1290','1610','1450'], w:[0.5,0.64,0.8,0.7] },
          { label:'Vertical wind shear, 0–6 km', unit:'m/s', vals:['9','12','14','13'], w:[0.4,0.52,0.62,0.56] },
          { label:'Cloud-top temperature drop-rate', unit:'°C/15min', vals:['-1.7','-2.8','-3.9','-3.3'], w:[0.32,0.48,0.64,0.53] },
          { label:'Low-level convergence', unit:'×10⁻⁵ s⁻¹', vals:['2.6','3.6','4.7','4.1'], w:[0.26,0.38,0.5,0.43] },
          { label:'Integrated water vapour, rate of change', unit:'kg/m²/hr', vals:['2.2','3.1','4.0','3.5'], w:[0.3,0.44,0.58,0.5] },
        ],
      },
      cloudburst: {
        vals: { mumbai:[33,45,58,51], thane:[36,48,62,55], panvel:[44,58,73,67], mahabaleshwar:[58,72,88,81], ratnagiri:[47,61,76,70] },
        drivers: [
          { label:'Integrated water vapour, rate of change', unit:'kg/m²/hr', vals:['4.6','6.2','8.1','7.2'], w:[0.62,0.78,0.96,0.87] },
          { label:'Orographic uplift index, Ghats escarpment', unit:'index', vals:['0.52','0.68','0.87','0.79'], w:[0.58,0.72,0.92,0.83] },
          { label:'Cloud-top temperature drop-rate', unit:'°C/15min', vals:['-3.2','-4.5','-6.2','-5.4'], w:[0.48,0.63,0.84,0.72] },
          { label:'Low-level convergence', unit:'×10⁻⁵ s⁻¹', vals:['4.4','5.9','7.6','6.7'], w:[0.4,0.54,0.71,0.61] },
          { label:'CAPE (surface-based)', unit:'J/kg', vals:['820','1010','1240','1130'], w:[0.24,0.34,0.45,0.38] },
        ],
      },
      flashflood: {
        vals: { mumbai:[45,58,72,66], thane:[42,54,68,62], panvel:[50,64,79,73], mahabaleshwar:[38,49,61,54], ratnagiri:[40,52,65,58] },
        drivers: [
          { label:'Integrated water vapour, rate of change', unit:'kg/m²/hr', vals:['4.2','5.6','7.3','6.5'], w:[0.56,0.71,0.9,0.81] },
          { label:'CartoDEM slope-routed runoff index', unit:'index', vals:['0.47','0.62','0.81','0.74'], w:[0.64,0.78,0.96,0.88] },
          { label:'Tidal backwater stage, Mithi/Ulhas', unit:'m above MSL', vals:['1.1','1.6','2.2','1.9'], w:[0.44,0.58,0.75,0.65] },
          { label:'Low-level convergence', unit:'×10⁻⁵ s⁻¹', vals:['4.0','5.3','6.9','6.1'], w:[0.36,0.49,0.64,0.55] },
          { label:'Cloud-top temperature drop-rate', unit:'°C/15min', vals:['-2.6','-3.8','-5.2','-4.5'], w:[0.28,0.4,0.54,0.46] },
        ],
      },
    },
    alerts: [
      { sev:'red', id:'IN-MH-20260910-0742', headline:'Extreme rainfall, urban flash flood warning for Mithi River basin', area:'Mumbai city & suburbs', window:'12:00–20:00 IST', sent:'07:42 IST' },
      { sev:'orange', id:'IN-MH-20260910-0718', headline:'Cloudburst risk over Western Ghats escarpment', area:'Mahabaleshwar–Panvel corridor', window:'13:00–19:00 IST', sent:'07:18 IST' },
    ],
  },
  meg: {
    title: 'Meghalaya & Assam',
    subtitle: 'Shillong Plateau windward slopes & Brahmaputra floodplain',
    stations: {
      cherrapunji: { name: 'Cherrapunji (Sohra)', lat: 25.2702, lng: 91.7323 },
      mawsynram:   { name: 'Mawsynram',            lat: 25.2967, lng: 91.5822 },
      shillong:    { name: 'Shillong',             lat: 25.5788, lng: 91.8933 },
      guwahati:    { name: 'Guwahati, Brahmaputra bank', lat: 26.1445, lng: 91.7362 },
      silchar:     { name: 'Silchar, Barak valley', lat: 24.8333, lng: 92.7789 },
    },
    hazards: {
      thunderstorm: {
        vals: { cherrapunji:[36,48,60,53], mawsynram:[38,50,63,55], shillong:[32,43,54,47], guwahati:[29,39,49,43], silchar:[27,36,46,40] },
        drivers: [
          { label:'CAPE (surface-based)', unit:'J/kg', vals:['1340','1680','2020','1860'], w:[0.58,0.7,0.86,0.78] },
          { label:'Vertical wind shear, 0–6 km', unit:'m/s', vals:['8','10','12','11'], w:[0.34,0.44,0.54,0.48] },
          { label:'Cloud-top temperature drop-rate', unit:'°C/15min', vals:['-2.3','-3.5','-4.8','-4.1'], w:[0.42,0.58,0.76,0.64] },
          { label:'Low-level convergence', unit:'×10⁻⁵ s⁻¹', vals:['3.6','4.9','6.3','5.5'], w:[0.36,0.5,0.66,0.56] },
          { label:'Integrated water vapour, rate of change', unit:'kg/m²/hr', vals:['3.9','5.3','6.9','6.0'], w:[0.44,0.58,0.75,0.65] },
        ],
      },
      cloudburst: {
        vals: { cherrapunji:[62,77,92,86], mawsynram:[65,80,94,89], shillong:[50,64,79,72], guwahati:[38,50,63,56], silchar:[34,45,58,51] },
        drivers: [
          { label:'Integrated water vapour, rate of change', unit:'kg/m²/hr', vals:['6.8','8.9','11.4','10.2'], w:[0.74,0.88,0.98,0.94] },
          { label:'Orographic uplift index, windward slope', unit:'index', vals:['0.71','0.86','0.97','0.93'], w:[0.7,0.84,0.97,0.9] },
          { label:'Cloud-top temperature drop-rate', unit:'°C/15min', vals:['-4.1','-5.6','-7.4','-6.5'], w:[0.56,0.72,0.9,0.8] },
          { label:'Low-level convergence', unit:'×10⁻⁵ s⁻¹', vals:['5.2','6.8','8.7','7.7'], w:[0.46,0.6,0.78,0.68] },
          { label:'CAPE (surface-based)', unit:'J/kg', vals:['1180','1420','1690','1560'], w:[0.32,0.42,0.54,0.47] },
        ],
      },
      flashflood: {
        vals: { cherrapunji:[54,68,83,77], mawsynram:[57,71,86,80], shillong:[44,57,71,64], guwahati:[47,60,75,68], silchar:[49,63,78,71] },
        drivers: [
          { label:'Integrated water vapour, rate of change', unit:'kg/m²/hr', vals:['5.9','7.7','9.9','8.8'], w:[0.62,0.76,0.93,0.84] },
          { label:'CartoDEM slope-routed runoff index', unit:'index', vals:['0.58','0.74','0.92','0.85'], w:[0.68,0.8,0.97,0.9] },
          { label:'Brahmaputra stage, rate of rise', unit:'cm/hr', vals:['2.4','3.6','5.1','4.4'], w:[0.5,0.64,0.82,0.72] },
          { label:'Low-level convergence', unit:'×10⁻⁵ s⁻¹', vals:['4.6','6.0','7.7','6.8'], w:[0.4,0.54,0.7,0.6] },
          { label:'Cloud-top temperature drop-rate', unit:'°C/15min', vals:['-3.0','-4.3','-5.9','-5.1'], w:[0.3,0.42,0.56,0.48] },
        ],
      },
    },
    alerts: [
      { sev:'red', id:'IN-ML-20260910-0742', headline:'Extreme cloudburst risk, Mawsynram–Cherrapunji windward belt', area:'East Khasi Hills district', window:'11:00–21:00 IST', sent:'07:42 IST' },
      { sev:'orange', id:'IN-AS-20260910-0705', headline:'Brahmaputra flash flood risk rising, Guwahati bank', area:'Kamrup Metropolitan district', window:'14:00–22:00 IST', sent:'07:05 IST' },
    ],
  },
  hp: {
    title: 'Himachal Pradesh',
    subtitle: 'Kullu–Manali & Beas basin cloudburst belt',
    stations: {
      manali:   { name: 'Manali',                lat: 32.2432, lng: 77.1892 },
      kullu:    { name: 'Kullu, Beas valley',     lat: 31.9576, lng: 77.1095 },
      mandi:    { name: 'Mandi',                  lat: 31.7080, lng: 76.9318 },
      shimla:   { name: 'Shimla',                 lat: 31.1048, lng: 77.1734 },
      kinnaur:  { name: 'Kinnaur, Sutlej gorge',  lat: 31.5900, lng: 78.2900 },
    },
    hazards: {
      thunderstorm: {
        vals: { manali:[31,42,52,45], kullu:[29,39,49,43], mandi:[33,44,55,48], shimla:[27,37,46,40], kinnaur:[35,47,58,51] },
        drivers: [
          { label:'CAPE (surface-based)', unit:'J/kg', vals:['960','1240','1520','1390'], w:[0.48,0.6,0.76,0.66] },
          { label:'Vertical wind shear, 0–6 km', unit:'m/s', vals:['13','16','19','17'], w:[0.52,0.64,0.76,0.68] },
          { label:'Cloud-top temperature drop-rate', unit:'°C/15min', vals:['-2.0','-3.2','-4.4','-3.7'], w:[0.38,0.54,0.7,0.6] },
          { label:'Low-level convergence', unit:'×10⁻⁵ s⁻¹', vals:['3.0','4.2','5.4','4.7'], w:[0.32,0.46,0.6,0.51] },
          { label:'Integrated water vapour, rate of change', unit:'kg/m²/hr', vals:['1.9','2.7','3.5','3.0'], w:[0.26,0.37,0.49,0.42] },
        ],
      },
      cloudburst: {
        vals: { manali:[57,71,87,81], kullu:[54,68,84,78], mandi:[45,58,73,66], shimla:[36,47,60,53], kinnaur:[61,76,91,86] },
        drivers: [
          { label:'Integrated water vapour, rate of change', unit:'kg/m²/hr', vals:['4.4','6.0','7.9','6.9'], w:[0.6,0.75,0.95,0.85] },
          { label:'Orographic uplift index, valley headwall', unit:'index', vals:['0.64','0.80','0.96','0.89'], w:[0.66,0.8,0.97,0.88] },
          { label:'Cloud-top temperature drop-rate', unit:'°C/15min', vals:['-3.5','-4.9','-6.7','-5.8'], w:[0.52,0.68,0.88,0.76] },
          { label:'Low-level convergence', unit:'×10⁻⁵ s⁻¹', vals:['4.6','6.1','7.9','6.9'], w:[0.44,0.58,0.76,0.66] },
          { label:'CAPE (surface-based)', unit:'J/kg', vals:['860','1050','1280','1170'], w:[0.26,0.36,0.48,0.4] },
        ],
      },
      flashflood: {
        vals: { manali:[62,77,93,88], kullu:[59,74,90,84], mandi:[48,62,78,71], shimla:[38,49,62,55], kinnaur:[65,80,96,91] },
        drivers: [
          { label:'Integrated water vapour, rate of change', unit:'kg/m²/hr', vals:['4.1','5.6','7.4','6.5'], w:[0.58,0.72,0.92,0.82] },
          { label:'CartoDEM slope-routed runoff index', unit:'index', vals:['0.62','0.78','0.95','0.88'], w:[0.72,0.85,0.98,0.92] },
          { label:'Glacial-lake outburst susceptibility, Beas/Sutlej', unit:'index', vals:['0.30','0.38','0.49','0.44'], w:[0.4,0.5,0.64,0.56] },
          { label:'Low-level convergence', unit:'×10⁻⁵ s⁻¹', vals:['4.2','5.6','7.2','6.3'], w:[0.38,0.52,0.68,0.58] },
          { label:'Cloud-top temperature drop-rate', unit:'°C/15min', vals:['-2.9','-4.2','-5.8','-5.0'], w:[0.3,0.42,0.57,0.48] },
        ],
      },
    },
    alerts: [
      { sev:'red', id:'IN-HP-20260910-0742', headline:'Cloudburst warning, Kinnaur–Manali headwall', area:'Kullu & Kinnaur districts, upper Beas/Sutlej', window:'13:00–21:00 IST', sent:'07:42 IST' },
      { sev:'orange', id:'IN-HP-20260910-0722', headline:'Flash flood risk along Beas valley approach roads', area:'Kullu–Manali corridor', window:'14:00–19:00 IST', sent:'07:22 IST' },
    ],
  },
  chn: {
    title: 'Chennai Metro',
    subtitle: 'Adyar–Cooum basin & northeast monsoon coast',
    stations: {
      chennai:    { name: 'Chennai, Cooum basin',   lat: 13.0827, lng: 80.2707 },
      tambaram:   { name: 'Tambaram, Adyar upper reach', lat: 12.9249, lng: 80.1000 },
      velachery:  { name: 'Velachery, Pallikaranai marsh', lat: 12.9791, lng: 80.2213 },
      avadi:      { name: 'Avadi, Cooum headwaters', lat: 13.1147, lng: 80.0970 },
      mahabalipuram: { name: 'Mahabalipuram coast', lat: 12.6269, lng: 80.1927 },
    },
    hazards: {
      thunderstorm: {
        vals: { chennai:[32,43,54,47], tambaram:[30,40,51,44], velachery:[34,46,57,50], avadi:[28,38,48,42], mahabalipuram:[31,42,52,46] },
        drivers: [
          { label:'CAPE (surface-based)', unit:'J/kg', vals:['1180','1500','1830','1670'], w:[0.56,0.68,0.85,0.75] },
          { label:'Vertical wind shear, 0–6 km', unit:'m/s', vals:['9','11','14','12'], w:[0.38,0.48,0.6,0.52] },
          { label:'Cloud-top temperature drop-rate', unit:'°C/15min', vals:['-2.1','-3.3','-4.5','-3.8'], w:[0.4,0.56,0.73,0.61] },
          { label:'Low-level convergence', unit:'×10⁻⁵ s⁻¹', vals:['3.3','4.6','5.9','5.2'], w:[0.34,0.48,0.63,0.54] },
          { label:'Integrated water vapour, rate of change', unit:'kg/m²/hr', vals:['2.7','3.8','4.9','4.3'], w:[0.34,0.47,0.61,0.52] },
        ],
      },
      cloudburst: {
        vals: { chennai:[36,48,61,54], tambaram:[33,44,57,50], velachery:[41,54,68,61], avadi:[29,39,50,44], mahabalipuram:[38,50,64,57] },
        drivers: [
          { label:'Integrated water vapour, rate of change', unit:'kg/m²/hr', vals:['3.7','5.0','6.6','5.8'], w:[0.5,0.65,0.86,0.75] },
          { label:'Cloud-top temperature drop-rate', unit:'°C/15min', vals:['-2.6','-3.9','-5.4','-4.6'], w:[0.44,0.6,0.79,0.67] },
          { label:'Low-level convergence', unit:'×10⁻⁵ s⁻¹', vals:['3.9','5.3','6.9','6.0'], w:[0.4,0.54,0.71,0.61] },
          { label:'Northeast monsoon moisture flux, Bay of Bengal', unit:'g/kg·m/s', vals:['210','290','380','330'], w:[0.46,0.6,0.78,0.68] },
          { label:'CAPE (surface-based)', unit:'J/kg', vals:['1030','1260','1520','1390'], w:[0.28,0.38,0.5,0.43] },
        ],
      },
      flashflood: {
        vals: { chennai:[48,62,77,70], tambaram:[42,55,69,62], velachery:[58,73,89,83], avadi:[36,47,60,53], mahabalipuram:[33,44,56,49] },
        drivers: [
          { label:'CartoDEM slope-routed runoff index', unit:'index', vals:['0.44','0.59','0.77','0.70'], w:[0.6,0.73,0.93,0.84] },
          { label:'Pallikaranai marsh & storm-drain saturation', unit:'% capacity', vals:['58','71','88','81'], w:[0.62,0.76,0.95,0.86] },
          { label:'Integrated water vapour, rate of change', unit:'kg/m²/hr', vals:['3.4','4.7','6.2','5.4'], w:[0.48,0.62,0.81,0.71] },
          { label:'Low-level convergence', unit:'×10⁻⁵ s⁻¹', vals:['3.6','4.9','6.4','5.6'], w:[0.36,0.49,0.64,0.55] },
          { label:'Cloud-top temperature drop-rate', unit:'°C/15min', vals:['-2.3','-3.5','-4.9','-4.1'], w:[0.26,0.38,0.51,0.43] },
        ],
      },
    },
    alerts: [
      { sev:'orange', id:'IN-TN-20260910-0742', headline:'Urban flash flood risk, Pallikaranai–Velachery low-lying belt', area:'Chennai South & Tambaram', window:'15:00–23:00 IST', sent:'07:42 IST' },
      { sev:'yellow', id:'IN-TN-20260910-0710', headline:'Thunderstorm cells developing off Bay of Bengal coast', area:'Mahabalipuram–ECR corridor', window:'14:00–18:00 IST', sent:'07:10 IST' },
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
