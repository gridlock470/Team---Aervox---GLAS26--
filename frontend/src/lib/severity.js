export const SEV = {
  green:  { hex: '#3F8F63', rgb: [63, 143, 99],   label: 'Green' },
  yellow: { hex: '#96982E', rgb: [150, 152, 46],  label: 'Yellow' },
  orange: { hex: '#C6621F', rgb: [198, 98, 31],   label: 'Orange' },
  red:    { hex: '#BC3B3E', rgb: [188, 59, 62],   label: 'Red' },
};

export const SEV_ORDER = ['green', 'yellow', 'orange', 'red'];

export function sevFor(pct) {
  if (pct >= 70) return 'red';
  if (pct >= 45) return 'orange';
  if (pct >= 20) return 'yellow';
  return 'green';
}

export function sevRgba(pct, alpha01 = 0.22 + (pct / 100) * 0.4) {
  const [r, g, b] = SEV[sevFor(pct)].rgb;
  return [r, g, b, Math.round(Math.max(0, Math.min(1, alpha01)) * 255)];
}

export function hexToRgba(hex, alpha01 = 1) {
  const h = hex.replace('#', '');
  const r = parseInt(h.substring(0, 2), 16);
  const g = parseInt(h.substring(2, 4), 16);
  const b = parseInt(h.substring(4, 6), 16);
  return [r, g, b, Math.round(alpha01 * 255)];
}

const BASE_RADIUS_M = 1500;
const PCT_RADIUS_SCALE_M = 60;
export function radiusMeters(pct) {
  return BASE_RADIUS_M + pct * PCT_RADIUS_SCALE_M;
}
