/* One palette serves both themes: these hexes are handed straight to inline
   styles and to deck.gl, which cannot see the CSS theme. They are therefore
   mid-tone on purpose — each clears 3:1 as a graphic mark against #050507 and
   against #FFFFFF. They also sit deeper than the interface gold (#D4AF37 /
   #EAB308) so a "watch" never reads as chrome, and chrome never reads as a
   warning. Keys, labels and thresholds are unchanged. */
export const SEV = {
  green:  { hex: '#0E9F6E', rgb: [14, 159, 110],  label: 'Green' },
  yellow: { hex: '#C08A0E', rgb: [192, 138, 14],  label: 'Yellow' },
  orange: { hex: '#E2650F', rgb: [226, 101, 15],  label: 'Orange' },
  red:    { hex: '#E03150', rgb: [224, 49, 80],   label: 'Red' },
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
