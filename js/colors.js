// Colours for the explorer, one set per theme (the page follows the system preference).
const CLASS = ["#2a78d6", "#16a39a", "#eda100", "#d6336c", "#7a8b12", "#eb6834", "#008300", "#e87ba4", "#6250d6", "#33508f"];
const VIRIDIS = ["#440154", "#472d7b", "#3b528b", "#2c728e", "#21918c", "#28ae80", "#5ec962", "#addc30", "#fde725"];

export const hexRgb = h => { const v = parseInt(h.slice(1), 16); return [(v >> 16) & 255, (v >> 8) & 255, v & 255]; };
const mix = (a, b, t) => a.map((x, i) => x + (b[i] - x) * t);
export const css = ([r, g, b], a = 1) => a === 1 ? `rgb(${r | 0},${g | 0},${b | 0})` : `rgba(${r | 0},${g | 0},${b | 0},${a})`;

function ramp(stops, t) {
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x));
  return mix(stops[i], stops[i + 1], x - i);
}

export function palette(dark) {
  const white = [255, 255, 255];
  const cls = CLASS.map(hexRgb).map(c => dark ? mix(c, white, 0.22) : c);
  const vir = VIRIDIS.map(hexRgb);
  const [t0, t1] = dark ? [0.14, 1] : [0, 0.88];          // drop the end that vanishes into the background
  const div = dark ? ["#5b9be0", "#5d5c63", "#ec5f68"].map(hexRgb) : ["#2166ac", "#cfcdc6", "#b2182b"].map(hexRgb);
  return {
    dark,
    cls,
    correct: hexRgb(dark ? "#6aa8ef" : "#2a78d6"),
    wrong: hexRgb(dark ? "#f39058" : "#d9541e"),
    undef: hexRgb(dark ? "#77767d" : "#a3a29c"),
    fg: hexRgb(dark ? "#ececea" : "#16161a"),
    bg: hexRgb(dark ? "#141417" : "#fafaf8"),
    accent: hexRgb(dark ? "#ff8a4c" : "#c2410c"),
    guide: hexRgb(dark ? "#8c8b93" : "#7b7a74"),
    seq: t => ramp(vir, t0 + (t1 - t0) * t),
    div: t => t < 0.5 ? mix(div[1], div[0], Math.pow((0.5 - t) * 2, 0.8)) : mix(div[1], div[2], Math.pow((t - 0.5) * 2, 0.8)),
  };
}
export const gradient = (f, n = 9) => `linear-gradient(90deg, ${Array.from({ length: n }, (_, i) => css(f(i / (n - 1)))).join(",")})`;
