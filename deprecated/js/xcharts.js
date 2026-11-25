// Small SVG charts for the experiment pages: one visual grammar with the explorer (site tokens, thin marks,
// hairline axes, a crosshair tooltip). chart(el, spec) draws lines, dots and grouped or stacked bars over a band
// axis (depth steps) or a numeric axis (linear, or log10(step + 1) for training steps).
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
export const fmt = (v, d = 2) => v == null || !Number.isFinite(v) ? "n/a" : Math.abs(v) >= 1000 ? v.toLocaleString("en-US", { maximumFractionDigits: 0 })
  : Math.abs(v) > 0 && Math.abs(v) < 1e-3 ? v.toExponential(1) : (+v.toFixed(d)).toString();
export const pct = (v, d = 1) => v == null || !Number.isFinite(v) ? "n/a" : (100 * v).toFixed(d) + "%";
export const stepLabel = s => s >= 1000 ? (s % 1000 === 0 ? s / 1000 + "k" : (s / 1000).toFixed(1) + "k") : String(s);

// colours: depth steps on an ordinal ramp (light to dark = shallow to deep), categories from the set palette
const DEPTH = ["#9ec5f4", "#7fb0ee", "#5f99e6", "#3f80d8", "#2a6bc4", "#2157a8", "#1b4689", "#15366b", "#0e2549"];
const DEPTH_DARK = ["#3c5f8f", "#4a72a8", "#5986c0", "#6b9ad3", "#7eaee2", "#93c0ec", "#a9d0f3", "#c0defa", "#d8ebff"];
export const depthColor = i => (matchMedia("(prefers-color-scheme: dark)").matches ? DEPTH_DARK : DEPTH)[i];
export const C = { a: "#2a78d6", b: "#eb6834", c: "#1baf7a", d: "#9a9993", e: "#d6336c", f: "#6250d6" };

const tip = () => {
  let t = document.getElementById("tooltip");
  if (!t) { t = document.createElement("div"); t.id = "tooltip"; t.className = "tooltip"; t.hidden = true; document.body.append(t); }
  return t;
};

function ticks(lo, hi, n = 5) {
  const span = hi - lo || 1, s0 = span / n, mag = 10 ** Math.floor(Math.log10(s0));
  const step = [1, 2, 5, 10].map(m => m * mag).find(s => span / s <= n + 0.5) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9 * span; v += step) out.push(+v.toFixed(10));
  return out;
}

/*
spec: { x: {type: "band", labels: [...]} | {type: "linear"|"logstep", domain: [a, b], label},
        y: {domain: [a, b], label, log: bool, fmt: v => string, ticks: [...]},
        series: [{name, values: [...] (band) or points: [[x, y], ...], color, kind: "line"|"dot"|"bar"|"stack", dash, width, faint}],
        height, legend (default true), hlines: [{y, label}], vband: [x0, x1] (shaded), marker: x (vertical line) }
*/
const stacksTotal = spec => { const st = spec.series.filter(s => s.kind === "stack"); if (!st.length) return 0;
  return Math.max(...st[0].values.map((_, i) => st.reduce((a, s) => a + (Number.isFinite(s.values[i]) ? s.values[i] : 0), 0))); };
export function chart(host, spec) {
  host.replaceChildren();
  host.classList.add("xc");
  const W = Math.max(spec.compact ? 90 : 260, host.clientWidth || 600), H = spec.height || 260;
  const m = spec.compact ? { l: 26, r: 4, t: 6, b: 10 } : { l: 58, r: 14, t: 10, b: spec.x.label ? 46 : spec.x.type === "band" ? 40 : 30 };
  // band labels that do not fit even when shortened are drawn rotated
  let rot = false;
  if (spec.x.type === "band" && !spec.compact) {
    const bw0 = (W - m.l - m.r) / spec.x.labels.length;
    const fit = l => l.length * 5.6 <= bw0 - 2;
    if (!spec.x.labels.every(l => fit(l) || (l.includes("→") && fit(l.split("→")[0])))) {
      rot = true; m.b += Math.min(70, Math.max(...spec.x.labels.map(l => l.length)) * 4.6);
    }
  }
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": spec.aria || "" }, host);
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const fg = cssVar("--fg"), mute = cssVar("--mute"), line = cssVar("--line"), bg = cssVar("--surface");
  // x scale
  let X, xs = [], band = spec.x.type === "band";
  if (band) {
    const n = spec.x.labels.length, bw = iw / n;
    X = i => m.l + bw * (i + 0.5); X.bw = bw;
    xs = spec.x.labels.map((_, i) => i);
  } else {
    const tf = spec.x.type === "logstep" ? v => Math.log10(v + 1) : v => v;
    const [a, b] = spec.x.domain.map(tf);
    X = v => m.l + (tf(v) - a) / (b - a || 1) * iw; X.tf = tf;
  }
  // y scale
  const ylog = !!spec.y.log;
  const [y0, y1] = spec.y.domain;
  const ty = ylog ? v => Math.log10(Math.max(v, y0)) : v => v;
  const Y = v => m.t + ih - (ty(v) - ty(y0)) / (ty(y1) - ty(y0) || 1) * ih;
  const yf = spec.y.fmt || (v => fmt(v, 3));
  // grid and axes
  const g = el("g", { class: "xc-grid" }, svg);
  const yt = spec.y.ticks || (ylog ? (() => { const out = []; for (let e = Math.ceil(Math.log10(y0)); e <= Math.floor(Math.log10(y1)); e++) out.push(10 ** e); return out; })() : ticks(y0, y1, Math.max(3, Math.round(ih / 50))));
  for (const v of yt) {
    const y = Y(v);
    el("line", { x1: m.l, x2: m.l + iw, y1: y, y2: y, stroke: line, "stroke-width": 1, "shape-rendering": "crispEdges" }, g);
    const t = el("text", { x: m.l - 6, y: y + 3.5, "text-anchor": "end", class: "xc-t" }, g); t.textContent = ylog ? (v >= 1e-2 && v <= 1e3 ? fmt(v, 2) : "1e" + Math.round(Math.log10(v))) : yf(v);
  }
  if (y0 < 0 && y1 > 0 && !ylog) el("line", { x1: m.l, x2: m.l + iw, y1: Y(0), y2: Y(0), stroke: mute, "stroke-width": 1, "shape-rendering": "crispEdges" }, g);
  if (band) {
    // long step labels ("b3→b4") shrink to their first site when the band is too narrow
    const long = spec.x.labels.some(l => l.length * 6.2 > X.bw - 2);
    if (spec.compact) spec.x.labels.forEach(() => {});
    const shown = long ? spec.x.labels.map(l => l.split("→")[0]) : spec.x.labels;
    const tiny = shown.some(l => l.length * 6.2 > X.bw - 2);
    shown.forEach((lab, i) => {
      if (rot) { const y = m.t + ih + 12, t = el("text", { x: X(i), y, "text-anchor": "end", class: "xc-t", transform: `rotate(-32 ${X(i)} ${y})` }, g); t.textContent = spec.x.labels[i]; return; }
      const t = el("text", { x: X(i), y: m.t + ih + 16, "text-anchor": "middle", class: "xc-t" }, g); t.textContent = lab; if (tiny) t.setAttribute("font-size", "9"); });
    if (!rot && long && shown.some((l, i) => l !== spec.x.labels[i]) && !spec.x.label) { const t = el("text", { x: m.l + iw, y: H - 6, "text-anchor": "end", class: "xc-t" }, g); t.textContent = "step from the site shown"; }
  } else {
    const tv = spec.x.ticks || (spec.x.type === "logstep" ? [0, 1, 10, 100, 1000, 10000, 64000].filter(v => v <= spec.x.domain[1]) : ticks(...spec.x.domain, 6));
    for (const v of tv) { const t = el("text", { x: X(v), y: m.t + ih + 16, "text-anchor": "middle", class: "xc-t" }, g); t.textContent = spec.x.type === "logstep" ? stepLabel(v) : fmt(v, 2); }
  }
  el("line", { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih, stroke: mute, "stroke-width": 1, "shape-rendering": "crispEdges" }, g);
  if (spec.x.label) { const t = el("text", { x: m.l + iw / 2, y: H - 6, "text-anchor": "middle", class: "xc-l" }, g); t.textContent = spec.x.label; }
  if (spec.y.label) { const t = el("text", { x: 12, y: m.t + ih / 2, "text-anchor": "middle", class: "xc-l", transform: `rotate(-90 12 ${m.t + ih / 2})` }, g); t.textContent = spec.y.label; }
  if (spec.vband) el("rect", { x: X(spec.vband[0]), width: Math.max(1, X(spec.vband[1]) - X(spec.vband[0])), y: m.t, height: ih, fill: mute, opacity: 0.1 }, g);
  for (const hl of spec.hlines || []) {
    el("line", { x1: m.l, x2: m.l + iw, y1: Y(hl.y), y2: Y(hl.y), stroke: hl.color || mute, "stroke-dasharray": "4 3", "stroke-width": 1 }, g);
    if (hl.label) { const t = el("text", { x: m.l + iw - 4, y: Y(hl.y) - 4, "text-anchor": "end", class: "xc-t" }, g); t.textContent = hl.label; }
  }
  const clipY = v => Math.max(m.t, Math.min(m.t + ih, Y(v)));
  // values outside the y range are drawn at its edge and counted in a note under the chart
  let clipped = 0;
  for (const s of spec.series) for (const v of (s.values || (s.points || []).map(p => p[1]))) {
    if (v == null || !Number.isFinite(v) || (ylog && v <= 0)) continue;
    if (v < y0 - 1e-12 * Math.abs(y0) || v > y1 + 1e-12 * Math.abs(y1)) clipped++;
  }
  if (stacksTotal(spec) > y1 + 1e-9) clipped++;
  // bars (grouped; series with kind "stack" stack within one group)
  const bars = spec.series.filter(s => s.kind === "bar"), stacks = spec.series.filter(s => s.kind === "stack");
  const groups = bars.length + (stacks.length ? 1 : 0);
  if (band && groups) {
    const gw = X.bw * 0.78, w = gw / groups;
    bars.forEach((s, k) => s.values.forEach((v, i) => {
      if (v == null || !Number.isFinite(v)) return;
      const x = X(i) - gw / 2 + k * w, base = ylog ? Y(y0) : Y(Math.max(y0, 0));
      el("rect", { x: x + 0.5, width: Math.max(1, w - 1), y: Math.min(base, clipY(v)), height: Math.abs(clipY(v) - base), fill: s.color, opacity: s.faint ? 0.45 : 0.9 }, svg);
    }));
    if (stacks.length) {
      const x = i => X(i) - gw / 2 + bars.length * w;
      const cum = spec.x.labels.map(() => 0);
      for (const s of stacks) s.values.forEach((v, i) => {
        if (v == null || !Number.isFinite(v)) return;
        el("rect", { x: x(i) + 0.5, width: Math.max(1, w - 1), y: Y(cum[i] + v), height: Math.max(0, Y(cum[i]) - Y(cum[i] + v)), fill: s.color, opacity: 0.88 }, svg);
        cum[i] += v;
      });
    }
  }
  // lines and dots
  for (const s of spec.series) {
    if (s.kind === "bar" || s.kind === "stack") continue;
    const pts = band ? s.values.map((v, i) => [X(i), v]) : s.points.map(([x, v]) => [X(x), v]);
    const ok = pts.filter(p => p[1] != null && Number.isFinite(p[1]) && (!ylog || p[1] > 0));
    if (s.kind !== "dot" && ok.length > 1) {
      let d = "", pen = false;
      for (const [x, v] of pts) { if (v == null || !Number.isFinite(v) || (ylog && v <= 0)) { pen = false; continue; } d += (pen ? "L" : "M") + x.toFixed(1) + " " + clipY(v).toFixed(1); pen = true; }
      el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 1.8, "stroke-dasharray": s.dash || "", opacity: s.faint ? 0.5 : 1, "stroke-linejoin": "round" }, svg);
    }
    if (s.kind === "dot" || s.markers || band) for (const [x, v] of ok) el("circle", { cx: x, cy: clipY(v), r: s.kind === "dot" ? 3.6 : 2.4, fill: s.hollow ? bg : s.color, stroke: s.color, "stroke-width": 1.4, opacity: s.faint ? 0.5 : 1 }, svg);
  }
  if (spec.marker != null) el("line", { x1: X(spec.marker), x2: X(spec.marker), y1: m.t, y2: m.t + ih, stroke: cssVar("--accent"), "stroke-dasharray": "3 3", "stroke-width": 1.2 }, svg);
  // legend
  if (spec.legend !== false) {
    const lg = document.createElement("div"); lg.className = "xc-legend";
    for (const s of spec.series) {
      if (!s.name || s.nolegend) continue;
      const k = document.createElement("span"); k.className = "k";
      const sw = document.createElement("i"); sw.className = s.kind === "bar" || s.kind === "stack" ? "box" : s.kind === "dot" ? "dot" : "ln";
      sw.style.setProperty("--sw", s.color); if (s.dash) sw.classList.add("dash");
      k.append(sw, s.name); lg.append(k);
    }
    if (lg.childElementCount) host.append(lg);
  }
  if (clipped) { const n = document.createElement("div"); n.className = "xc-clip"; n.textContent = `${clipped} value${clipped > 1 ? "s" : ""} beyond the axis range, drawn at its edge`; host.append(n); }
  // crosshair tooltip
  const hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" }, svg);
  const cross = el("line", { y1: m.t, y2: m.t + ih, stroke: mute, "stroke-width": 1, visibility: "hidden" }, svg);
  const xsAll = band ? xs : [...new Set(spec.series.flatMap(s => (s.points || []).map(p => p[0])))].sort((a, b) => a - b);
  const move = ev => {
    const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * (W / r.width);
    let best = null, bd = Infinity;
    for (const xv of xsAll) { const d = Math.abs(X(xv) - px); if (d < bd) { bd = d; best = xv; } }
    if (best == null) return;
    cross.setAttribute("x1", X(best)); cross.setAttribute("x2", X(best)); cross.setAttribute("visibility", "visible");
    const rows = spec.series.filter(s => s.name && !s.notip).map(s => {
      const v = band ? s.values[best] : (s.points.find(p => p[0] === best) || [])[1];
      return v == null ? "" : `<div class="tt-row"><span><i class="tt-line" style="--sw:${s.color}"></i>${s.name}</span><span class="tt-v">${(s.tfmt || yf)(v)}</span></div>`;
    }).join("");
    const head = band ? spec.x.labels[best] : (spec.x.tipLabel || (v => (spec.x.type === "logstep" ? "step " : "") + fmt(v, 3)))(best);
    const t = tip(); t.innerHTML = `<div class="tt-head">${head}</div>${rows}`; t.hidden = false;
    const tr = t.getBoundingClientRect();
    t.style.left = Math.min(innerWidth - tr.width - 8, ev.clientX + 14) + "px"; t.style.top = Math.min(innerHeight - tr.height - 8, ev.clientY + 12) + "px";
  };
  hit.addEventListener("pointermove", move);
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); tip().hidden = true; });
  return svg;
}

// a table whose cells are shaded by value in [0, 1] (shares of images)
export function shareTable(host, { rows, cols, values, fmtv = v => v === 1 ? "100%" : pct(v, v > 0 && v < 0.01 ? 2 : v > 0.99 ? 1 : 0), rowHead = "" }) {
  const t = document.createElement("table"); t.className = "xc-share";
  const hr = t.createTHead().insertRow(); hr.insertCell().outerHTML = `<th>${rowHead}</th>`;
  for (const c of cols) hr.insertCell().outerHTML = `<th>${c}</th>`;
  const body = t.createTBody();
  rows.forEach((r, i) => {
    const tr = body.insertRow(); tr.insertCell().outerHTML = `<th>${r}</th>`;
    values[i].forEach(v => {
      const td = tr.insertCell();
      if (v == null) { td.textContent = "–"; td.className = "na"; return; }
      td.textContent = fmtv(v); td.style.setProperty("--a", Math.max(0.04, Math.min(1, v)).toFixed(3));
      if (v > 0.55) td.classList.add("hi");
    });
  });
  host.replaceChildren(t);
}

// redraw every registered chart when the width or the theme changes
const drawn = [];
export function register(fn) { drawn.push(fn); fn(); }
let rq = 0;
const redraw = () => { cancelAnimationFrame(rq); rq = requestAnimationFrame(() => drawn.forEach(f => f())); };
addEventListener("resize", redraw);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
