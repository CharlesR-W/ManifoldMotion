// The explorer: the lens cloud in one fixed 3D projection, moved through depth and training.
import * as D from "./data.js";
import { palette, css, gradient } from "./colors.js";
import { View } from "./view3d.js";
// the experiment pages' chart module, shared so the totals card reads like the figures there
const XC = { ready: import("./xcharts.js").then(mod => Object.assign(XC, mod)) };

const K = 20;                                    // neighbours carried forward
const ARROW_EVERY = 16;                          // arrow subsample: every 16th image (625 arrows) when nothing is selected
const SAMPLES = 8;                               // points per projected great-circle arc (arrows, trails)
const DWELL = 0.3;                               // share of each playback unit spent resting on the data frame
const SPEED = { depth: 1 / 1300, train: 1 / 800 }; // playback units per ms
const dark = matchMedia("(prefers-color-scheme: dark)");
const coarse = matchMedia("(pointer: coarse)");

export function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const k in attrs) {
    const v = attrs[k];
    if (v == null || v === false) continue;
    if (k === "text") e.textContent = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") Object.assign(e.style, v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) e.append(c);
  return e;
}
const fmtInt = n => n.toLocaleString("en-US");
const pct = v => (100 * v).toFixed(1) + "%";
const short = n => n >= 1000 ? (n % 1000 === 0 ? n / 1000 : (n / 1000).toFixed(1)) + "k" : String(n);
const sig = v => !Number.isFinite(v) ? "n/a" : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(3);

// great-circle interpolation between unit vectors a (offset ao) and b (offset bo) of length 10, into out
function slerp(a, ao, b, bo, f, out) {
  let dot = 0;
  for (let c = 0; c < 10; c++) dot += a[ao + c] * b[bo + c];
  dot = Math.max(-1, Math.min(1, dot));
  const th = Math.acos(dot);
  if (th < 1e-7) { for (let c = 0; c < 10; c++) out[c] = a[ao + c] + (b[bo + c] - a[ao + c]) * f; return out; }
  const s = Math.sin(th), wa = Math.sin((1 - f) * th) / s, wb = Math.sin(f * th) / s;
  for (let c = 0; c < 10; c++) out[c] = wa * a[ao + c] + wb * b[bo + c];
  return out;
}
// indices of the k nearest rows to row i (squared chord, which orders like Fisher-Rao distance on the sphere)
function knn(s, i, k, N) {
  const d = new Float64Array(N), o = i * 10;
  for (let j = 0; j < N; j++) {
    let t = 0;
    for (let c = 0; c < 10; c++) { const x = s[j * 10 + c] - s[o + c]; t += x * x; }
    d[j] = t;
  }
  d[i] = Infinity;
  const idx = Array.from({ length: N }, (_, j) => j);
  idx.sort((x, y) => d[x] - d[y] || x - y);
  return idx.slice(0, k);
}
const fisher = (s, i, j) => {      // Fisher-Rao distance between rows i and j, X = 2 s
  let a = 0, b = 0;
  for (let c = 0; c < 10; c++) { const u = s[i * 10 + c], v = s[j * 10 + c]; a += (u - v) ** 2; b += (u + v) ** 2; }
  return 4 * Math.atan2(Math.sqrt(a), Math.sqrt(b));
};

export class Explorer {
  constructor(root, params) {
    this.root = root;
    this.ready = this.init(params).catch(e => {
      console.error(e);
      root.replaceChildren(h("p", { class: "placeholder", text: `The explorer could not load its data: ${e.message}` }));
    });
  }

  async init(params) {
    const [m, lab] = await Promise.all([D.meta(), D.labels()]);
    this.m = m; this.lab = lab; this.N = m.N; this.S = m.sites.length;
    this.fields = new Map(m.fields.map(f => [f.name, f]));
    this.fieldData = new Map();
    this.pal = palette(dark.matches);
    this.st = { ci: m.steps.length - 1, site: this.S - 1, proj: "octant", a: 3, b: 5, color: "class", arrows: false,
      filter: new Set(m.classes.map((_, c) => c)), sel: -1, trail: "depth", carry: null };
    this.readParams(params);
    this.anim = null; this.hover = -1; this.cur = null;
    this.tmp = new Float64Array(10); this.p3 = new Float64Array(3);
    this.build();
    this.view = new View(this.stage, this.N, ctx => this.overlay(ctx));
    this.setGuides(true);
    this.wire();
    await this.show();
    this.prefetchDepth();
    dark.addEventListener("change", () => { this.pal = palette(dark.matches); this.setGuides(false); this.renderLegend(); this.renderSel(); this.paint(); });
  }

  // ------------------------------------------------------------------ state <-> URL
  readParams(q) {
    if (!q) return;
    const st = this.st, m = this.m, int = (k, lo, hi) => { const v = parseInt(q.get(k), 10); return Number.isFinite(v) && v >= lo && v <= hi ? v : null; };
    const step = int("step", 0, Infinity); if (step != null && m.steps.includes(step)) st.ci = m.steps.indexOf(step);
    const site = int("site", 0, this.S - 1); if (site != null) st.site = site;
    if (["octant", "pca"].includes(q.get("proj"))) st.proj = q.get("proj");
    const a = int("a", 0, 9), b = int("b", 0, 9); if (a != null && b != null && a !== b) { st.a = a; st.b = b; }
    const col = q.get("color"); if (col && (["class", "pred", "correct"].includes(col) || (col.startsWith("f:") && this.fields.has(col.slice(2))))) st.color = col;
    const sel = int("sel", 0, this.N - 1); if (sel != null) st.sel = sel;
    if (["depth", "train", "off"].includes(q.get("trail"))) st.trail = q.get("trail");
    st.arrows = q.get("arrows") === "1";
  }
  applyParams(q) { this.stop(); this.readParams(q); this.syncControls(); this.setGuides(true); this.show(); }
  writeHash() {
    clearTimeout(this.hashT);
    this.hashT = setTimeout(() => {
      const st = this.st, q = new URLSearchParams({ step: this.m.steps[st.ci], site: st.site, proj: st.proj });
      if (st.proj === "octant") { q.set("a", st.a); q.set("b", st.b); }
      if (st.color !== "class") q.set("color", st.color);
      if (st.arrows) q.set("arrows", "1");
      if (st.sel >= 0) { q.set("sel", st.sel); if (st.trail !== "depth") q.set("trail", st.trail); }
      const hash = "#explorer?" + q.toString();
      if (location.hash.startsWith("#explorer")) { history.replaceState(null, "", hash); this.lastHash = hash; }
    }, 250);
  }

  // ------------------------------------------------------------------ DOM
  build() {
    const m = this.m, st = this.st;
    const clsOpts = sel => m.classes.map((c, i) => h("option", { value: i, selected: i === sel }, c));
    this.el = {};
    const E = this.el;
    E.proj = h("div", { class: "seg", role: "radiogroup", "aria-label": "Projection" },
      h("button", { type: "button", role: "radio", "data-v": "octant" }, "Fisher octant"),
      h("button", { type: "button", role: "radio", "data-v": "pca" }, "Principal axes"));
    E.a = h("select", { "aria-label": "Class A" }, clsOpts(st.a));
    E.b = h("select", { "aria-label": "Class B" }, clsOpts(st.b));
    E.ab = h("div", { class: "ctl" }, h("span", { class: "ctl-label" }, "A"), E.a, h("span", { class: "ctl-label" }, "B"), E.b);
    E.color = h("select", { "aria-label": "Colour points by" });
    E.arrows = h("input", { type: "checkbox" });
    E.helpBtn = h("button", { type: "button", class: "guide-btn", "aria-expanded": "false", "aria-controls": "ex-help" }, "How to read this");
    E.help = h("div", { class: "ex-help", id: "ex-help", hidden: true, html: HELP });
    const bar = h("div", { class: "ex-bar" },
      h("div", { class: "ctl" }, h("span", { class: "ctl-label" }, "Projection"), E.proj),
      E.ab,
      h("div", { class: "ctl" }, h("span", { class: "ctl-label" }, "Colour"), E.color),
      h("label", { class: "chk" }, E.arrows, "Arrows to next site"),
      h("div", { class: "ctl ctl-end" }, E.helpBtn));

    E.frame = h("div", { class: "ex-frame" });
    E.interp = h("div", { class: "ex-interp", hidden: true });
    E.reset = h("button", { type: "button", class: "mini-btn", title: "Return to the starting view" }, "Reset view");
    E.loading = h("div", { class: "ex-loadnote", hidden: true }, "loading…");
    this.stage = h("div", { class: "ex-stage" },
      h("canvas", { class: "gl", "aria-label": "3D view of the image cloud; drag to rotate, scroll or pinch to zoom, click a point to select it" }),
      h("canvas", { class: "ov", "aria-hidden": "true" }), E.frame, E.interp, h("div", { class: "ex-stage-tools" }, E.reset), E.loading);

    const slider = (kind, max, label) => {
      const play = h("button", { type: "button", class: "play", "aria-label": `Play through ${label.toLowerCase()}` }, "▶");
      const input = h("input", { type: "range", min: 0, max, step: 1, "aria-label": label });
      const val = h("span", { class: "sl-val" });
      const ticks = h("div", { class: "sl-ticks", "aria-hidden": "true" });
      return { play, input, val, ticks, row: h("div", { class: "sl-row" }, play, h("span", { class: "sl-label" }, label), h("div", { class: "sl-track" }, input, ticks), val) };
    };
    E.dep = slider("depth", this.S - 1, "Depth");
    E.trn = slider("train", m.steps.length - 1, "Training");
    m.sites.forEach((s, i) => E.dep.ticks.append(h("span", { style: { left: `calc(7px + (100% - 14px) * ${i / (this.S - 1)})` } }, s)));
    for (const s of [0, 16, 256, 2560, 25600, m.steps[m.steps.length - 1]]) {
      const i = m.steps.indexOf(s);
      if (i >= 0) E.trn.ticks.append(h("span", { style: { left: `calc(7px + (100% - 14px) * ${i / (m.steps.length - 1)})` } }, short(s)));
    }
    E.legend = h("div", { class: "ex-legend" });
    E.note = h("p", { class: "ex-note" });

    // side cards
    E.read = h("div", { class: "card" });
    E.sel = h("div", { class: "card card-sel" });
    E.filter = h("div", { class: "card" });
    E.totals = h("div", { class: "card", hidden: true });
    const main = h("div", { class: "ex-main" }, this.stage, E.legend,
      h("div", { class: "ex-sliders" }, E.dep.row, E.trn.row,
        h("p", { class: "sl-note" }, `Training slider: the ${m.steps.length} included checkpoints, evenly spaced by index (not by step).`)),
      E.note);
    const side = h("aside", { class: "ex-side" }, E.read, E.sel, E.filter, E.totals);
    this.root.replaceChildren(bar, E.help, h("div", { class: "ex-body" }, main, side));
    this.buildFilter();
    this.syncControls();
    let seen = false;
    try { seen = localStorage.getItem("mm-help-seen") === "1"; } catch (e) { /* storage unavailable */ }
    if (!seen) this.toggleHelp(true);
  }

  toggleHelp(open) {
    const E = this.el;
    open = open ?? E.help.hidden;
    E.help.hidden = !open;
    E.helpBtn.setAttribute("aria-expanded", String(open));
    if (!open) try { localStorage.setItem("mm-help-seen", "1"); } catch (e) { /* storage unavailable */ }
  }

  syncControls() {
    const E = this.el, st = this.st, m = this.m;
    E.proj.querySelectorAll("button").forEach(b => b.setAttribute("aria-checked", String(b.dataset.v === st.proj)));
    E.ab.hidden = st.proj !== "octant";
    E.a.value = st.a; E.b.value = st.b;
    E.arrows.checked = st.arrows;
    // colour menu (fields available at this checkpoint are enabled)
    const step = m.steps[st.ci];
    const opts = [h("option", { value: "class" }, "True class"), h("option", { value: "pred" }, "Lens prediction (this frame)"),
      h("option", { value: "correct" }, "Correct at the last site")];
    // fields, grouped by fields_meta "group" (in order of first appearance); each colours the step to the next site
    const groups = new Map();
    for (const f of m.fields) {
      const g = f.group || "Per-point fields";
      if (!groups.has(g)) groups.set(g, h("optgroup", { label: `${g} (step to next site)` }));
      const here = f.steps.includes(step);
      groups.get(g).append(h("option", { value: "f:" + f.name, disabled: !here }, f.label + (here ? "" : " (not at this checkpoint)")));
    }
    opts.push(...groups.values());
    E.color.replaceChildren(...opts);
    E.color.value = st.color;
    E.dep.input.value = st.site; E.trn.input.value = st.ci;
    this.renderLegend();
    this.renderNote();
  }

  buildFilter() {
    const E = this.el, m = this.m, st = this.st;
    const counts = new Array(10).fill(0);
    for (let i = 0; i < this.N; i++) counts[this.lab[i]]++;
    this.counts = counts;
    E.chips = m.classes.map((c, i) => h("button", { type: "button", class: "chip", "aria-pressed": "true", "data-c": i,
      onclick: ev => { if (ev.altKey || ev.metaKey) { st.filter = new Set([i]); } else if (st.filter.has(i)) st.filter.delete(i); else st.filter.add(i); this.afterFilter(); } },
      h("i", { class: "sw" }), c));
    E.filter.replaceChildren(
      h("h3", { class: "card-title" }, "Show classes"),
      h("div", { class: "chips" }, E.chips),
      h("div", { class: "chip-acts" },
        h("button", { type: "button", class: "link-btn", onclick: () => { st.filter = new Set(m.classes.map((_, c) => c)); this.afterFilter(); } }, "all"),
        h("button", { type: "button", class: "link-btn", onclick: () => { st.filter = new Set(); this.afterFilter(); } }, "none"),
        h("span", { class: "mute" }, "Alt-click a class to show only it.")));
    this.styleChips();
  }
  styleChips() {
    this.el.chips.forEach((b, i) => { b.setAttribute("aria-pressed", String(this.st.filter.has(i))); b.firstChild.style.background = css(this.pal.cls[i]); });
  }
  afterFilter() { this.styleChips(); this.paint(); }

  wire() {
    const E = this.el, st = this.st;
    E.proj.addEventListener("click", e => {
      const b = e.target.closest("button"); if (!b || b.dataset.v === st.proj) return;
      st.proj = b.dataset.v; this.syncControls(); this.setGuides(true); this.paint(); this.writeHash();
    });
    const ab = () => {
      let a = +E.a.value, b = +E.b.value;
      if (a === b) { if (a === st.a) a = st.b; else b = st.a; }
      st.a = a; st.b = b; E.a.value = a; E.b.value = b;
      this.setGuides(false); this.paint(); this.writeHash();
    };
    E.a.addEventListener("change", ab); E.b.addEventListener("change", ab);
    E.color.addEventListener("change", () => { st.color = E.color.value; this.renderLegend(); this.show(); this.writeHash(); });
    E.arrows.addEventListener("change", () => { st.arrows = E.arrows.checked; this.show(); this.writeHash(); });
    E.helpBtn.addEventListener("click", () => this.toggleHelp());
    E.reset.addEventListener("click", () => this.setGuides(true));
    E.dep.input.addEventListener("input", () => { this.stop(); st.site = +E.dep.input.value; this.show(); this.writeHash(); });
    E.trn.input.addEventListener("input", () => { this.stop(); st.ci = +E.trn.input.value; this.syncControls(); this.show(); this.writeHash(); this.prefetchDepth(); });
    E.dep.play.addEventListener("click", () => this.play("depth"));
    E.trn.play.addEventListener("click", () => this.play("train"));

    // picking: hover shows a card, a click (not a drag) selects
    const gl = this.view.gl;
    let down = null, hoverQ = false, lastMove = null;
    const at = e => { const r = gl.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    gl.addEventListener("pointerdown", e => { down = [e.clientX, e.clientY]; });
    gl.addEventListener("pointerup", e => {
      if (!down) return;
      const moved = Math.hypot(e.clientX - down[0], e.clientY - down[1]); down = null;
      if (moved > 5) return;
      const [x, y] = at(e);
      const i = this.view.pick(x, y, coarse.matches || e.pointerType === "touch" ? 18 : 9);
      this.select(i);
    });
    gl.addEventListener("pointermove", e => {
      if (e.pointerType === "touch" || e.buttons) { this.setHover(-1); return; }
      lastMove = e;
      if (hoverQ) return;
      hoverQ = true;
      requestAnimationFrame(() => {
        hoverQ = false;
        const [x, y] = at(lastMove);
        this.setHover(this.view.pick(x, y, 9), lastMove.clientX, lastMove.clientY);
      });
    });
    gl.addEventListener("pointerleave", () => this.setHover(-1));
    window.addEventListener("keydown", e => {
      if (document.body.dataset.tab !== "explorer" || e.target.closest("input, select, textarea")) return;
      if (e.key === "Escape") { if (!this.el.help.hidden) this.toggleHelp(false); else this.select(-1); }
    });
  }

  // ------------------------------------------------------------------ data
  stepOf(ci) { return this.m.steps[ci]; }
  // what the current state needs in memory before it can be drawn
  needs() {
    const st = this.st, m = this.m, step = m.steps[st.ci], out = [D.frame(m, step, st.site)];
    if (st.arrows && st.site < this.S - 1) out.push(D.frame(m, step, st.site + 1));
    if (st.color === "correct") out.push(D.frame(m, step, this.S - 1));
    if (st.color.startsWith("f:")) out.push(this.loadField(st.color.slice(2), step));
    if (st.sel >= 0 && st.trail === "depth") for (let s = 0; s < this.S; s++) out.push(D.frame(m, step, s));
    if (st.carry) for (let s = 0; s < this.S; s++) out.push(D.frame(m, step, s));
    return out;
  }
  loadField(name, step) {
    const f = this.fields.get(name), key = `${name}_${step}`;
    if (!f || !f.steps.includes(step)) return Promise.resolve(null);
    if (this.fieldData.has(key)) return Promise.resolve(this.fieldData.get(key));
    return D.field(name, step).then(a => { this.fieldData.set(key, a); return a; });
  }
  async show() {
    const st = this.st, gen = (this.gen = (this.gen || 0) + 1);
    this.el.loading.hidden = false; this.el.loading.textContent = "loading…";
    try { await Promise.all(this.needs()); }
    catch (e) { console.error(e); this.el.loading.textContent = "failed to load: " + e.message; return; }
    if (gen !== this.gen) return;
    this.el.loading.hidden = true;
    if (this.anim) return;
    if (st.trail === "train" && st.sel >= 0) this.loadTrainTrail();
    this.selKey = null;
    this.paint();
    this.renderTotals();
  }
  prefetchDepth() { const step = this.m.steps[this.st.ci]; for (let s = 0; s < this.S; s++) D.frame(this.m, step, s).catch(() => {}); }

  // ------------------------------------------------------------------ playback
  play(kind) {
    if (this.anim && this.anim.kind === kind) { this.stop(); this.show(); return; }
    this.stop();
    const st = this.st, m = this.m;
    const end = kind === "depth" ? this.S - 1 : m.steps.length - 1;
    let u = Math.max(0, Math.min(end, kind === "depth" ? st.site : st.ci));
    if (u >= end) u = 0;
    this.anim = { kind, u, end, last: performance.now() };
    const btn = kind === "depth" ? this.el.dep.play : this.el.trn.play;
    btn.textContent = "❚❚"; btn.classList.add("on");
    this.el.loading.hidden = true;
    if (kind === "depth") this.prefetchDepth();
    requestAnimationFrame(t => this.tick(t));
  }
  stop() {
    if (!this.anim) return;
    this.anim = null;
    for (const b of [this.el.dep.play, this.el.trn.play]) { b.textContent = "▶"; b.classList.remove("on"); }
    this.view.setFade(1);
    this.el.interp.hidden = true; this.stage.classList.remove("interp");
    this.writeHash();
  }
  // frames (and the field) for playback unit l are in memory; requests them if not
  unitReady(kind, l) {
    const st = this.st, m = this.m;
    const step = kind === "depth" ? m.steps[st.ci] : m.steps[l], site = kind === "depth" ? l : st.site;
    let ok = !!D.frameNow(step, site);
    if (!ok) D.frame(m, step, site).catch(e => { console.error(e); this.failed = e; });
    if (st.color === "correct" && !D.frameNow(step, this.S - 1)) { ok = false; D.frame(m, step, this.S - 1).catch(() => {}); }
    if (st.color.startsWith("f:")) {
      const f = this.fields.get(st.color.slice(2));
      if (f && f.steps.includes(step) && !this.fieldData.has(`${f.name}_${step}`)) { ok = false; this.loadField(f.name, step).catch(() => {}); }
    }
    return ok;
  }
  tick(ts) {
    const a = this.anim;
    if (!a) return;
    const dt = Math.min(80, ts - a.last); a.last = ts;
    const l = Math.floor(a.u), nxt = Math.min(l + 1, a.end);
    const ok = this.unitReady(a.kind, l) && this.unitReady(a.kind, nxt);
    if (a.kind === "train") for (let k = 2; k <= 4; k++) if (l + k <= a.end) this.unitReady("train", l + k);
    if (ok) { a.u = Math.min(a.end, a.u + dt * SPEED[a.kind]); a.stall = 0; }
    else if ((a.stall = (a.stall || 0) + dt) > 20000 || this.failed) {      // data did not arrive: give up
      this.stop(); this.failed = null;
      this.el.loading.hidden = false; this.el.loading.textContent = "playback stopped: data did not load";
      return;
    }
    const L = Math.floor(a.u), f = Math.max(0, (a.u - L - DWELL) / (1 - DWELL));
    const st = this.st;
    if (a.kind === "depth") { if (st.site !== L) { st.site = L; this.el.dep.input.value = L; } }
    else if (st.ci !== L) { st.ci = L; this.el.trn.input.value = L; this.syncControls(); }
    this.frac = L < a.end ? f : 0;
    this.paint();
    if (a.u >= a.end) { this.frac = 0; this.stop(); this.show(); if (a.kind === "train") this.prefetchDepth(); return; }
    requestAnimationFrame(t => this.tick(t));
  }

  // ------------------------------------------------------------------ painting
  // from-frame, to-frame and fraction for what is on screen
  frames() {
    const st = this.st, m = this.m, step = m.steps[st.ci];
    const from = D.frameNow(step, st.site);
    const f = this.anim ? this.frac || 0 : 0;
    let to = null;
    if (f > 0) to = this.anim.kind === "depth" ? D.frameNow(step, st.site + 1) : D.frameNow(m.steps[st.ci + 1], st.site);
    return { from, to: to || null, f: to ? f : 0, step, site: st.site };
  }
  projector() {
    const st = this.st;
    if (st.proj === "octant") {
      const A = st.a, B = st.b;
      return (s, o, out) => { const a = s[o + A], b = s[o + B]; out[0] = 2 * a; out[1] = 2 * b; out[2] = 2 * Math.sqrt(Math.max(0, 1 - a * a - b * b)); return out; };
    }
    const M = this.m.pca.axes, mu = this.m.pca.mean;
    return (s, o, out) => {
      for (let r = 0; r < 3; r++) { let t = 0; const row = M[r]; for (let c = 0; c < 10; c++) t += row[c] * (2 * s[o + c] - mu[c]); out[r] = t; }
      return out;
    };
  }
  // projected position of row i on the great circle between frames a and b at fraction f
  projAt(a, b, i, f, out) {
    const P = this.projector();
    if (!b || f <= 0) return P(a, i * 10, out);
    return P(slerp(a, i * 10, b, i * 10, f, this.tmp), 0, out);
  }

  paint() {
    if (!this.view) return;
    const fr = this.frames();
    this.cur = fr;
    if (!fr.from) return;
    const { from, to, f } = fr, N = this.N, pos = this.view.pos, P = this.projector(), out = this.p3, tmp = this.tmp;
    for (let i = 0; i < N; i++) {
      if (to) P(slerp(from, i * 10, to, i * 10, f, tmp), 0, out); else P(from, i * 10, out);
      pos[3 * i] = out[0]; pos[3 * i + 1] = out[1]; pos[3 * i + 2] = out[2];
    }
    this.style(fr);
    this.view.setFade(to ? 0.6 : 1);
    this.view.touch("pos"); this.view.touch("style");
    this.renderFrame(fr);
    this.renderRead(fr);
    const key = `${fr.step}_${fr.site}`;
    if (key !== this.selKey) { this.selKey = key; this.renderSel(); this.renderTotals(); if (this.st.color.startsWith("f:")) this.renderLegend(); }
  }

  colourFn(fr) {
    const st = this.st, pal = this.pal, lab = this.lab, s = fr.from, N = this.N;
    if (st.color === "pred") return i => [pal.cls[s.top[i]], false];
    if (st.color === "correct") {
      const last = D.frameNow(fr.step, this.S - 1);
      if (!last) return () => [pal.undef, true];
      return i => [last.top[i] === lab[i] ? pal.correct : pal.wrong, false];
    }
    if (st.color.startsWith("f:")) {
      const fm = this.fields.get(st.color.slice(2)), data = this.fieldData.get(`${fm?.name}_${fr.step}`);
      if (!fm || !data || fr.site >= this.S - 1) return () => [pal.undef, true];
      // codes: 0 below the range, 1..253 bins, 254 above, 255 undefined (build_site.py)
      const ramp = fm.scale === "diverging" ? pal.div : pal.seq, o = fr.site * N,
        lut = Array.from({ length: 255 }, (_, k) => ramp(k === 0 ? 0 : k === 254 ? 1 : (k - 0.5) / 253));
      return i => { const c = data[o + i]; return c === 255 ? [pal.undef, true] : [lut[c], false]; };
    }
    return i => [pal.cls[lab[i]], false];
  }
  style(fr) {
    const st = this.st, v = this.view, N = this.N, lab = this.lab, col = this.colourFn(fr);
    const base = coarse.matches ? 3.0 : 3.4, carried = st.carry ? new Set(st.carry.ids) : null;
    for (let i = 0; i < N; i++) {
      const [c, hollow] = col(i);
      v.col[3 * i] = c[0] / 255; v.col[3 * i + 1] = c[1] / 255; v.col[3 * i + 2] = c[2] / 255;
      v.hollow[i] = hollow ? 1 : 0;
      if (!st.filter.has(lab[i])) { v.size[i] = 0; v.alpha[i] = 0; continue; }
      let a = hollow ? 0.55 : 0.82, sz = hollow ? base + 0.6 : base;
      if (carried) { if (carried.has(i) || i === st.carry.anchor) { sz = base + 3; a = 1; } else a *= 0.22; }
      if (i === st.sel) { sz = base + 4; a = 1; }
      v.size[i] = sz; v.alpha[i] = a;
    }
  }

  renderFrame(fr) {
    const m = this.m, E = this.el;
    E.frame.innerHTML = `<b>${m.sites[fr.site]}</b> <span>step ${fmtInt(fr.step)}</span>`;
    if (fr.to) {
      const a = this.anim.kind === "depth" ? `${m.sites[fr.site]} to ${m.sites[fr.site + 1]}` : `step ${fmtInt(fr.step)} to ${fmtInt(m.steps[this.st.ci + 1])}`;
      E.interp.textContent = `Interpolated, not data: ${Math.round(fr.f * 100)}% of the way from ${a} along great circles. Colours are those of the frame being left.`;
      E.interp.hidden = false; this.stage.classList.add("interp");
    } else { E.interp.hidden = true; this.stage.classList.remove("interp"); }
  }
  renderRead(fr) {
    const st = this.st, m = this.m;
    let shown = 0; for (const c of st.filter) shown += this.counts[c];
    const acc = m.acc[st.ci][fr.site];
    const key = `${fr.step}_${fr.site}_${shown}_${!!fr.to}_${st.color}`;
    if (key === this.readKey) return;
    this.readKey = key;
    const rows = [
      ["Frame", `${m.sites[fr.site]}, step ${fmtInt(fr.step)}${fr.to ? " (moving)" : ""}`],
      ["Lens top-1 accuracy", pct(acc)],
      ["Points shown", `${fmtInt(shown)} of ${fmtInt(this.N)}`],
    ];
    if (st.color.startsWith("f:")) {
      const fm = this.fields.get(st.color.slice(2)), data = this.fieldData.get(`${fm.name}_${fr.step}`);
      if (fr.site >= this.S - 1) rows.push([fm.label, "undefined at the last site (no next step)"]);
      else if (!data) rows.push([fm.label, "not computed at this checkpoint"]);
      else {
        let n = 0, d = 0; const o = fr.site * this.N;
        for (let i = 0; i < this.N; i++) if (st.filter.has(this.lab[i])) { n++; if (data[o + i] !== 255) d++; }
        rows.push([`${fm.label} defined at`, n ? pct(d / n) + " of shown points" : "n/a"]);
      }
    }
    this.el.read.replaceChildren(h("h3", { class: "card-title" }, "This frame"),
      h("dl", { class: "kv" }, rows.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])));
  }

  renderLegend() {
    const st = this.st, pal = this.pal, m = this.m, E = this.el;
    if (!E) return;
    const sw = (c, t, hollow) => h("span", { class: "kk" }, h("i", { class: hollow ? "hollow" : "", style: hollow ? { borderColor: css(c) } : { background: css(c) } }), t);
    let kids;
    if (st.color === "class" || st.color === "pred") {
      kids = [h("span", { class: "kt" }, st.color === "class" ? "True class:" : "Lens prediction at this frame:"), ...m.classes.map((c, i) => sw(pal.cls[i], c))];
    } else if (st.color === "correct") {
      kids = [h("span", { class: "kt" }, "Lens at the last site (b9), this checkpoint:"), sw(pal.correct, "correct"), sw(pal.wrong, "incorrect")];
    } else {
      const fm = this.fields.get(st.color.slice(2));
      const ramp = fm.scale === "diverging" ? pal.div : pal.seq;
      const clamp = fm.vmin < fm.lo || fm.vmax > fm.hi;
      kids = [h("span", { class: "kt", title: fm.description || "" }, fm.units ? `${fm.label} (${fm.units}):` : `${fm.label}:`),
        h("span", { class: "kbar-wrap" }, h("span", { class: "kt" }, sig(fm.lo)), h("i", { class: "kbar", style: { background: gradient(ramp) } }), h("span", { class: "kt" }, sig(fm.hi))),
        sw(pal.undef, "undefined", true),
        h("span", { class: "kt kfull" }, `Points at site l show the step from l to l + 1; the last site has none. Same colour range at every frame${fm.scale === "diverging" ? ", centred on zero" : ""}${clamp ? `; values beyond it (data range ${sig(fm.vmin)} to ${sig(fm.vmax)}) take the end colours` : ""}.`), ...(fm.defined < 0.5 ? [h("span", { class: "kt kfull sparse" }, `Released on few images: ${pct(fm.defined)} on average over steps and checkpoints (${this.definedHere(fm)} here); the rest are grey rings.`)] : [h("span", { class: "kt" }, `Defined on ${pct(fm.defined)} of points on average.`)])];
    }
    E.legend.replaceChildren(...kids);
  }
  // share of images where the field is defined at the frame on screen
  definedHere(fm) {
    const fr = this.cur, data = fr && this.fieldData.get(`${fm.name}_${fr.step}`);
    if (!data || fr.site >= this.S - 1) return "none at this frame";
    let d = 0; const o = fr.site * this.N;
    for (let i = 0; i < this.N; i++) if (data[o + i] !== 255) d++;
    return pct(d / this.N) + " at this frame";
  }
  renderNote() {
    const st = this.st, m = this.m, E = this.el;
    if (st.proj === "octant") {
      E.note.innerHTML = `<b>Fisher octant.</b> Each point is (2√p<sub>A</sub>, 2√p<sub>B</sub>, 2√p<sub>rest</sub>), with p<sub>rest</sub> the total probability of the other eight classes. This lies on an eighth of the sphere of radius 2, and distance along that surface is exactly the Fisher–Rao distance between the grouped three-way distributions. Grouping classes can only shrink distances, so points that look close here may be far apart in the full ten-class space. Faint arcs: 25%, 50% and 75% probability of A, of B and of the rest.`;
    } else {
      const ex = m.pca.explained, tot = ex.reduce((a, b) => a + b, 0);
      E.note.innerHTML = `<b>Principal axes.</b> A fixed linear projection of X = 2√p ∈ ℝ<sup>10</sup> onto the three leading principal directions of the points pooled over all ten sites at ${m.pca.steps.length} checkpoints (centred; ${pct(ex[0])}, ${pct(ex[1])} and ${pct(ex[2])} of the pooled variance, ${pct(tot)} together). Labelled dots are the ten one-hot distributions (class vertices); faint curves are the great-circle edges between them. Distances in this picture are not Fisher–Rao distances.`;
    }
  }

  // the totals card: a short fixed selection of per-step totals (build_site.py TOTALS), one view at a time,
  // on a y-range fixed across checkpoints
  renderTotals() {
    const E = this.el, m = this.m, st = this.st;
    if (!m.totals) return;
    D.totals(m).then(T => {
      if (!T || !T.views || !T.views.length) return;
      E.totals.hidden = false;
      const V = T.views.find(v => v.id === this.totView) || T.views[0];
      this.totView = V.id;
      const si = T.steps.indexOf(m.steps[st.ci]);
      const sel = h("select", { "aria-label": "Total shown" }, T.views.map(v => h("option", { value: v.id, selected: v.id === V.id }, v.label)));
      sel.addEventListener("change", () => { this.totView = sel.value; this.renderTotals(); });
      const host = h("div", { class: "tot-chart" });
      let note = null;
      E.totals.replaceChildren(h("h3", { class: "card-title" }, `Totals per depth step · step ${fmtInt(m.steps[st.ci])}`), sel, host,
        h("p", { class: "card-note" }, V.reading), (note = h("p", { class: "card-note" })));
      if (si < 0) { host.replaceChildren(h("p", { class: "card-note" }, "Not measured at this checkpoint.")); return; }
      const all = V.series.flatMap(s => s.value.flat()).filter(Number.isFinite);
      const lo = Math.min(0, ...all), hi = Math.max(...all);
      const rng = V.range ? [Math.min(V.range[0], lo), Math.max(V.range[1], hi)] : [lo, hi * 1.05];
      const cols = ["#9a9993", "#2a78d6", "#eb6834"], cols2 = ["#2a78d6", "#eb6834", "#9a9993"];
      const kind = V.kind === "stack" ? "stack" : V.kind === "lines" ? "line" : "bar";
      const series = V.series.map((s, k) => ({ name: s.name, values: s.value[si], color: V.series.length === 1 ? "#2a78d6" : (kind === "stack" ? cols : cols2)[k], kind,
        dash: kind === "line" && k ? (k === 1 ? "5 3" : "2 3") : "", tfmt: v => sig(v) }));
      const draw = () => XC.chart(host, { height: 170, x: { type: "band", labels: m.sites.slice(0, this.S - 1) }, y: { domain: rng },
        series, legend: V.series.length > 1, marker: st.site < this.S - 1 ? st.site : null });
      if (XC.chart) draw(); else XC.ready.then(draw);
      const mass = Math.min(...V.series.flatMap(s => s.mass[si]).filter(Number.isFinite));
      note.textContent = `Bar at site l: the step from l to l + 1 (dashed line: this site). Defined on ${mass >= 0.9995 ? "all" : "at least " + pct(mass)} images at every step of this checkpoint.`;
    });
  }

  // ------------------------------------------------------------------ selection, trails, carry forward
  select(i) {
    const st = this.st;
    if (i === st.sel && i >= 0) return;
    st.sel = i;
    if (i < 0) st.carry = null;
    else if (st.carry && st.carry.anchor !== i) st.carry = null;
    this.show(); this.writeHash();
    if (i >= 0) D.sprite(this.m).catch(() => {});
  }
  setHover(i, x, y) {
    const tip = document.getElementById("tooltip");
    if (i !== this.hover) { this.hover = i; this.view.render(); }
    if (i < 0) { tip.hidden = true; return; }
    const fr = this.cur; if (!fr || !fr.from) return;
    const m = this.m, s = fr.from;
    const b = s.top[i], bv = s[i * 10 + b];
    tip.replaceChildren(h("div", { class: "tt-img" }, this.thumb(i),
      h("div", {}, h("div", { class: "tt-head" }, `#${i} · ${m.classes[this.lab[i]]}`),
        h("div", {}, `lens at ${m.sites[fr.site]}: ${m.classes[b]} (${pct(bv * bv)})`),
        h("div", { class: "mute" }, "click to select"))));
    tip.hidden = false;
    const r = tip.getBoundingClientRect();
    tip.style.left = Math.min(window.innerWidth - r.width - 8, x + 14) + "px";
    tip.style.top = Math.min(window.innerHeight - r.height - 8, y + 14) + "px";
  }
  thumb(i, cls = "") {
    const cv = h("canvas", { width: 32, height: 32, class: cls });
    const m = this.m, cols = m.sprite.cols, px = m.sprite.px;
    D.sprite(m).then(im => cv.getContext("2d").drawImage(im, (i % cols) * px, Math.floor(i / cols) * px, px, px, 0, 0, 32, 32)).catch(() => {});
    return cv;
  }

  renderSel() {
    const E = this.el, st = this.st, m = this.m, i = st.sel, pal = this.pal;
    if (i < 0) {
      E.sel.replaceChildren(h("h3", { class: "card-title" }, "Selected image"),
        h("p", { class: "card-note" }, "Click a point to see its image, its lens probabilities at this frame, and its trail through depth or training."));
      return;
    }
    const fr = this.cur || this.frames(); if (!fr.from) return;
    const s = fr.from, p = Array.from({ length: 10 }, (_, c) => s[i * 10 + c] ** 2);
    const pred = s.top[i], y = this.lab[i];
    const bars = h("div", { class: "pbars" }, m.classes.map((c, k) => h("div", { class: "prow" + (k === y ? " true" : "") },
      h("span", { class: "pn" }, c), h("span", { class: "pb" }, h("i", { style: { width: `${100 * p[k]}%`, background: css(pal.cls[k]) } })), h("span", { class: "pv" }, p[k] < 0.001 ? "<0.1%" : pct(p[k])))));
    let fieldLine = null;
    if (st.color.startsWith("f:")) {
      const fm = this.fields.get(st.color.slice(2)), data = this.fieldData.get(`${fm.name}_${fr.step}`);
      let txt = "undefined";
      if (fr.site >= this.S - 1) txt = "no step after the last site";
      else if (!data) txt = "not computed at this checkpoint";
      else {
        const c = data[fr.site * this.N + i];
        // the code is a bin, so show the interval it stands for
        const w = (fm.hi - fm.lo) / 253;
        if (c === 0) txt = `below ${sig(fm.lo)}`;
        else if (c === 254) txt = `above ${sig(fm.hi)}`;
        else if (c !== 255) txt = `${sig(fm.lo + (c - 1) * w)} to ${sig(fm.lo + c * w)}`;
      }
      fieldLine = h("div", { class: "kvline" }, `${fm.label}, ${m.sites[fr.site]} → ${m.sites[Math.min(fr.site + 1, this.S - 1)]}: `, h("b", {}, txt));
    }
    const trail = h("div", { class: "seg seg-sm", role: "radiogroup", "aria-label": "Trail" },
      [["depth", "Through depth"], ["train", "Through training"], ["off", "Off"]].map(([v, t]) =>
        h("button", { type: "button", role: "radio", "aria-checked": String(st.trail === v), onclick: () => { st.trail = v; this.show(); this.writeHash(); } }, t)));
    this.trailNote = h("div", { class: "card-note" }, st.trail === "depth" ? `The image's positions at the ten sites, step ${fmtInt(fr.step)}.` :
      st.trail === "train" ? `The image's positions at ${m.sites[fr.site]} over the ${m.steps.length} included checkpoints.` : "");
    this.carryBox = h("div", { class: "carry" });
    E.sel.replaceChildren(
      h("div", { class: "sel-head" }, this.thumb(i, "big"),
        h("div", {}, h("div", { class: "sel-id" }, `Image #${i}`, h("button", { type: "button", class: "x-btn", "aria-label": "Clear selection", onclick: () => this.select(-1) }, "×")),
          h("div", {}, "True class: ", h("b", {}, m.classes[y])),
          h("div", {}, `Lens at ${m.sites[fr.site]}, step ${fmtInt(fr.step)}: `, h("b", {}, m.classes[pred]), ` (${pct(p[pred])})`))),
      ...[fieldLine].filter(Boolean), h("div", { class: "sub-h" }, "Lens probabilities at this frame"), bars,
      h("div", { class: "sub-h" }, "Trail"), trail, this.trailNote, this.carryBox);
    this.renderCarry();
  }

  startCarry() {
    const st = this.st, fr = this.frames();
    if (st.sel < 0 || !fr.from) return;
    st.carry = { anchor: st.sel, ids: knn(fr.from, st.sel, K, this.N), step: fr.step, site: fr.site };
    this.carryCache = new Map();
    this.show();
  }
  retained(step, site) {
    const c = this.st.carry, key = `${step}_${site}`;
    if (this.carryCache.has(key)) return this.carryCache.get(key);
    const s = D.frameNow(step, site);
    if (!s) return null;
    const now = new Set(knn(s, c.anchor, K, this.N));
    const r = c.ids.filter(j => now.has(j)).length;
    this.carryCache.set(key, r);
    return r;
  }
  renderCarry() {
    const box = this.carryBox, st = this.st, m = this.m;
    if (!box) return;
    const fr = this.cur; if (!fr || !fr.from) return;
    if (!st.carry) {
      box.replaceChildren(h("div", { class: "sub-h" }, "Neighbours"),
        h("button", { type: "button", class: "act-btn", onclick: () => this.startCarry() }, `Carry forward its ${K} nearest neighbours`),
        h("div", { class: "card-note" }, `Finds the ${K} images nearest to it at this frame (Fisher–Rao distance over all ten classes, not distance in the picture) and keeps them highlighted as you change site or checkpoint. Computed in the browser from the site's 16-bit values; in spot checks on eight frames the set matched the one from full-precision data for 90–100% of images.`));
      return;
    }
    const c = st.carry, r = this.retained(fr.step, fr.site);
    const bars = m.sites.map((sn, l) => {
      const v = this.retained(fr.step, l);
      return h("div", { class: "tbar" + (l === fr.site ? " on" : "") + (fr.step === c.step && l === c.site ? " origin" : ""), title: `${sn}: ${v ?? "?"} of ${K} still among its ${K} nearest` },
        h("i", { style: { height: `${v == null ? 0 : 100 * v / K}%` } }), h("span", {}, sn));
    });
    const d = c.ids.map(j => fisher(fr.from, c.anchor, j)).sort((a, b) => a - b);
    box.replaceChildren(h("div", { class: "sub-h" }, "Neighbours carried forward"),
      h("div", { class: "card-sub" }, `Chosen at ${m.sites[c.site]}, step ${fmtInt(c.step)}. At this frame `, h("b", {}, r == null ? "…" : `${r} of ${K}`), ` are still among its ${K} nearest.`),
      h("div", { class: "tbars small" }, bars),
      h("div", { class: "card-note" }, `Bars: how many of the ${K} are among its ${K} nearest at each site, step ${fmtInt(fr.step)}${fr.step === c.step ? " (outlined: where they were chosen)" : ""}. Their Fisher–Rao distances to it here: median ${sig(d[K >> 1])}, largest ${sig(d[K - 1])}.`),
      h("div", { class: "imgs" }, c.ids.map(j => { const t = this.thumb(j); t.title = `#${j} · ${m.classes[this.lab[j]]}`; t.addEventListener("click", () => this.select(j)); return t; })),
      h("button", { type: "button", class: "link-btn", onclick: () => { st.carry = null; this.show(); } }, "Clear neighbours"));
  }
  loadTrainTrail() {
    const st = this.st, m = this.m, site = st.site;
    let done = 0;
    const ps = m.steps.map(step => D.frame(m, step, site).then(() => { done++; if (this.trailNote && st.trail === "train" && done < m.steps.length) this.trailNote.textContent = `Loading checkpoints for the training trail: ${done} of ${m.steps.length}…`; }));
    Promise.all(ps).then(() => {
      if (st.trail !== "train" || st.site !== site) return;
      if (this.trailNote) this.trailNote.textContent = `The image's positions at ${m.sites[site]} over the ${m.steps.length} included checkpoints; the ring marks this checkpoint.`;
      this.view.render();
    }).catch(e => console.error(e));
  }

  // ------------------------------------------------------------------ guides and overlay
  setGuides(refit) {
    const st = this.st, pal = this.pal, v = this.view, segs = [], lines = (arr, color, opacity) => ({ kind: "lines", segs: new Float32Array(arr), color, opacity });
    const arc = (f, n, out) => { let prev = f(0); for (let k = 1; k <= n; k++) { const q = f(k / n); out.push(...prev, ...q); prev = q; } };
    if (st.proj === "octant") {
      const n = 28, pos = [], idx = [];
      for (let a = 0; a <= n; a++) for (let b = 0; b <= n; b++) {
        const th = a / n * Math.PI / 2, ph = b / n * Math.PI / 2;
        pos.push(2 * Math.sin(th) * Math.cos(ph), 2 * Math.sin(th) * Math.sin(ph), 2 * Math.cos(th));
      }
      for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) { const k = a * (n + 1) + b; idx.push(k, k + n + 1, k + 1, k + 1, k + n + 1, k + n + 2); }
      const edge = [], iso = [];
      for (let g = 0; g < 3; g++) {
        const o1 = (g + 1) % 3, o2 = (g + 2) % 3;
        for (const p of [0, 0.25, 0.5, 0.75]) {
          const r = 2 * Math.sqrt(1 - p);
          arc(t => { const q = [0, 0, 0]; q[g] = 2 * Math.sqrt(p); q[o1] = r * Math.cos(t * Math.PI / 2); q[o2] = r * Math.sin(t * Math.PI / 2); return q; }, 40, p === 0 ? edge : iso);
        }
      }
      v.setGuides([{ kind: "mesh", pos: new Float32Array(pos), index: new Uint16Array(idx), color: pal.guide, opacity: pal.dark ? 0.1 : 0.07 },
        lines(iso, pal.guide, pal.dark ? 0.32 : 0.28), lines(edge, pal.guide, 0.75)]);
      if (refit) v.fit([2 / 3, 2 / 3, 2 / 3], [1, 1, 1], [0, 0, 1], v.w < 520 ? 2.15 : 1.8);
    } else {
      const P = this.projector(), e = new Float64Array(10), out = [0, 0, 0];
      let R = 0;
      this.vertices = [];
      for (let i = 0; i < 10; i++) { e.fill(0); e[i] = 1; const q = Array.from(P(e, 0, out)); this.vertices.push(q); R = Math.max(R, Math.hypot(...q)); }
      for (let i = 0; i < 10; i++) for (let j = i + 1; j < 10; j++) {
        arc(t => { e.fill(0); e[i] = Math.cos(t * Math.PI / 2); e[j] = Math.sin(t * Math.PI / 2); return Array.from(P(e, 0, out)); }, 24, segs);
      }
      v.setGuides([lines(segs, pal.guide, pal.dark ? 0.34 : 0.3)]);
      if (refit) v.fit([0, 0, 0], [0, 0, 1], [0, 1, 0], R * 1.12);
    }
    this.renderNote();
  }

  overlay(ctx) {
    const st = this.st, pal = this.pal, v = this.view, m = this.m, fr = this.cur;
    const font = (w, s) => `${w} ${s}px Inter, system-ui, sans-serif`;
    // labels are nudged up or down until they no longer overlap one already drawn
    const placed = [];
    const label = (t, x, y, color, { size = 12, weight = 600, align = "center" } = {}) => {
      ctx.font = font(weight, size);
      const w = ctx.measureText(t).width + 4, hh = size + 2, x0 = align === "left" ? x - 2 : x - w / 2;
      let yy = y;
      for (const dy of [0, -1, 1, -2, 2, -3, 3]) {
        yy = y + dy * hh;
        if (!placed.some(r => x0 < r[0] + r[2] && r[0] < x0 + w && yy - hh / 2 < r[1] + r[3] && r[1] < yy + hh / 2)) break;
      }
      placed.push([x0, yy - hh / 2, w, hh]);
      ctx.textAlign = align; ctx.textBaseline = "middle";
      ctx.lineJoin = "round"; ctx.lineWidth = 3.5; ctx.strokeStyle = css(pal.bg, 0.9); ctx.strokeText(t, x, yy);
      ctx.fillStyle = color; ctx.fillText(t, x, yy);
    };
    const S = (q) => v.toScreen(q[0], q[1], q[2]);
    const cx = v.w / 2, cy = v.h / 2;
    const outward = (p, d) => { const dx = p[0] - cx, dy = p[1] - cy, n = Math.hypot(dx, dy) || 1; return [p[0] + dx / n * d, p[1] + dy / n * d]; };
    // landmarks
    const u = new Float64Array(10).fill(Math.sqrt(0.1)), P = this.projector(), o3 = [0, 0, 0];
    const uni = S(P(u, 0, o3));
    const dot = (p, r) => placed.push([p[0] - r, p[1] - r, 2 * r, 2 * r]);
    const verts = st.proj === "octant"
      ? [[[2, 0, 0], `A: ${m.classes[st.a]}`, pal.cls[st.a]], [[0, 2, 0], `B: ${m.classes[st.b]}`, pal.cls[st.b]], [[0, 0, 2], "rest (other 8 classes)", pal.fg]]
      : (this.vertices || []).map((q, k) => [q, m.classes[k], pal.cls[k]]);
    const vp = verts.map(([q]) => S(q));
    vp.forEach((p, k) => {
      ctx.fillStyle = css(verts[k][2]); ctx.strokeStyle = css(pal.bg); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(p[0], p[1], 5, 0, 7); ctx.fill(); ctx.stroke(); dot(p, 5);
    });
    dot(uni, 5);
    vp.forEach((p, k) => { const [x, y] = outward(p, st.proj === "octant" ? 18 : 14); label(verts[k][1], x, y, css(verts[k][2]), { size: st.proj === "octant" ? 13 : 12 }); });
    ctx.strokeStyle = css(pal.fg, 0.75); ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.moveTo(uni[0] - 4, uni[1] - 4); ctx.lineTo(uni[0] + 4, uni[1] + 4); ctx.moveTo(uni[0] + 4, uni[1] - 4); ctx.lineTo(uni[0] - 4, uni[1] + 4); ctx.stroke();
    label("uniform", uni[0] + 7, uni[1] - 9, css(pal.fg, 0.8), { size: 11, weight: 500, align: "left" });
    if (!fr || !fr.from) return;
    const out = [0, 0, 0];
    // a projected great-circle arc between rows ia of a and ib of b, as screen points
    const arcPts = (a, ia, b, ib) => { const pts = []; for (let k = 0; k <= SAMPLES; k++) { P(slerp(a, ia * 10, b, ib * 10, k / SAMPLES, this.tmp), 0, out); pts.push(S(out)); } return pts; };
    const poly = pts => { ctx.beginPath(); ctx.moveTo(...pts[0]); for (const q of pts.slice(1)) ctx.lineTo(...q); ctx.stroke(); };
    const head = (pts, color, sz = 6) => {
      const [x1, y1] = pts[pts.length - 1]; let k = pts.length - 2; while (k > 0 && Math.hypot(pts[k][0] - x1, pts[k][1] - y1) < 2) k--;
      const [x0, y0] = pts[k], a = Math.atan2(y1 - y0, x1 - x0);
      if (Math.hypot(x1 - x0, y1 - y0) < 0.5) return;
      ctx.fillStyle = color; ctx.beginPath(); ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - sz * Math.cos(a - 0.45), y1 - sz * Math.sin(a - 0.45)); ctx.lineTo(x1 - sz * Math.cos(a + 0.45), y1 - sz * Math.sin(a + 0.45)); ctx.fill();
    };
    // arrows: displacement to the next site (same checkpoint), along the projected great circle
    const nextSite = fr.site + 1;
    if (st.arrows && nextSite < this.S) {
      const b = D.frameNow(fr.step, nextSite);
      if (b) {
        let ids;
        if (st.sel >= 0) ids = [st.sel, ...(st.carry ? st.carry.ids : [])];
        else { ids = []; for (let i = 0; i < this.N; i += ARROW_EVERY) if (st.filter.has(this.lab[i])) ids.push(i); }
        ctx.lineWidth = ids.length > 50 ? 1 : 1.6;
        for (const i of ids) {
          const c = css([v.col[3 * i] * 255, v.col[3 * i + 1] * 255, v.col[3 * i + 2] * 255], ids.length > 50 ? 0.7 : 0.95);
          const pts = arcPts(fr.from, i, b, i); ctx.strokeStyle = c; poly(pts); head(pts, c, ids.length > 50 ? 5 : 7);
        }
      }
    }
    // trail of the selected image (line), rings, then the trail's marks and labels on top
    const i = st.sel, pts = [], marks = [];
    if (i >= 0 && st.trail !== "off") {
      if (st.trail === "depth") {
        const fs = m.sites.map((_, l) => D.frameNow(fr.step, l));
        if (fs.every(Boolean)) for (let l = 0; l < this.S; l++) {
          const show = l === 0 || l === this.S - 1 || l === fr.site;
          marks.push([S(P(fs[l], i * 10, out)), show ? m.sites[l] : "", l === fr.site]);
          if (l < this.S - 1) { const seg = arcPts(fs[l], i, fs[l + 1], i); pts.push(...(l ? seg.slice(1) : seg)); }
        }
      } else {
        const fs = m.steps.map(s => D.frameNow(s, fr.site));
        if (fs.every(Boolean)) for (let k = 0; k < fs.length; k++) {
          marks.push([S(P(fs[k], i * 10, out)), k === 0 || k === fs.length - 1 ? `step ${short(m.steps[k])}` : "", k === st.ci]);
          if (k < fs.length - 1) { const seg = arcPts(fs[k], i, fs[k + 1], i); pts.push(...(k ? seg.slice(1) : seg)); }
        }
      }
    }
    if (pts.length) {
      ctx.strokeStyle = css(pal.bg, 0.85); ctx.lineWidth = 4.5; poly(pts);
      ctx.strokeStyle = css(pal.fg, 0.9); ctx.lineWidth = 1.8; poly(pts);
    }
    const ring = (j, r, color, w) => {
      const q = S([v.pos[3 * j], v.pos[3 * j + 1], v.pos[3 * j + 2]]);
      ctx.strokeStyle = css(pal.bg, 0.9); ctx.lineWidth = w + 2.5; ctx.beginPath(); ctx.arc(q[0], q[1], r, 0, 7); ctx.stroke();
      ctx.strokeStyle = color; ctx.lineWidth = w; ctx.beginPath(); ctx.arc(q[0], q[1], r, 0, 7); ctx.stroke();
    };
    if (st.carry) for (const j of st.carry.ids) if (st.filter.has(this.lab[j])) ring(j, 5, css(pal.accent, 0.9), 1.3);
    if (i >= 0) ring(i, 8, css(pal.accent), 2.2);
    if (this.hover >= 0 && this.hover !== i) ring(this.hover, 6, css(pal.fg), 1.5);
    for (const [p, t, on] of marks) {
      ctx.fillStyle = css(on ? pal.accent : pal.fg); ctx.beginPath(); ctx.arc(p[0], p[1], on ? 4 : 2.6, 0, 7); ctx.fill();
      if (t) label(t, p[0] + 9, p[1] - 9, css(pal.fg), { size: 11, weight: 600, align: "left" });
    }
  }
}

const HELP = `
<p><b>What a point is.</b> Each point is one of the 10,000 CIFAR-10 test images. Its position is the network's
class distribution p for that image, read out at the chosen site by the logit lens (the network's own pooling and
final layer applied to the residual stream along the skip path), and drawn as X = 2√p.</p>
<p><b>Where the corners are.</b> In the <b>Fisher octant</b> the three labelled corners are certainty: a point at
corner A means the lens puts all probability on class A; at "rest", all of it on the other eight classes. The cross
marks the uniform distribution. Distance along the curved surface is the Fisher–Rao distance between the grouped
distributions, which is never larger than the distance between the full ten-class distributions. In <b>Principal
axes</b> the ten labelled dots are the ten class corners, seen through a fixed linear projection of all ten
dimensions; distances in that picture are only indicative.</p>
<p><b>Moving.</b> The view is fixed: only the points move. <b>Depth</b> moves through the network at one training
checkpoint; <b>Training</b> moves through checkpoints at one site. ▶ plays. Between two frames the points slide along
great circles of the sphere; such in-between positions are not data, and are shown faded inside a dashed frame.
<b>Arrows</b> show where each point goes at the next site. Drag to rotate, scroll or pinch to zoom; click a point for
its image, probabilities and trail. "Carry forward" marks its 20 nearest images (Fisher–Rao distance over all ten
classes) and follows them as you move.</p>
<p><b>Colouring by a measured field.</b> Every field describes the step from the site shown to the next one, so the
last site has none. <b>Motion</b> fields (speed, the part not explained by one rotation of the whole cloud, path length,
movement toward the true class) are defined on every image. <b>Neighbourhood</b> and <b>relative-motion</b> fields are
measured over each image's neighbours (k = 256), at the neighbourhood radius rather than at a point. At the resolution
10,000 images give, no stable thin directions are resolved at any site, so there is no split into motion along and
across the cloud; <b>strain</b> fields are statistics of a local linear fit and are released only where calibrated
gates pass, which is on few images (grey rings elsewhere).
The <b>totals</b> card summarises each depth step; the Experiments tab explains every quantity.</p>`;
