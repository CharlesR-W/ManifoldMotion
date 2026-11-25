// The explorer: the 10,000 test images as points in a site's raw activation space, drawn on the first three of
// a set of axes fixed per stage, moved through depth (site to site) and training (checkpoint to checkpoint).
import * as D from "./data.js";
import { palette, css, gradient } from "./colors.js";
// the experiment pages' chart module, shared so the side charts read like the figures there
const XC = { ready: import("./xcharts.js").then(mod => Object.assign(XC, mod)) };

const ARROW_EVERY = 16;                            // arrows for every 16th shown image when nothing is selected
const DWELL = 0.3;                                 // share of each playback unit spent resting on the data frame
const SPEED = { depth: 1 / 1400, train: 1 / 800 }; // playback units per ms
const BREAK_MS = 2800;                             // how long the "axes change" notice stays up
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
const pct = v => v == null || !Number.isFinite(v) ? "n/a" : (100 * v).toFixed(v < 0.001 ? 2 : 1) + "%";
const short = n => n >= 1000 ? (n % 1000 === 0 ? n / 1000 : (n / 1000).toFixed(1)) + "k" : String(n);
const sig = v => v == null || !Number.isFinite(v) ? "n/a" : v === 0 ? "0" : Math.abs(v) >= 100 ? v.toFixed(0)
  : Math.abs(v) >= 1 ? v.toFixed(2) : Math.abs(v) >= 1e-3 ? v.toPrecision(2) : v.toExponential(1);
const arrow = s => s.replace("->", " → ");

function webgl2() {
  try {
    const c = document.createElement("canvas"), g = c.getContext("webgl2");
    if (!g) return false;
    g.getExtension("WEBGL_lose_context")?.loseContext();
    return true;
  } catch (e) { return false; }
}

export class Explorer {
  constructor(root, params) {
    this.root = root;
    this.ready = this.init(params).catch(e => {
      console.error(e);
      root.replaceChildren(h("p", { class: "placeholder", text: `The explorer could not load its data: ${e.message}` }));
    });
  }

  async init(params) {
    const [m, lab, pred, T] = await Promise.all([D.meta(), D.labels(), D.pred(), D.totals()]);
    this.m = m; this.lab = lab; this.pred = pred; this.T = T; this.N = m.N; this.S = m.sites.length;
    this.fields = new Map(m.fields.map(f => [f.id, f]));
    this.stageOf = m.sites.map((_, l) => D.stageOf(m, l));
    this.pal = palette(dark.matches);
    this.st = { ci: m.ckpts.length - 1, site: 0, color: "s.class", arrows: "off", sel: -1, trail: "depth",
      filter: new Set(m.classes.map((_, c) => c)) };
    this.readParams(params);
    this.anim = null; this.hover = -1; this.disp = null; this.shownStage = -1;
    this.build();
    await this.makeView();
    this.resetView();
    this.wire();
    await this.show();
    this.prefetchDepth();
    dark.addEventListener("change", () => { this.pal = palette(dark.matches); this.styleChips(); this.renderLegend(); this.renderSel(); this.paint(); });
  }

  async makeView() {
    if (webgl2()) {
      try {
        const { View } = await import("./view3d.js");
        this.view = new View(this.stage, this.N, ctx => this.overlay(ctx));
      } catch (e) { console.warn("WebGL view failed; using the flat view instead:", e); this.view = null; }
    }
    if (!this.view) {
      const { View2D } = await import("./view2d.js");
      this.view = new View2D(this.stage, this.N, ctx => this.overlay(ctx));
      this.el.flat.hidden = false;
      this.el.reset.title = "Return to the starting zoom";
    }
  }

  // ------------------------------------------------------------------ state <-> URL
  readParams(q) {
    if (!q) return;
    const st = this.st, m = this.m, int = (k, lo, hi) => { const v = parseInt(q.get(k), 10); return Number.isFinite(v) && v >= lo && v <= hi ? v : null; };
    const ck = int("ckpt", 0, Infinity); if (ck != null && m.ckpts.includes(ck)) st.ci = m.ckpts.indexOf(ck);
    const site = int("site", 0, this.S - 1); if (site != null) st.site = site;
    const col = q.get("color"); if (col && (["s.class", "s.pred", "s.correct"].includes(col) || this.fields.has(col))) st.color = col;
    const sel = int("sel", 0, this.N - 1); if (sel != null) st.sel = sel;
    if (["depth", "train", "off"].includes(q.get("trail"))) st.trail = q.get("trail");
    if (["off", "site", "ckpt"].includes(q.get("arrows"))) st.arrows = q.get("arrows");
  }
  applyParams(q) { this.stop(); this.readParams(q); this.syncControls(); this.show(); }
  writeHash() {
    clearTimeout(this.hashT);
    this.hashT = setTimeout(() => {
      const st = this.st, q = new URLSearchParams({ ckpt: this.m.ckpts[st.ci], site: st.site });
      if (st.color !== "s.class") q.set("color", st.color);
      if (st.arrows !== "off") q.set("arrows", st.arrows);
      if (st.sel >= 0) { q.set("sel", st.sel); if (st.trail !== "depth") q.set("trail", st.trail); }
      const hash = "#explorer?" + q.toString();
      if (location.hash.startsWith("#explorer")) { history.replaceState(null, "", hash); this.lastHash = hash; }
    }, 250);
  }

  // ------------------------------------------------------------------ DOM
  build() {
    const m = this.m, E = (this.el = {});
    E.color = h("select", { "aria-label": "Colour points by" });
    E.arrows = h("select", { "aria-label": "Arrows" },
      h("option", { value: "off" }, "off"), h("option", { value: "site" }, "to the next site"), h("option", { value: "ckpt" }, "to the next checkpoint"));
    E.helpBtn = h("button", { type: "button", class: "guide-btn", "aria-expanded": "false", "aria-controls": "ex-help" }, "How to read this");
    E.help = h("div", { class: "ex-help", id: "ex-help", hidden: true, html: HELP });
    const bar = h("div", { class: "ex-bar" },
      h("div", { class: "ctl" }, h("span", { class: "ctl-label" }, "Colour"), E.color),
      h("div", { class: "ctl" }, h("span", { class: "ctl-label" }, "Arrows"), E.arrows),
      h("div", { class: "ctl ctl-end" }, E.helpBtn));

    E.frame = h("div", { class: "ex-frame" });
    E.interp = h("div", { class: "ex-interp", hidden: true });
    E.brk = h("div", { class: "ex-break", hidden: true, role: "status" });
    E.reset = h("button", { type: "button", class: "mini-btn", title: "Return to the starting view" }, "Reset view");
    E.flat = h("div", { class: "ex-flat", hidden: true }, "WebGL is not available in this browser, so the cloud is drawn flat: axes 1 and 2 only, no rotation. Scroll to zoom, drag to pan.");
    E.loading = h("div", { class: "ex-loadnote", hidden: true }, "loading…");
    this.stage = h("div", { class: "ex-stage" },
      h("canvas", { class: "gl", "aria-label": "View of the image cloud; drag to rotate, scroll or pinch to zoom, click a point to select it" }),
      h("canvas", { class: "ov", "aria-hidden": "true" }), E.frame, E.interp, E.brk,
      h("div", { class: "ex-stage-tools" }, E.reset, E.flat), E.loading);

    const slider = (max, label) => {
      const play = h("button", { type: "button", class: "play", "aria-label": `Play through ${label.toLowerCase()}` }, "▶");
      const input = h("input", { type: "range", min: 0, max, step: 1, "aria-label": label });
      const ticks = h("div", { class: "sl-ticks", "aria-hidden": "true" });
      return { play, input, ticks, row: h("div", { class: "sl-row" }, play, h("span", { class: "sl-label" }, label), h("div", { class: "sl-track" }, input, ticks)) };
    };
    E.dep = slider(this.S - 1, "Depth");
    E.trn = slider(m.ckpts.length - 1, "Training");
    const at = (i, n) => `calc(7px + (100% - 14px) * ${i / Math.max(1, n)})`;
    m.sites.forEach((s, i) => E.dep.ticks.append(h("span", { style: { left: at(i, this.S - 1) } }, s)));
    m.stages.slice(1).forEach(sg => E.dep.ticks.append(h("i", { class: "sl-stage", title: "stage change: the axes change here", style: { left: at(sg.sites[0] - 0.5, this.S - 1) } })));
    const lastCk = m.ckpts[m.ckpts.length - 1];
    const tickCk = [0, 16, 256, 2560, 25600, lastCk].filter(s => m.ckpts.includes(s));
    if (m.ckpts.length === 1) tickCk.splice(0, tickCk.length, m.ckpts[0]);
    for (const s of tickCk) E.trn.ticks.append(h("span", { style: { left: at(m.ckpts.indexOf(s), m.ckpts.length - 1) } }, short(s)));
    E.legend = h("div", { class: "ex-legend" });
    E.coloring = h("p", { class: "ex-coloring" });

    E.read = h("div", { class: "card" });
    E.sel = h("div", { class: "card card-sel" });
    E.charts = h("div", { class: "card" });
    E.filter = h("div", { class: "card" });
    const ckNote = m.ckpts.length > 1
      ? `Training slider: the ${m.ckpts.length} checkpoints with data, evenly spaced by index (not by step).`
      : `Training slider: only checkpoint ${fmtInt(m.ckpts[0])} has data so far; the others are added when they are measured.`;
    const main = h("div", { class: "ex-main" }, this.stage, E.coloring, E.legend,
      h("div", { class: "ex-sliders" }, E.dep.row, E.trn.row, h("p", { class: "sl-note" }, ckNote)));
    const side = h("aside", { class: "ex-side" }, E.read, E.sel, E.charts, E.filter);
    this.root.replaceChildren(bar, E.help, h("div", { class: "ex-body" }, main, side));
    this.buildFilter();
    this.syncControls();
  }

  toggleHelp(open) {
    const E = this.el;
    open = open ?? E.help.hidden;
    E.help.hidden = !open;
    E.helpBtn.setAttribute("aria-expanded", String(open));
    if (!open) try { localStorage.setItem("mm-help-seen-euclid", "1"); } catch (e) { /* storage unavailable */ }
  }

  syncControls() {
    const E = this.el, st = this.st, m = this.m, ck = m.ckpts[st.ci];
    const opt = (v, t, dis) => h("option", { value: v, disabled: dis }, t);
    const groups = [h("optgroup", { label: "Image (final checkpoint)" },
      opt("s.class", "True class"), opt("s.pred", "Network's prediction"), opt("s.correct", "Correct or not"),
      ...m.fields.filter(f => f.clock === "final").map(f => opt(f.id, "Loss")))];
    const dep = m.fields.filter(f => f.clock === "depth"), trn = m.fields.filter(f => f.clock === "train");
    if (dep.length) groups.push(h("optgroup", { label: "Depth step leaving this site" },
      dep.map(f => opt(f.id, f.label + (f.ckpts.includes(ck) ? "" : " (not at this checkpoint)")))));
    groups.push(h("optgroup", { label: "Training interval leaving this checkpoint" },
      trn.length ? trn.map(f => opt(f.id, f.label + (f.ckpts.includes(ck) ? "" : " (not from this checkpoint)")))
        : [opt("", "not measured yet", true)]));
    E.color.replaceChildren(...groups);
    E.color.value = st.color;
    E.arrows.value = st.arrows;
    E.dep.input.value = st.site; E.trn.input.value = st.ci;
    this.renderLegend();
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
  afterFilter() { this.styleChips(); this.readKey = null; this.paint(); }

  wire() {
    const E = this.el, st = this.st;
    E.color.addEventListener("change", () => { st.color = E.color.value; this.renderLegend(); this.chartKey = null; this.show(); this.writeHash(); });
    E.arrows.addEventListener("change", () => { st.arrows = E.arrows.value; this.show(); this.writeHash(); });
    E.helpBtn.addEventListener("click", () => this.toggleHelp());
    E.reset.addEventListener("click", () => this.resetView());
    E.dep.input.addEventListener("input", () => {
      this.stop(); const s0 = st.site; st.site = +E.dep.input.value;
      if (this.stageOf[s0] !== this.stageOf[st.site]) this.flagBreak(st.site);
      this.show(); this.writeHash();
    });
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
      this.select(this.view.pick(x, y, coarse.matches || e.pointerType === "touch" ? 18 : 9));
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

  // ------------------------------------------------------------------ fields
  // where a colour field's value for (checkpoint, site) lives: {f, ckpt (file), row} or {f, why}
  fieldAt(id, ck, site) {
    const f = this.fields.get(id), m = this.m;
    if (!f) return null;
    if (f.clock === "final") return { f, ckpt: f.ckpts[0], row: 0 };
    if (f.clock === "depth") {
      if (site >= this.S - 1) return { f, why: `no depth step leaves ${m.sites[site]}, the last site` };
      if (!f.ckpts.includes(ck)) return { f, why: `depth steps are not measured at checkpoint ${fmtInt(ck)}` };
      return { f, ckpt: ck, row: site };
    }
    if (!f.ckpts.includes(ck)) return { f, why: `no measured training interval starts at checkpoint ${fmtInt(ck)}` };
    return { f, ckpt: ck, row: site };
  }
  interval(ck) { const iv = (this.m.intervals || []).find(([a]) => a === ck); return iv || null; }
  // plain description of the step a field value belongs to
  stepText(id, ck, site) {
    const f = this.fields.get(id), m = this.m;
    if (!f) return "";
    if (f.clock === "final") return `(checkpoint ${fmtInt(m.final.ckpt)}); the same in every frame`;
    if (f.clock === "depth") {
      if (site >= this.S - 1) return `no depth step leaves ${m.sites[site]}, the last site, so every point is grey`;
      if (!f.ckpts.includes(ck)) return `depth steps are not measured at checkpoint ${fmtInt(ck)}, so every point is grey`;
      const chg = m.stage_change[site] ? ` (a stage change: the step starts from the shortcut's downsampled copy of ${m.sites[site]})` : "";
      return `for the depth step ${m.sites[site]} → ${m.sites[site + 1]} at checkpoint ${fmtInt(ck)}${chg}`;
    }
    const iv = this.interval(ck);
    return iv && f.ckpts.includes(ck) ? `for the training interval ${fmtInt(iv[0])} → ${fmtInt(iv[1])} at ${m.sites[site]} (${fmtInt(iv[1] - iv[0])} steps)`
      : `no measured training interval starts at checkpoint ${fmtInt(ck)}, so every point is grey`;
  }
  loadField(id, ck) {
    const loc = this.fieldAt(id, ck, 0);
    if (!loc || loc.ckpt == null) return Promise.resolve(null);
    return D.field(loc.f, loc.ckpt, this.N);
  }
  // the decoded value interval of one code, in data units: [centre, label]
  value(f, c) {
    const inv = f.transform === "log10" ? v => 10 ** v : v => v, w = (f.hi - f.lo) / 253;
    if (c === 255) return [null, "undefined"];
    if (c === 0) return [inv(f.lo), `below ${sig(inv(f.lo))}`];
    if (c === 254) return [inv(f.hi), `above ${sig(inv(f.hi))}`];
    return [inv(f.lo + (c - 0.5) * w), "≈ " + sig(inv(f.lo + (c - 0.5) * w))];
  }

  // ------------------------------------------------------------------ data
  needs() {
    const st = this.st, m = this.m, ck = m.ckpts[st.ci], out = [D.frame(m, ck, st.site)];
    if (st.arrows === "site" && st.site < this.S - 1 && this.stageOf[st.site + 1] === this.stageOf[st.site]) out.push(D.frame(m, ck, st.site + 1));
    if (st.arrows === "ckpt" && st.ci < m.ckpts.length - 1) out.push(D.frame(m, m.ckpts[st.ci + 1], st.site));
    if (this.fields.has(st.color)) out.push(this.loadField(st.color, ck));
    if (st.sel >= 0 && st.trail === "depth") for (const l of m.stages[this.stageOf[st.site]].sites) out.push(D.frame(m, ck, l));
    return out;
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
    this.paint();
    if (st.sel >= 0) this.loadSelFields();
  }
  prefetchDepth() { const ck = this.m.ckpts[this.st.ci]; for (let s = 0; s < this.S; s++) D.frame(this.m, ck, s).catch(() => {}); }

  // ------------------------------------------------------------------ playback
  play(kind) {
    if (this.anim && this.anim.kind === kind) { this.stop(); this.show(); return; }
    this.stop();
    const st = this.st, m = this.m;
    const end = kind === "depth" ? this.S - 1 : m.ckpts.length - 1;
    if (end <= 0) return;
    let u = kind === "depth" ? st.site : st.ci;
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
    this.anim = null; this.disp = null;
    for (const b of [this.el.dep.play, this.el.trn.play]) { b.textContent = "▶"; b.classList.remove("on"); }
    this.view.setFade(1);
    this.el.interp.hidden = true; this.stage.classList.remove("interp");
    this.writeHash();
  }
  // positions (and the colour field) for playback unit l are in memory; requests them if not
  unitReady(kind, l) {
    const st = this.st, m = this.m;
    const ck = kind === "depth" ? m.ckpts[st.ci] : m.ckpts[l], site = kind === "depth" ? l : st.site;
    let ok = !!D.frameNow(ck, site);
    if (!ok) D.frame(m, ck, site).catch(e => { console.error(e); this.failed = e; });
    const loc = this.fields.has(st.color) && this.fieldAt(st.color, ck, site);
    if (loc && loc.ckpt != null && !D.fieldNow(st.color, loc.ckpt)) { ok = false; D.field(loc.f, loc.ckpt, this.N).catch(e => { console.error(e); this.failed = e; }); }
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
    const st = this.st, m = this.m;
    if (a.kind === "depth") { if (st.site !== L) { st.site = L; this.el.dep.input.value = L; } }
    else if (st.ci !== L) { st.ci = L; this.el.trn.input.value = L; this.syncControls(); }
    const ck = m.ckpts[st.ci];
    let d = { ck, site: st.site, to: null, f: 0, fade: 1 };
    if (L < a.end && f > 0) {
      if (a.kind === "depth" && this.stageOf[L] !== this.stageOf[L + 1]) {
        // stage change: the axes are unrelated, so no motion is drawn; fade out, switch axes, fade in
        this.flagBreak(L + 1);
        d = f < 0.5 ? { ck, site: L, to: null, f: 0, fade: 1 - 2 * f } : { ck, site: L + 1, to: null, f: 0, fade: 2 * f - 1 };
      } else {
        const to = a.kind === "depth" ? D.frameNow(ck, L + 1) : D.frameNow(m.ckpts[L + 1], st.site);
        if (to) d = { ck, site: st.site, to, f, fade: 0.6, kind: a.kind, toLabel: a.kind === "depth" ? m.sites[L + 1] : `checkpoint ${fmtInt(m.ckpts[L + 1])}` };
      }
    }
    this.disp = d;
    this.paint();
    if (a.u >= a.end) { this.disp = null; this.stop(); this.show(); if (a.kind === "train") this.prefetchDepth(); return; }
    requestAnimationFrame(t => this.tick(t));
  }
  // the notice shown when the picture moves to another stage: new space, new axes, no motion drawn across
  flagBreak(site) {
    const m = this.m, sg = m.stages[this.stageOf[site]], E = this.el, names = sg.sites.map(l => m.sites[l]);
    E.brk.innerHTML = `<b>Axes change: now ${sg.name}.</b> ${names[0]}–${names[names.length - 1]} live in a different space
      (${fmtInt(sg.dim)} numbers per image) and are drawn on their own three axes, unrelated to those of the other stages.
      No motion is drawn across a stage change.`;
    E.brk.hidden = false;
    clearTimeout(this.brkT);
    this.brkT = setTimeout(() => { E.brk.hidden = true; }, BREAK_MS);
  }

  // ------------------------------------------------------------------ painting
  display() { return this.disp || { ck: this.m.ckpts[this.st.ci], site: this.st.site, to: null, f: 0, fade: 1 }; }
  resetView() {
    const sg = this.m.stages[this.stageOf[this.st.site]];
    this.shownStage = this.stageOf[this.st.site];
    this.view.fit(sg.camera.target, [0.42, 0.3, 1], [0, 1, 0], sg.camera.extent * (this.view.w < 520 ? 1.08 : 1.0));
  }
  paint() {
    if (!this.view) return;
    const d = this.display(), from = D.frameNow(d.ck, d.site);
    this.cur = d;
    if (!from) return;
    const s = this.stageOf[d.site];
    if (s !== this.shownStage) {
      this.shownStage = s;
      const cam = this.m.stages[s].camera;
      this.view.retarget(cam.target, cam.extent * (this.view.w < 520 ? 1.08 : 1.0));
    }
    const pos = this.view.pos, N = this.N;
    if (d.to) { const f = d.f, to = d.to; for (let k = 0; k < 3 * N; k++) pos[k] = from[k] + (to[k] - from[k]) * f; }
    else pos.set(from);
    this.style(d);
    this.view.setFade(d.fade);
    this.view.touch("pos"); this.view.touch("style");
    this.renderFrame(d);
    this.renderRead(d);
    const key = `${d.ck}_${d.site}_${this.st.color}`;
    if (key !== this.paintKey) { this.paintKey = key; this.renderColoring(d); this.renderCharts(d); if (!this.anim) this.renderSel(); }
  }

  colourFn(d) {
    const st = this.st, pal = this.pal, lab = this.lab, pred = this.pred;
    if (st.color === "s.pred") return i => [pal.cls[pred[i]], false];
    if (st.color === "s.correct") return i => [pred[i] === lab[i] ? pal.correct : pal.wrong, false];
    if (this.fields.has(st.color)) {
      const loc = this.fieldAt(st.color, d.ck, d.site), data = loc && loc.ckpt != null ? D.fieldNow(st.color, loc.ckpt) : null;
      if (!data) return () => [pal.undef, true];
      const fm = loc.f, ramp = fm.scale === "diverging" ? pal.div : pal.seq, o = loc.row * this.N,
        lut = Array.from({ length: 255 }, (_, k) => ramp(k === 0 ? 0 : k === 254 ? 1 : (k - 0.5) / 253));
      return i => { const c = data[o + i]; return c === 255 ? [pal.undef, true] : [lut[c], false]; };
    }
    return i => [pal.cls[lab[i]], false];
  }
  style(d) {
    const st = this.st, v = this.view, N = this.N, lab = this.lab, col = this.colourFn(d);
    const base = coarse.matches ? 3.0 : 3.4;
    for (let i = 0; i < N; i++) {
      const [c, hollow] = col(i);
      v.col[3 * i] = c[0] / 255; v.col[3 * i + 1] = c[1] / 255; v.col[3 * i + 2] = c[2] / 255;
      v.hollow[i] = hollow ? 1 : 0;
      if (!st.filter.has(lab[i])) { v.size[i] = 0; v.alpha[i] = 0; continue; }
      let a = hollow ? 0.5 : 0.82, sz = hollow ? base + 0.6 : base;
      if (i === st.sel) { sz = base + 4; a = 1; }
      v.size[i] = sz; v.alpha[i] = a;
    }
  }

  // view-fraction numbers for a frame: share of spread on the 3 axes, and of the displacement to the next site
  fractions(ck, site) {
    const T = this.T, vi = T.view.ckpts.indexOf(ck), di = T.depth.ckpts.indexOf(ck);
    const pos = vi >= 0 ? T.view.position_frac[vi][site] : null;
    const disp = di >= 0 && site < this.S - 1 ? T.depth.scalars.view3_displacement_frac[di][site] : null;
    return { pos, disp };
  }
  renderFrame(d) {
    const m = this.m, E = this.el, sg = m.stages[this.stageOf[d.site]];
    const { pos } = this.fractions(d.ck, d.site);
    const range = `${m.sites[sg.sites[0]]}–${m.sites[sg.sites[sg.sites.length - 1]]}`;
    const key = `${d.ck}_${d.site}`;
    if (key !== this.frameKey) {
      this.frameKey = key;
      E.frame.innerHTML = `<div><b>${m.sites[d.site]}</b> <span>checkpoint ${fmtInt(d.ck)}</span></div>
        <div class="ex-axes">axes of ${sg.name} (${range}, ${fmtInt(sg.dim)} numbers per image)</div>
        <div class="ex-vis">${pos == null ? "Share of the cloud on the 3 axes: not recorded." : this.view.flat
          ? `Axes 1–3 hold <b>${pct(pos)}</b> of this cloud's spread; this flat view draws only axes 1 and 2, so it shows less than that.`
          : `These 3 axes show <b>${pct(pos)}</b> of this cloud's spread; the other ${pct(1 - pos)} is in directions not drawn.`}</div>`;
    }
    if (d.to) {
      E.interp.textContent = `Interpolated, not data: ${Math.round(d.f * 100)}% of the straight line from ${m.sites[d.site]}${d.kind === "train" ? ` at checkpoint ${fmtInt(d.ck)}` : ""} to ${d.toLabel}. Colours are those of the frame being left.`;
      E.interp.hidden = false; this.stage.classList.add("interp");
    } else { E.interp.hidden = true; this.stage.classList.remove("interp"); }
  }
  renderRead(d) {
    const st = this.st, m = this.m, T = this.T;
    let shown = 0; for (const c of st.filter) shown += this.counts[c];
    const key = `${d.ck}_${d.site}_${shown}_${st.color}`;
    if (key === this.readKey) return;
    this.readKey = key;
    const sg = m.stages[this.stageOf[d.site]], { pos, disp } = this.fractions(d.ck, d.site);
    const di = T.depth.ckpts.indexOf(d.ck);
    const R = di >= 0 ? T.depth.R_site[di][d.site] : null;
    const rk = di >= 0 && d.site < this.S - 1 ? T.depth.scalars.r_k_rel[di][d.site] : null;
    const rows = [
      ["Frame", `${m.sites[d.site]}, checkpoint ${fmtInt(d.ck)}`],
      ["Spread on axes 1–3", pos == null ? "not recorded" : pct(pos)],
      [d.site < this.S - 1 ? `Displacement to ${m.sites[d.site + 1]} on axes 1–3` : "Displacement on axes 1–3",
        d.site >= this.S - 1 ? "no next site" : m.stage_change[d.site] ? `n/a: ${m.sites[d.site + 1]} is on other axes` : disp == null ? "not measured" : pct(disp)],
      ["RMS radius R of the cloud", R == null ? "not measured" : `${sig(R)} activation units`],
    ];
    if (rk != null) rows.push([`Distance to the 128th nearest image${m.stage_change[d.site] ? " (in the shortcut's copy, against its R)" : ""}`, `${sig(rk)} R on average`]);
    if (di >= 0 && T.depth.accuracy[di] != null) rows.push(["Test accuracy at this checkpoint", pct(T.depth.accuracy[di])]);
    rows.push(["Points shown", `${fmtInt(shown)} of ${fmtInt(this.N)}`]);
    const loc = this.fields.has(st.color) && this.fieldAt(st.color, d.ck, d.site);
    if (loc) {
      const data = loc.ckpt != null ? D.fieldNow(st.color, loc.ckpt) : null;
      if (data) {
        let n = 0, k = 0; const o = loc.row * this.N;
        for (let i = 0; i < this.N; i++) if (st.filter.has(this.lab[i])) { n++; if (data[o + i] !== 255) k++; }
        rows.push([`${loc.f.label}: defined on`, n ? pct(k / n) + " of shown points" : "n/a"]);
      }
    }
    this.el.read.replaceChildren(h("h3", { class: "card-title" }, "This frame"),
      h("dl", { class: "kv kv-wrap" }, rows.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])),
      h("p", { class: "card-note" }, `Spread: mean squared distance from ${sg.name}'s fixed centre; the share on axes 1–3 is what the 3D view shows${this.view.flat ? " (this flat view drops axis 3)" : ""}. Displacement: share of the mean squared displacement to the next site that lies along axes 1–3.`));
  }

  renderColoring(d) {
    const st = this.st, E = this.el;
    if (!this.fields.has(st.color)) {
      E.coloring.innerHTML = st.color === "s.class" ? "Colour: <b>true class</b> of each image."
        : st.color === "s.pred" ? `Colour: <b>the network's prediction</b> at the final checkpoint (${fmtInt(this.m.final.ckpt)}).`
          : `Colour: whether the network's prediction at the final checkpoint (${fmtInt(this.m.final.ckpt)}) is <b>correct</b>.`;
      return;
    }
    const f = this.fields.get(st.color);
    const t = this.stepText(st.color, d.ck, d.site);
    E.coloring.innerHTML = `Colour: <b>${f.label}</b>${/^(for|\()/.test(t) ? " " : "; "}${t}.`;
  }
  renderLegend() {
    const st = this.st, pal = this.pal, m = this.m, E = this.el;
    if (!E) return;
    const sw = (c, t, hollow) => h("span", { class: "kk" }, h("i", { class: hollow ? "hollow" : "", style: hollow ? { borderColor: css(c) } : { background: css(c) } }), t);
    let kids;
    if (st.color === "s.class" || st.color === "s.pred") kids = m.classes.map((c, i) => sw(pal.cls[i], c));
    else if (st.color === "s.correct") kids = [sw(pal.correct, `correct (${pct(m.final.accuracy)})`), sw(pal.wrong, "incorrect")];
    else {
      const fm = this.fields.get(st.color), ramp = fm.scale === "diverging" ? pal.div : pal.seq;
      const inv = fm.transform === "log10" ? v => 10 ** v : v => v;
      const clamp = fm.vmin < fm.lo || fm.vmax > fm.hi;
      kids = [h("span", { class: "kt" }, fm.units ? `${fm.label} (${fm.units}${fm.transform === "log10" ? ", log scale" : ""}):` : `${fm.label}:`),
        h("span", { class: "kbar-wrap" }, h("span", { class: "kt" }, sig(inv(fm.lo))), h("i", { class: "kbar", style: { background: gradient(ramp) } }), h("span", { class: "kt" }, sig(inv(fm.hi)))),
        sw(pal.undef, "undefined", true),
        h("span", { class: "kt kfull" }, fm.description),
        h("span", { class: "kt kfull" }, `Same colour range in every frame${fm.scale === "diverging" ? ", centred on zero" : ""}${clamp ? `; values beyond it (sampled data range ${sig(inv(fm.vmin))} to ${sig(inv(fm.vmax))}) take the end colours` : ""}.`)];
    }
    E.legend.replaceChildren(...kids);
  }

  // ------------------------------------------------------------------ side charts: step totals
  renderCharts(d) {
    const E = this.el, m = this.m, T = this.T, st = this.st;
    const f = this.fields.has(st.color) && this.fields.get(st.color).clock !== "final" ? this.fields.get(st.color) : this.fields.get("d.travel");
    if (!f) { E.charts.replaceChildren(h("h3", { class: "card-title" }, "Step totals"), h("p", { class: "card-note" }, "No depth fields measured yet.")); return; }
    const isD = f.clock === "depth", tot = isD ? T.depth.fields[f.id] : T.train.fields[f.id];
    const keys = isD ? T.depth.ckpts : T.train.intervals.map(iv => iv[0]);
    const ki = keys.indexOf(d.ck), stat = tot.stat === "median" ? "median" : "mean";
    const all = tot.v.flat().filter(Number.isFinite);
    const log = f.transform === "log10" && all.length && all.every(v => v > 0);
    let dom;
    if (!all.length) dom = [0, 1];
    else if (log) dom = [Math.min(...all) / 1.3, Math.max(...all) * 1.3];
    else { const lo = Math.min(0, ...all), hi = Math.max(0, ...all), pad = 0.06 * (hi - lo || 1); dom = [lo < 0 ? lo - pad : 0, hi + pad]; }
    const c1 = h("div", { class: "tot-chart" }), c2 = h("div", { class: "tot-chart" }), n1 = h("p", { class: "card-note" }), n2 = h("p", { class: "card-note" });
    const fallback = !this.fields.has(st.color) || this.fields.get(st.color).clock === "final";
    E.charts.replaceChildren(
      h("h3", { class: "card-title" }, `${f.label}, ${stat} over images`),
      ...(fallback ? [h("p", { class: "card-note" }, "Shown for travel because the colour is not a measured step field.")] : []),
      h("div", { class: "sub-h" }, isD ? `Over depth · checkpoint ${fmtInt(d.ck)}` : `Over sites · interval from checkpoint ${fmtInt(d.ck)}`), c1, n1,
      h("div", { class: "sub-h" }, `Over training · ${isD ? `step leaving ${m.sites[d.site]}` : m.sites[d.site]}`), c2, n2);
    const tf = v => sig(v), y = { domain: dom, log, fmt: v => sig(v) };
    const draw = () => {
      const col = getComputedStyle(document.documentElement).getPropertyValue("--c0").trim() || "#2a78d6";
      const col2 = getComputedStyle(document.documentElement).getPropertyValue("--c1").trim() || "#eb6834";
      if (ki < 0) n1.textContent = isD ? "Depth steps are not measured at this checkpoint." : "No measured training interval starts at this checkpoint.";
      else if (isD) {
        const v = tot.v[ki], chg = T.depth.stage_change;
        XC.chart(c1, { height: 150, x: { type: "band", labels: T.depth.steps.map(arrow) }, y, legend: true, marker: d.site < this.S - 1 ? d.site : null,
          series: [{ name: "within a stage", kind: "bar", color: col, values: v.map((x, i) => chg[i] ? null : x), tfmt: tf },
            { name: "stage change", kind: "dot", color: col2, values: v.map((x, i) => chg[i] ? x : null), tfmt: tf }] });
        const mass = Math.min(...tot.mass[ki]);
        n1.textContent = `Each mark is one depth step, named by its source and target site; the dashed line marks the step leaving the site shown. At a stage change the source is the shortcut's downsampled copy.${mass < 0.9995 ? ` Defined on at least ${pct(mass)} of images at every step.` : ""}`;
      } else {
        XC.chart(c1, { height: 150, x: { type: "band", labels: m.sites }, y, legend: false, marker: d.site,
          series: [{ name: f.label, kind: "bar", color: col, values: tot.v[ki], tfmt: tf }] });
        const iv = T.train.intervals[ki];
        n1.textContent = `One bar per site for the interval ${fmtInt(iv[0])} → ${fmtInt(iv[1])}; the dashed line marks the site shown.`
          + (tot.v[ki].every(x => x == null) ? " Not defined for this interval." : "");
      }
      if (isD && d.site >= this.S - 1) { c2.replaceChildren(); n2.textContent = `No depth step leaves ${m.sites[d.site]}.`; return; }
      const pts = keys.map((k, j) => [k, tot.v[j][d.site]]).filter(p => Number.isFinite(p[1]));
      if (!pts.length) { c2.replaceChildren(); n2.textContent = "No values at this site."; return; }
      const last = Math.max(...m.ckpts, ...keys);
      XC.chart(c2, { height: 150, x: { type: "logstep", domain: [0, last], tipLabel: v => (isD ? "checkpoint " : "interval from ") + fmtInt(v) }, y, legend: false,
        marker: d.ck, series: [{ name: f.label, kind: pts.length > 1 ? "line" : "dot", markers: true, color: col, points: pts, tfmt: tf }] });
      n2.textContent = isD ? `One point per measured checkpoint (${pts.length} so far), on a log axis of training step; the dashed line marks the checkpoint shown.`
        : `One point per measured interval, placed at its first checkpoint, on a log axis of training step${f.id === "t.travel_rate" ? "; a rate, so intervals of different length compare" : "; values are over the whole interval, and intervals differ in length"}.`;
    };
    if (XC.chart) draw(); else XC.ready.then(draw);
  }

  // ------------------------------------------------------------------ selection
  select(i) {
    const st = this.st;
    if (i === st.sel && i >= 0) return;
    st.sel = i;
    this.show(); this.writeHash();
    if (i >= 0) D.sprite(this.m).catch(() => {});
    this.renderSel();
  }
  setHover(i, x, y) {
    const tip = document.getElementById("tooltip");
    if (i !== this.hover) { this.hover = i; this.view.render(); }
    if (i < 0) { tip.hidden = true; return; }
    const d = this.cur; if (!d) return;
    const m = this.m, st = this.st;
    let line = null;
    if (this.fields.has(st.color)) {
      const loc = this.fieldAt(st.color, d.ck, d.site), data = loc && loc.ckpt != null ? D.fieldNow(st.color, loc.ckpt) : null;
      if (data) line = h("div", {}, `${loc.f.label}: ${this.value(loc.f, data[loc.row * this.N + i])[1]}`);
    }
    tip.replaceChildren(h("div", { class: "tt-img" }, this.thumb(i),
      h("div", {}, h("div", { class: "tt-head" }, `#${i} · ${m.classes[this.lab[i]]}`),
        h("div", {}, `network: ${m.classes[this.pred[i]]}${this.pred[i] === this.lab[i] ? "" : " (wrong)"}`), line,
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
  // every step field of the frame on screen, for the selected image's value table
  loadSelFields() {
    const d = this.display(), ids = this.m.fields.filter(f => f.clock !== "final").map(f => f.id);
    const ps = ids.map(id => { const loc = this.fieldAt(id, d.ck, d.site); return loc && loc.ckpt != null ? D.field(loc.f, loc.ckpt, this.N) : null; }).filter(Boolean);
    const loss = this.fields.get("s.loss");
    if (loss) ps.push(D.field(loss, loss.ckpts[0], this.N));
    Promise.all(ps).then(() => { if (!this.anim) this.renderSel(); }).catch(e => console.error(e));
  }
  renderSel() {
    const E = this.el, st = this.st, m = this.m, i = st.sel;
    if (!E) return;
    if (i < 0) {
      E.sel.replaceChildren(h("h3", { class: "card-title" }, "Selected image"),
        h("p", { class: "card-note" }, "Click a point to see its image, the network's prediction, its values for the step leaving this frame, and its trail through depth or training."));
      return;
    }
    const d = this.cur || this.display(), y = this.lab[i], p = this.pred[i];
    const loss = this.fields.get("s.loss"), lossData = loss && D.fieldNow("s.loss", loss.ckpts[0]);
    const table = (clock, head) => {
      const fs = m.fields.filter(f => f.clock === clock);
      if (!fs.length) return [];
      const loc0 = this.fieldAt(fs[0].id, d.ck, d.site);
      if (loc0.ckpt == null) return [h("div", { class: "sub-h" }, head), h("p", { class: "card-note" }, loc0.why[0].toUpperCase() + loc0.why.slice(1) + ".")];
      const rows = fs.flatMap(f => {
        const data = D.fieldNow(f.id, d.ck);
        const txt = data ? this.value(f, data[d.site * this.N + i])[1] : "…";
        const unit = ["R_X", "R_X per step", "log ratio"].includes(f.units) && data && data[d.site * this.N + i] !== 255 ? ` ${f.units}` : "";
        return [h("dt", { title: f.description }, f.label), h("dd", {}, txt + unit)];
      });
      return [h("div", { class: "sub-h" }, head), h("dl", { class: "kv kv-tight" }, rows)];
    };
    const iv = this.interval(d.ck);
    const depthHead = d.site < this.S - 1 ? `Depth step ${m.sites[d.site]} → ${m.sites[d.site + 1]}, checkpoint ${fmtInt(d.ck)}` : `Depth step leaving ${m.sites[d.site]}`;
    const trainHead = iv ? `Training interval ${fmtInt(iv[0])} → ${fmtInt(iv[1])}, ${m.sites[d.site]}` : `Training interval from checkpoint ${fmtInt(d.ck)}`;
    const trail = h("div", { class: "seg seg-sm", role: "radiogroup", "aria-label": "Trail" },
      [["depth", "Through depth"], ["train", "Through training"], ["off", "Off"]].map(([v, t]) =>
        h("button", { type: "button", role: "radio", "aria-checked": String(st.trail === v), onclick: () => { st.trail = v; this.show(); this.writeHash(); this.renderSel(); } }, t)));
    const sg = m.stages[this.stageOf[d.site]];
    this.trailNote = h("div", { class: "card-note" }, st.trail === "depth" ? `Its positions at the sites of ${sg.name} (${sg.sites.map(l => m.sites[l]).join(", ")}) at checkpoint ${fmtInt(d.ck)}; other stages have other axes.` :
      st.trail === "train" ? `Its positions at ${m.sites[d.site]} over the ${m.ckpts.length} checkpoint${m.ckpts.length > 1 ? "s" : ""} with data.` : "");
    E.sel.replaceChildren(
      h("div", { class: "sel-head" }, this.thumb(i, "big"),
        h("div", {}, h("div", { class: "sel-id" }, `Image #${i}`, h("button", { type: "button", class: "x-btn", "aria-label": "Clear selection", onclick: () => this.select(-1) }, "×")),
          h("div", {}, "True class: ", h("b", {}, m.classes[y])),
          h("div", {}, `Network at checkpoint ${fmtInt(m.final.ckpt)}: `, h("b", {}, m.classes[p]), p === y ? " (correct)" : " (wrong)"),
          lossData ? h("div", {}, `Loss there: ${this.value(loss, lossData[i])[1]} nats`) : null)),
      ...table("depth", depthHead), ...table("train", trainHead),
      h("p", { class: "card-note" }, "Values are read from the colour files, so each is the centre of a bin 1/253 of the colour range wide (on a log scale where the legend says so). Hover a name for its definition."),
      h("div", { class: "sub-h" }, "Trail"), trail, this.trailNote);
  }
  loadTrainTrail() {
    const st = this.st, m = this.m, site = st.site;
    Promise.all(m.ckpts.map(ck => D.frame(m, ck, site))).then(() => {
      if (st.trail === "train" && st.site === site) this.view.render();
    }).catch(e => console.error(e));
  }

  // ------------------------------------------------------------------ overlay: axes, scale bar, arrows, trail, rings
  overlay(ctx) {
    const st = this.st, pal = this.pal, v = this.view, m = this.m, d = this.cur;
    if (!d) return;
    const font = (w, s) => `${w} ${s}px Inter, system-ui, sans-serif`;
    const label = (t, x, y, color, { size = 11, weight = 600, align = "left" } = {}) => {
      ctx.font = font(weight, size); ctx.textAlign = align; ctx.textBaseline = "middle";
      ctx.lineJoin = "round"; ctx.lineWidth = 3.5; ctx.strokeStyle = css(pal.bg, 0.9); ctx.strokeText(t, x, y);
      ctx.fillStyle = color; ctx.fillText(t, x, y);
    };
    const S = q => v.toScreen(q[0], q[1], q[2]);
    const sg = m.stages[this.stageOf[d.site]], tgt = sg.camera.target, L = sg.camera.extent * 0.22;
    // axis directions through the stage's centre
    const o = S(tgt);
    ctx.lineWidth = 1.2;
    for (let a = 0; a < 3; a++) {
      const q = tgt.slice(); q[a] += L;
      const p = S(q);
      if (Math.hypot(p[0] - o[0], p[1] - o[1]) < 4) continue;
      ctx.strokeStyle = css(pal.guide, 0.8); ctx.beginPath(); ctx.moveTo(...o); ctx.lineTo(...p); ctx.stroke();
      label(`axis ${a + 1}`, p[0] + 4, p[1], css(pal.guide), { size: 10.5, weight: 600 });
    }
    // scale bar, in activation units
    const ppu = v.pxPerUnit(), want = 90 / ppu, mag = 10 ** Math.floor(Math.log10(want));
    const len = [1, 2, 5, 10].map(k => k * mag).find(k => k >= want * 0.6) || mag * 10, px = len * ppu;
    const bx = v.w - 16 - px, by = v.h - 16;
    ctx.strokeStyle = css(pal.fg, 0.75); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(bx, by - 4); ctx.lineTo(bx, by); ctx.lineTo(bx + px, by); ctx.lineTo(bx + px, by - 4); ctx.stroke();
    label(`${len >= 1 ? +len.toPrecision(3) : sig(len)} activation units`, bx + px, by - 11, css(pal.fg, 0.8), { size: 10.5, weight: 500, align: "right" });
    const from = D.frameNow(d.ck, d.site);
    if (!from || d.fade < 1 && !d.to) return;
    const poly = pts => { ctx.beginPath(); ctx.moveTo(...pts[0]); for (const q of pts.slice(1)) ctx.lineTo(...q); ctx.stroke(); };
    const head = (p0, p1, color, sz) => {
      const a = Math.atan2(p1[1] - p0[1], p1[0] - p0[0]);
      if (Math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 1.5) return;
      ctx.fillStyle = color; ctx.beginPath(); ctx.moveTo(...p1);
      ctx.lineTo(p1[0] - sz * Math.cos(a - 0.45), p1[1] - sz * Math.sin(a - 0.45)); ctx.lineTo(p1[0] - sz * Math.cos(a + 0.45), p1[1] - sz * Math.sin(a + 0.45)); ctx.fill();
    };
    const P = (fr, i) => [fr[3 * i], fr[3 * i + 1], fr[3 * i + 2]];
    // arrows: straight displacement to the next site (same stage) or the next checkpoint (same site), projected
    if (!d.to && st.arrows !== "off") {
      let to = null;
      if (st.arrows === "site" && d.site < this.S - 1 && this.stageOf[d.site + 1] === this.stageOf[d.site]) to = D.frameNow(d.ck, d.site + 1);
      if (st.arrows === "ckpt") { const k = m.ckpts.indexOf(d.ck); if (k < m.ckpts.length - 1) to = D.frameNow(m.ckpts[k + 1], d.site); }
      if (to) {
        let ids;
        if (st.sel >= 0) ids = [st.sel];
        else { ids = []; for (let i = 0; i < this.N; i += ARROW_EVERY) if (st.filter.has(this.lab[i])) ids.push(i); }
        ctx.lineWidth = ids.length > 50 ? 1 : 1.8;
        for (const i of ids) {
          const c = css([v.col[3 * i] * 255, v.col[3 * i + 1] * 255, v.col[3 * i + 2] * 255], ids.length > 50 ? 0.7 : 0.95);
          const p0 = S(P(from, i)), p1 = S(P(to, i));
          ctx.strokeStyle = c; poly([p0, p1]); head(p0, p1, c, ids.length > 50 ? 5 : 7);
        }
      }
    }
    // trail of the selected image
    const i = st.sel, pts = [], marks = [];
    if (i >= 0 && st.trail !== "off" && !d.to) {
      if (st.trail === "depth") {
        const fs = sg.sites.map(l => D.frameNow(d.ck, l));
        if (fs.every(Boolean)) sg.sites.forEach((l, k) => { const p = S(P(fs[k], i)); pts.push(p); marks.push([p, m.sites[l], l === d.site]); });
      } else {
        const fs = m.ckpts.map(ck => D.frameNow(ck, d.site));
        if (fs.every(Boolean)) fs.forEach((fr, k) => { const p = S(P(fr, i)); pts.push(p); marks.push([p, k === 0 || k === fs.length - 1 ? `ckpt ${short(m.ckpts[k])}` : "", m.ckpts[k] === d.ck]); });
      }
    }
    if (pts.length > 1) {
      ctx.strokeStyle = css(pal.bg, 0.85); ctx.lineWidth = 4.5; poly(pts);
      ctx.strokeStyle = css(pal.fg, 0.9); ctx.lineWidth = 1.8; poly(pts);
    }
    const ring = (j, r, color, w) => {
      const q = S([v.pos[3 * j], v.pos[3 * j + 1], v.pos[3 * j + 2]]);
      ctx.strokeStyle = css(pal.bg, 0.9); ctx.lineWidth = w + 2.5; ctx.beginPath(); ctx.arc(q[0], q[1], r, 0, 7); ctx.stroke();
      ctx.strokeStyle = color; ctx.lineWidth = w; ctx.beginPath(); ctx.arc(q[0], q[1], r, 0, 7); ctx.stroke();
    };
    if (i >= 0 && st.filter.has(this.lab[i])) ring(i, 8, css(pal.accent), 2.2);
    if (this.hover >= 0 && this.hover !== i) ring(this.hover, 6, css(pal.fg), 1.5);
    for (const [p, t, on] of marks) {
      ctx.fillStyle = css(on ? pal.accent : pal.fg); ctx.beginPath(); ctx.arc(p[0], p[1], on ? 4 : 2.6, 0, 7); ctx.fill();
      if (t) label(t, p[0] + 9, p[1] - 9, css(pal.fg));
    }
  }
}

const HELP = `
<p><b>What a point is.</b> Each point is one of the 10,000 CIFAR-10 test images. Its position is the network's
activation for that image at the chosen site, the stem output or the output of block b1 to b9 (after the ReLU, all
channels and pixel positions as one long vector: 16,384 numbers per image up to b3, 8,192 for b4 to b6, 4,096 for b7
to b9). Distances are ordinary Euclidean distances in that space.</p>
<p><b>The axes.</b> The network has three stages, and each has its own fixed axes: the leading principal directions
of that stage's sites pooled at the final checkpoint. The picture uses the first three. Within a stage every site and
every checkpoint is drawn on the same axes, at the same scale and from the same camera, so motion you see there is
motion along those axes. At b3 → b4 and b6 → b7 the network halves the image resolution and doubles the channels, so
the space changes: the axes change, the picture jumps, and nothing is interpolated across that step.</p>
<p><b>How much is shown.</b> Three directions out of thousands. The share of the cloud's spread (mean squared distance
from the stage's centre) that the three axes show is printed in the view and in the side card, with the share of each
step's displacement that they show. Points that look close here can be far apart in the full space.</p>
<p><b>Moving.</b> <b>Depth</b> moves through the sites at one checkpoint; <b>Training</b> moves through the measured
checkpoints at one site. ▶ plays. Between two frames of the same stage the points slide along straight lines; such
in-between positions are not data and are drawn faded inside a dashed frame. Arrows show each point's straight-line
displacement to the next site or checkpoint, on the same three axes. Drag to rotate, scroll or pinch to zoom; click a
point for its image, its values and its trail.</p>
<p><b>Colouring by a measured field.</b> A depth field belongs to the step leaving the site shown (b9 has none); a
training field to the interval from the checkpoint shown to the next measured one. The line under the picture says
which step is coloured. Neighbourhood fields use each image's 128 nearest images in the full activation space. At
10,000 images the 128th nearest is about as far away as a typical image is from the cloud's centre (see "Distance to
the 128th nearest image" in the side card), so these are graph quantities at the scale of the whole cloud, not
properties of a point. No reproducible local plane was found at any site at these scales, so nothing here splits
motion into parts along and across the cloud.</p>
<p><b>Colours</b> keep one range across every frame; values beyond it take the end colours, and grey rings are images
where the field is undefined. The side charts show the step totals of the field over depth at this checkpoint and
over training at this site, with the y range fixed across frames.</p>`;
