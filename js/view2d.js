// Flat fallback for browsers without WebGL 2: the same interface as view3d.js, drawn on a 2D canvas. It shows the
// first two axes only (axis 1 to the right, axis 2 up); there is no rotation. Scroll or pinch to zoom, drag to pan.
export class View2D {
  constructor(stage, N, onOverlay) {
    this.stage = stage; this.N = N; this.onOverlay = onOverlay; this.flat = true;
    // a fresh canvas: one on which a WebGL context was attempted cannot always give a 2D context
    const old = stage.querySelector("canvas.gl"), cv = old.cloneNode(false);
    old.replaceWith(cv);
    this.gl = cv;
    this.ctx = cv.getContext("2d");
    this.ov = stage.querySelector("canvas.ov");
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.pos = new Float32Array(N * 3); this.col = new Float32Array(N * 3);
    this.size = new Float32Array(N); this.alpha = new Float32Array(N); this.hollow = new Float32Array(N);
    this.fade = 1; this.extent = 1; this.cx = 0; this.cy = 0; this.zoom = 1; this.panX = 0; this.panY = 0;
    this.screen = new Float32Array(N * 2); this.screenOk = false; this.queued = false;
    this.wire();
    new ResizeObserver(() => this.resize()).observe(stage);
    this.resize();
  }
  wire() {
    const cv = this.gl;
    cv.addEventListener("wheel", e => {
      e.preventDefault();
      this.zoom = Math.max(0.3, Math.min(30, this.zoom * Math.exp(-e.deltaY * 0.0015)));
      this.render();
    }, { passive: false });
    let drag = null;
    const pts = new Map();
    cv.addEventListener("pointerdown", e => { pts.set(e.pointerId, [e.clientX, e.clientY]); drag = [e.clientX, e.clientY, this.panX, this.panY, null]; cv.setPointerCapture?.(e.pointerId); });
    cv.addEventListener("pointermove", e => {
      if (!pts.has(e.pointerId) || !drag) return;
      pts.set(e.pointerId, [e.clientX, e.clientY]);
      if (pts.size === 2) {                                  // pinch
        const [a, b] = [...pts.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]);
        if (drag[4]) this.zoom = Math.max(0.3, Math.min(30, this.zoom * d / drag[4]));
        drag[4] = d; this.render(); return;
      }
      this.panX = drag[2] + (e.clientX - drag[0]); this.panY = drag[3] + (e.clientY - drag[1]);
      this.render();
    });
    const up = e => { pts.delete(e.pointerId); if (!pts.size) drag = null; };
    cv.addEventListener("pointerup", up); cv.addEventListener("pointercancel", up);
  }
  resize() {
    const w = Math.max(1, this.stage.clientWidth), h = Math.max(1, this.stage.clientHeight);
    this.w = w; this.h = h;
    for (const c of [this.gl, this.ov]) { c.width = Math.round(w * this.dpr); c.height = Math.round(h * this.dpr); }
    this.render();
  }
  pxPerUnit() { return Math.min(this.w, this.h) / (2 * this.extent) * this.zoom; }
  fit(target, _dir, _up, extent) { this.cx = target[0]; this.cy = target[1]; this.extent = extent; this.zoom = 1; this.panX = this.panY = 0; this.render(); }
  retarget(target, extent) { this.cx = target[0]; this.cy = target[1]; this.extent = extent; this.render(); }
  setFade(f) { this.fade = f; }
  setGuides() {}
  touch() { this.screenOk = false; this.render(); }
  toScreen(x, y) {
    const s = this.pxPerUnit();
    return [this.w / 2 + this.panX + (x - this.cx) * s, this.h / 2 + this.panY - (y - this.cy) * s];
  }
  screenAll() {
    if (this.screenOk) return this.screen;
    const p = this.pos, o = this.screen, s = this.pxPerUnit(), x0 = this.w / 2 + this.panX, y0 = this.h / 2 + this.panY;
    for (let i = 0; i < this.N; i++) { o[2 * i] = x0 + (p[3 * i] - this.cx) * s; o[2 * i + 1] = y0 - (p[3 * i + 1] - this.cy) * s; }
    this.screenOk = true;
    return o;
  }
  render() {
    if (this.queued) return;
    this.queued = true;
    requestAnimationFrame(() => { this.queued = false; this.draw(); });
  }
  draw() {
    this.screenOk = false;
    const ctx = this.ctx, s = this.screenAll(), c = this.col;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    for (let i = 0; i < this.N; i++) {
      const a = this.alpha[i] * this.fade, r = this.size[i] / 2;
      if (a <= 0 || r <= 0) continue;
      const style = `rgba(${c[3 * i] * 255 | 0},${c[3 * i + 1] * 255 | 0},${c[3 * i + 2] * 255 | 0},${a.toFixed(3)})`;
      ctx.beginPath(); ctx.arc(s[2 * i], s[2 * i + 1], r, 0, 6.2832);
      if (this.hollow[i] > 0.5) { ctx.strokeStyle = style; ctx.lineWidth = Math.max(1, r * 0.45); ctx.stroke(); }
      else { ctx.fillStyle = style; ctx.fill(); }
    }
    const o = this.ov.getContext("2d");
    o.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    o.clearRect(0, 0, this.w, this.h);
    if (this.onOverlay) this.onOverlay(o, this);
  }
  pick(px, py, r) {
    const s = this.screenAll(), a = this.alpha, sz = this.size;
    let best = -1, bd = r * r;
    for (let i = 0; i < this.N; i++) {
      if (a[i] <= 0.05 || sz[i] <= 0) continue;
      const dx = s[2 * i] - px, dy = s[2 * i + 1] - py, d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }
}
