// The 3D view: a three.js point cloud (one round sprite per image), guide geometry, an orbiting orthographic camera,
// and a 2D overlay canvas for labels, arrows, trails and rings. Picking is done on the CPU from projected positions.
import * as THREE from "three";
import { OrbitControls } from "../vendor/three/OrbitControls.js";

const VS = `
attribute vec3 aColor; attribute float aSize; attribute float aAlpha; attribute float aHollow;
uniform float uPx; uniform float uFade;
varying vec3 vColor; varying float vAlpha; varying float vHollow;
void main() {
  vColor = aColor; vAlpha = aAlpha * uFade; vHollow = aHollow;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = aSize * uPx;
}`;
const FS = `
varying vec3 vColor; varying float vAlpha; varying float vHollow;
void main() {
  float r = length(gl_PointCoord * 2.0 - 1.0);
  if (r > 1.0 || vAlpha <= 0.0) discard;
  float a = 1.0 - smoothstep(0.72, 1.0, r);
  if (vHollow > 0.5) a *= smoothstep(0.38, 0.62, r);
  gl_FragColor = vec4(vColor, vAlpha * a);
}`;

export class View {
  constructor(stage, N, onOverlay) {
    this.stage = stage; this.N = N; this.onOverlay = onOverlay;
    this.gl = stage.querySelector("canvas.gl");
    this.ov = stage.querySelector("canvas.ov");
    this.renderer = new THREE.WebGLRenderer({ canvas: this.gl, antialias: true, alpha: true });
    this.renderer.setClearColor(0x000000, 0);
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.renderer.setPixelRatio(this.dpr);
    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -100, 100);
    this.extent = 2;
    this.controls = new OrbitControls(this.camera, this.gl);
    this.controls.enableDamping = false;
    this.controls.zoomSpeed = 1.2;
    this.controls.addEventListener("change", () => this.render());
    this.guides = new THREE.Group();
    this.scene.add(this.guides);

    const g = new THREE.BufferGeometry();
    this.pos = new Float32Array(N * 3); this.col = new Float32Array(N * 3);
    this.size = new Float32Array(N); this.alpha = new Float32Array(N); this.hollow = new Float32Array(N);
    g.setAttribute("position", new THREE.BufferAttribute(this.pos, 3));
    g.setAttribute("aColor", new THREE.BufferAttribute(this.col, 3));
    g.setAttribute("aSize", new THREE.BufferAttribute(this.size, 1));
    g.setAttribute("aAlpha", new THREE.BufferAttribute(this.alpha, 1));
    g.setAttribute("aHollow", new THREE.BufferAttribute(this.hollow, 1));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 10);
    this.geom = g;
    this.mat = new THREE.ShaderMaterial({
      vertexShader: VS, fragmentShader: FS, transparent: true, depthWrite: false, depthTest: false,
      uniforms: { uPx: { value: this.dpr }, uFade: { value: 1 } },
    });
    this.points = new THREE.Points(g, this.mat);
    this.points.frustumCulled = false;
    this.points.renderOrder = 2;
    this.scene.add(this.points);

    this.mvp = new THREE.Matrix4();
    this.screen = new Float32Array(N * 2); this.screenOk = false;
    this.queued = false;
    new ResizeObserver(() => this.resize()).observe(stage);
    this.resize();
  }

  resize() {
    const w = Math.max(1, this.stage.clientWidth), h = Math.max(1, this.stage.clientHeight);
    this.w = w; this.h = h;
    this.renderer.setSize(w, h, false);
    this.ov.width = Math.round(w * this.dpr); this.ov.height = Math.round(h * this.dpr);
    this.frustum();
    this.render();
  }
  frustum() {
    const a = this.w / this.h, e = this.extent, c = this.camera;
    const hh = a >= 1 ? e : e / a;
    c.left = -hh * a; c.right = hh * a; c.top = hh; c.bottom = -hh;
    c.updateProjectionMatrix();
  }
  // look at `target` from direction `dir`, with `up` up, showing a disc of radius `extent`
  fit(target, dir, up, extent) {
    this.extent = extent;
    const t = new THREE.Vector3(...target), d = new THREE.Vector3(...dir).normalize();
    this.camera.up.set(...up);
    this.camera.position.copy(t).addScaledVector(d, 20);
    this.camera.zoom = 1;
    this.controls.target.copy(t);
    this.frustum();
    this.controls.update();
    this.render();
  }

  touch(what) {
    const a = this.geom.attributes;
    if (what === "pos") a.position.needsUpdate = true;
    else { a.aColor.needsUpdate = true; a.aSize.needsUpdate = true; a.aAlpha.needsUpdate = true; a.aHollow.needsUpdate = true; }
    this.render();
  }
  setFade(f) { this.mat.uniforms.uFade.value = f; }

  // guides: list of {kind: "lines", segs: Float32Array of xyz pairs, color: [r, g, b] 0-255, opacity}
  //         or {kind: "mesh", pos: Float32Array, index: Uint16Array|Uint32Array, color, opacity}
  setGuides(list) {
    for (const o of [...this.guides.children]) { o.geometry.dispose(); o.material.dispose(); this.guides.remove(o); }
    for (const gd of list) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(gd.pos || gd.segs, 3));
      const color = new THREE.Color(gd.color[0] / 255, gd.color[1] / 255, gd.color[2] / 255);
      let obj;
      if (gd.kind === "mesh") {
        geo.setIndex(new THREE.BufferAttribute(gd.index, 1));
        obj = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: gd.opacity, side: THREE.DoubleSide, depthWrite: false }));
      } else {
        obj = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: gd.opacity, depthWrite: false }));
      }
      obj.renderOrder = gd.kind === "mesh" ? 0 : 1;
      this.guides.add(obj);
    }
    this.render();
  }

  render() {
    if (this.queued) return;
    this.queued = true;
    requestAnimationFrame(() => { this.queued = false; this.draw(); });
  }
  draw() {
    this.camera.updateMatrixWorld();
    this.renderer.render(this.scene, this.camera);
    this.mvp.multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse);
    this.screenOk = false;
    const ctx = this.ov.getContext("2d");
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    if (this.onOverlay) this.onOverlay(ctx, this);
  }

  // world -> CSS pixels in the stage (orthographic camera, so w = 1)
  toScreen(x, y, z) {
    const e = this.mvp.elements;
    const nx = e[0] * x + e[4] * y + e[8] * z + e[12], ny = e[1] * x + e[5] * y + e[9] * z + e[13];
    return [(nx + 1) / 2 * this.w, (1 - ny) / 2 * this.h];
  }
  screenAll() {
    if (this.screenOk) return this.screen;
    const e = this.mvp.elements, p = this.pos, s = this.screen, w = this.w, h = this.h;
    for (let i = 0; i < this.N; i++) {
      const x = p[3 * i], y = p[3 * i + 1], z = p[3 * i + 2];
      s[2 * i] = ((e[0] * x + e[4] * y + e[8] * z + e[12]) + 1) / 2 * w;
      s[2 * i + 1] = (1 - (e[1] * x + e[5] * y + e[9] * z + e[13])) / 2 * h;
    }
    this.screenOk = true;
    return s;
  }
  // nearest drawn point within r CSS pixels of (px, py), or -1
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
