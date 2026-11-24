// Loading and decoding of the binary assets written by build_site.py (formats described there and in meta.json).
const ROOT = "data/";            // one run per deploy (build_site.py); meta.json names it
const cache = new Map();          // url -> Promise, for the small files
const pending = new Map();        // "step_site" -> Promise<Float32Array>
const ready = new Map();          // "step_site" -> Float32Array [N * 10] of sqrt(p), unit norm per row (LRU order)
const MAX_FRAMES = 160;           // 160 x 400 kB of decoded floats

async function get(url, kind) {
  if (!cache.has(url)) {
    const p = fetch(url).then(r => { if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return kind === "json" ? r.json() : r.arrayBuffer(); });
    p.catch(() => cache.delete(url));
    cache.set(url, p);
  }
  return cache.get(url);
}

export const meta = () => get(ROOT + "meta.json", "json");
export const labels = async () => new Uint8Array(await get(ROOT + "labels.bin"));
export const manifest = () => get("experiments/manifest.json", "json");
export const totals = m => m.totals ? get(ROOT + m.totals, "json") : Promise.resolve(null);

// The lens's top class holds 65535; the others sqrt(p_c) = exp(-q L / 131068). The top coordinate is then
// sqrt(1 - sum of the others' squares), so |s| = 1 exactly. s.top[i] is the lens prediction (exact).
function decode(buf, L, N) {
  const q = new Uint16Array(buf), s = new Float32Array(N * 10), top = new Uint8Array(N), k = L / 131068;
  for (let i = 0; i < N; i++) {
    const o = i * 10;
    let t = 0, ss = 0;
    for (let c = 0; c < 10; c++) {
      if (q[o + c] === 65535) { t = c; continue; }
      const v = Math.exp(-q[o + c] * k);
      s[o + c] = v; ss += v * v;
    }
    s[o + t] = Math.sqrt(Math.max(0, 1 - ss));
    top[i] = t;
  }
  s.top = top;
  return s;
}

// the decoded frame if it is in memory, else null
export function frameNow(step, site) {
  const key = `${step}_${site}`, v = ready.get(key);
  if (v) { ready.delete(key); ready.set(key, v); }
  return v || null;
}
export function frame(m, step, site) {
  if (!Number.isInteger(site) || site < 0 || site >= m.sites.length || !m.steps.includes(step))
    return Promise.reject(new Error(`no frame for step ${step}, site ${site}`));
  const key = `${step}_${site}`, v = frameNow(step, site);
  if (v) return Promise.resolve(v);
  if (!pending.has(key)) {
    const url = `${ROOT}x/${key}.bin`;
    const p = fetch(url).then(r => { if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return r.arrayBuffer(); })
      .then(buf => {
        const s = decode(buf, m.L[m.steps.indexOf(step)][site], m.N);
        ready.set(key, s);
        while (ready.size > MAX_FRAMES) ready.delete(ready.keys().next().value);
        return s;
      })
      .finally(() => pending.delete(key));
    pending.set(key, p);
  }
  return pending.get(key);
}

// uint8 [9, N]: 0..254 = lo + code / 254 * (hi - lo), 255 = undefined
export async function field(name, step) {
  return new Uint8Array(await get(`${ROOT}f/${name}/${step}.bin`));
}

let spriteP = null;
export function sprite(m) {
  if (!spriteP) {
    spriteP = new Promise((res, rej) => {
      const im = new Image();
      im.onload = () => res(im);
      im.onerror = () => { spriteP = null; rej(new Error("image atlas failed to load")); };
      im.src = ROOT + m.sprite.file;
    });
  }
  return spriteP;
}
