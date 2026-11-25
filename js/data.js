// Loading and decoding of the binary assets written by build_site.py (formats described there and in meta.json).
const ROOT = "data/";             // one run per deploy (build_site.py); meta.json names it
const cache = new Map();          // url -> Promise, for the small files
const pending = new Map();        // key -> Promise
const frames = new Map();         // "ckpt_site" -> Float32Array [N * 3] of positions on the stage's first 3 axes (LRU)
const fieldBufs = new Map();      // "id_ckpt" -> Uint8Array [rows * N] (LRU)
const MAX_FRAMES = 420;           // every frame of 40 checkpoints x 10 sites, 120 kB each
const MAX_FIELDS = 240;           // up to 100 kB each

async function get(url, kind) {
  if (!cache.has(url)) {
    const p = fetch(url).then(r => { if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return kind === "json" ? r.json() : r.arrayBuffer(); });
    p.catch(() => cache.delete(url));
    cache.set(url, p);
  }
  return cache.get(url);
}

export const meta = () => get(ROOT + "meta.json", "json");
export const totals = () => get(ROOT + "totals.json", "json");
export const labels = async () => new Uint8Array(await get(ROOT + "labels.bin"));
export const pred = async () => new Uint8Array(await get(ROOT + "pred.bin"));
export const manifest = () => get("experiments/manifest.json", "json");

function lru(map, key, max) {
  const v = map.get(key);
  if (v) { map.delete(key); map.set(key, v); }
  return v || null;
}
function load(map, max, key, url, decode) {
  const v = lru(map, key, max);
  if (v) return Promise.resolve(v);
  if (!pending.has(key)) {
    const p = fetch(url).then(r => { if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return r.arrayBuffer(); })
      .then(buf => {
        const out = decode(buf);
        map.set(key, out);
        while (map.size > max) map.delete(map.keys().next().value);
        return out;
      })
      .finally(() => pending.delete(key));
    pending.set(key, p);
  }
  return pending.get(key);
}

// three 10-bit codes per image: x = centre + (code - 511.5) / 512 * half, with the site's centre and half-width
function decodeFrame(buf, q, N) {
  const u = new Uint32Array(buf), out = new Float32Array(N * 3), k = q.half / 512;
  if (u.length !== N) throw new Error(`frame has ${u.length} rows, expected ${N}`);
  for (let i = 0; i < N; i++) {
    const c = u[i];
    out[3 * i] = q.centre[0] + ((c & 1023) - 511.5) * k;
    out[3 * i + 1] = q.centre[1] + (((c >>> 10) & 1023) - 511.5) * k;
    out[3 * i + 2] = q.centre[2] + (((c >>> 20) & 1023) - 511.5) * k;
  }
  return out;
}

export const stageOf = (m, site) => m.stages.findIndex(s => s.sites.includes(site));
export const frameNow = (ckpt, site) => lru(frames, `${ckpt}_${site}`, MAX_FRAMES);
export function frame(m, ckpt, site) {
  if (!Number.isInteger(site) || site < 0 || site >= m.sites.length || !m.ckpts.includes(ckpt))
    return Promise.reject(new Error(`no frame for checkpoint ${ckpt}, site ${site}`));
  const q = m.quant[site];
  return load(frames, MAX_FRAMES, `${ckpt}_${site}`, `${ROOT}x/${ckpt}_${site}.bin`, buf => decodeFrame(buf, q, m.N));
}

// uint8 [rows, N]: 0 below the colour range, 1..253 bins, 254 above, 255 undefined
export const fieldNow = (id, ckpt) => lru(fieldBufs, `${id}_${ckpt}`, MAX_FIELDS);
export function field(f, ckpt, N) {
  return load(fieldBufs, MAX_FIELDS, `${f.id}_${ckpt}`, `${ROOT}f/${f.id}/${ckpt}.bin`, buf => {
    const a = new Uint8Array(buf);
    if (a.length !== f.rows * N) throw new Error(`${f.id} at ${ckpt}: ${a.length} bytes, expected ${f.rows * N}`);
    return a;
  });
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
