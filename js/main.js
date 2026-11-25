// Page shell: three hash-routed tabs (#overview, #explorer[?state], #experiments), maths, the experiments index.
import { manifest } from "./data.js";

const TABS = ["overview", "explorer", "experiments"];
let explorer = null, listed = false;

async function route() {
  const raw = location.hash.slice(1), qi = raw.indexOf("?");
  const name = qi < 0 ? raw : raw.slice(0, qi), query = new URLSearchParams(qi < 0 ? "" : raw.slice(qi + 1));
  const [tab0, anchor] = name.split("/");
  const tab = TABS.includes(tab0) ? tab0 : "overview";
  const changed = document.body.dataset.tab !== tab;
  document.body.dataset.tab = tab;
  document.querySelectorAll(".tab-panel").forEach(p => { p.hidden = p.dataset.tab !== tab; });
  document.querySelectorAll(".tablist a").forEach(a => a.setAttribute("aria-selected", String(a.dataset.tab === tab)));
  if (tab === "explorer") {
    if (!explorer) {
      const { Explorer } = await import("./explorer.js");
      explorer = window.__ex = new Explorer(document.getElementById("ex"), query);
      await explorer.ready;
    } else if ([...query.keys()].length && location.hash !== explorer.lastHash) explorer.applyParams(query);
  } else if (explorer) explorer.stop?.();
  if (tab === "experiments" && !listed) { listed = true; listExperiments(); }
  const target = anchor && document.getElementById(anchor);
  if (target) requestAnimationFrame(() => target.scrollIntoView({ block: "start" }));
  else if (changed) window.scrollTo(0, 0);
}

async function listExperiments() {
  const box = document.getElementById("x-list");
  let xs;
  try { xs = (await manifest()).experiments || []; }
  catch (e) { box.innerHTML = `<p class="placeholder">The experiment list could not be loaded (${e.message}).</p>`; return; }
  if (!xs.length) {
    box.innerHTML = `<aside class="callout empty-slot"><p class="callout-title">No experiments published yet</p>
      <p>Experiments will appear here once the kinematics library has been run and checked. Until then the
      <a href="#explorer">Explorer</a> shows the cloud itself.</p></aside>`;
    return;
  }
  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const items = list => `<ol class="x-items">${list.map(x => `<li class="x-item${x.deprecated ? " x-dep" : ""}">
    <a class="x-title" href="${esc(x.path)}">${esc(x.title)}</a>
    <div class="x-meta"><time>${esc(x.date)}</time>${x.status ? ` <span class="tag">${esc(x.status)}</span>` : ""}${x.deprecated ? ` <span class="tag tag-dep">deprecated</span>` : ""}</div>
    <p class="x-q">${esc(x.question)}</p></li>`).join("")}</ol>`;
  // entries flagged "deprecated": true in the manifest go in their own group below the current ones
  const cur = xs.filter(x => !x.deprecated), dep = xs.filter(x => x.deprecated);
  box.innerHTML = (cur.length ? items(cur) : `<p class="mute">No current experiments yet.</p>`)
    + (dep.length ? `<h2 class="x-group" id="deprecated">Deprecated (Fisher lens)</h2>
      <p class="x-group-note">These measured the cloud in the Fisher space of logit-lens outputs, a readout withdrawn on
      2026-11-24. They are kept as they were, each with a notice at the top.</p>${items(dep)}` : "");
}

function maths() {
  if (!window.renderMathInElement) return;
  window.renderMathInElement(document.getElementById("tab-overview"), {
    delimiters: [{ left: "$$", right: "$$", display: true }, { left: "$", right: "$", display: false }],
    throwOnError: false,
  });
}

window.addEventListener("hashchange", route);
function init() {
  if (window.renderMathInElement) maths(); else window.addEventListener("load", maths);
  route();
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
