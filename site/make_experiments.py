"""Experiment pages and the Overview's numbers, from the measurement files.

Every number the site prints is a named fact with its source (file and key). Page templates in
site/experiments/src/<slug>.html write {{f:name}} for a fact and {{data}} for the page's embedded JSON; build()
renders them into <out>/experiments/<slug>.html and returns the facts, which build_site.py also substitutes into
index.html. Facts are written to site/facts.tsv (name, printed text, value, source) for fact-checking.

    uv run --with numpy python site/make_experiments.py      # print every fact (no files written)
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIELDS = ROOT / "data" / "fields" / "r20s0"
LENS = ROOT / "data" / "lens" / "r20s0"
FIX = ROOT / "data" / "fixture_matched"
SUMMARY = "data/fields/r20s0/extra/summary_results.json"
SITES = ["stem", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9"]
STEPS9 = [f"{a}→{b}" for a, b in zip(SITES, SITES[1:])]
FINAL = 64000


def jload(p):
    return json.loads(Path(p).read_text())


SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(v):
    m, e = f"{v:.2e}".split("e")
    return f"{float(m):.3g}×10{str(int(e)).translate(SUP)}"


BOUND_HI = {"cov.gain.max", "flow.gain.max", "tr.vertex.pre1024", "tr.gain.early.max", "tr.gain.late.max", "scale.frac.change",
            "scale.radius.change", "scale.eps1.radius", "scale.eps1.ho", "scale.ho.change", "sub.ke.sd",
            "sub.frac.sd", "sub.rigid.sd", "fx.fr.scatter.max", "fx.fr.15.max", "fx.cov.15.max", "cov.E.max_early", "cov.tilt.max",
            "strict.tangent.max", "hco.max", "hid.max", "fx.fr.curved.max"}
BOUND_LO = {"fx.cov.clean.min", "fit9.min.all", "split.mass.min", "tr.split.mass.min", "split.supp.min", "fit9.min"}


class Facts:
    def __init__(self):
        self.d = {}

    def add(self, name, value, src, kind="f2", text=None):
        """Register a fact. Facts printed as bounds ("at most", "up to": BOUND_HI; "at least": BOUND_LO) are never
        rounded inward: digits are added until the printed number is on the safe side."""
        if name in self.d:
            raise KeyError(f"duplicate fact {name}")
        if text is None:
            v = float(value)
            bound = "hi" if name in BOUND_HI else "lo" if name in BOUND_LO else None
            if kind in ("int", "sci"):
                text = f"{int(round(v)):,}" if kind == "int" else sci(v)
            else:
                pct = kind.startswith("pct")
                dec = {"f0": 0, "f1": 1, "f2": 2, "f3": 3, "x1": 1, "signed2": 2, "pct0": 0, "pct1": 1, "pct2": 2}[kind]
                x = 100 * v if pct else v
                for d in range(dec, dec + 4):
                    s = f"{x:+.{d}f}" if kind == "signed2" else f"{x:.{d}f}"
                    r = float(s)
                    if bound is None or (bound == "hi" and r >= x - 1e-12) or (bound == "lo" and r <= x + 1e-12) or r == 0:
                        break
                text = (s + "%" if pct else s).replace("-", "−")
        self.d[name] = dict(value=value, text=text, src=src)
        return value

    def text(self, name):
        return self.d[name]["text"]

    def fill(self, html, where):
        out, i = [], 0
        while True:
            j = html.find("{{f:", i)
            if j < 0:
                out.append(html[i:])
                break
            k = html.index("}}", j)
            name = html[j + 4:k]
            if name not in self.d:
                raise KeyError(f"{where}: unknown fact {name}")
            out.append(html[i:j])
            out.append(self.d[name]["text"])
            self.d[name].setdefault("used_in", set()).add(where)
            i = k + 2
        return "".join(out)

    def tsv(self):
        rows = ["name\tprinted\tvalue\tsource\tused in"]
        for n, f in self.d.items():
            rows.append(f"{n}\t{f['text']}\t{f['value']}\t{f['src']}\t{', '.join(sorted(f.get('used_in', [])))}")
        return "\n".join(rows) + "\n"


def collect():
    """-> (facts, data per page)."""
    F = Facts()
    S = jload(ROOT / SUMMARY)
    fin = S["final"]
    steps = [int(k) for k in S["depth_over_training"]["flow:ke"]]
    v = lambda key: [None if r[0] is None else float(r[0]) for r in fin[key]]
    mass = lambda key: [float(r[1]) for r in fin[key]]

    # ---------------- page 1: depth at the final checkpoint
    diag = S["diag_final"]
    spec = [d["spectrum_median"] for d in diag]
    b7 = spec[7]
    F.add("b7.spectrum", b7, f"{SUMMARY} › diag_final[7].spectrum_median", text=", ".join(f"{x:.2f}".lstrip("0") if x < 1 else "1" for x in b7))
    F.add("b7.smallest", b7[-1], f"{SUMMARY} › diag_final[7].spectrum_median[8]", "f2")
    os_ = [d["one_sided_strict_median"] for d in diag]
    F.add("onesided.min", min(os_), f"{SUMMARY} › diag_final[*].one_sided_strict_median (min over stem..b8)", "f2")
    F.add("onesided.max", max(os_), f"{SUMMARY} › diag_final[*].one_sided_strict_median (max over stem..b8)", "f2")
    rad = [d["radius_q"][1] for d in diag]
    F.add("radius.stem", rad[0], f"{SUMMARY} › diag_final[0].radius_q[1] (median)", "f3")
    F.add("radius.b8", rad[8], f"{SUMMARY} › diag_final[8].radius_q[1] (median)", "f2")
    F.add("radius.b7", rad[7], f"{SUMMARY} › diag_final[7].radius_q[1] (median)", "f2")
    neff = [d["n_eff_q"][1] for d in diag]
    cov = S["coverage_final"]
    strict_tan = [cov[s]["strict"]["tangent"] for s in cov]
    strict_nb = [cov[s]["strict"]["neighbourhood"] for s in cov]
    F.add("strict.tangent.max", max(strict_tan), f"{SUMMARY} › coverage_final[*].strict.tangent (max over steps)", "pct2")
    F.add("strict.neighbourhood.max", max(strict_nb), f"{SUMMARY} › coverage_final[*].strict.neighbourhood (max over steps)", "pct1")
    thick_tan = [cov[s]["thick"]["tangent"] for s in cov]
    F.add("thick.tangent.max", max(thick_tan), f"{SUMMARY} › coverage_final[*].thick.tangent (max over steps)", "pct1")
    flow_rel = [cov[s]["flow"]["relmotion_fitsplit"] for s in cov]
    F.add("fit9.min", min(flow_rel), f"{SUMMARY} › coverage_final[*].flow.relmotion_fitsplit (min over steps)", "pct2")
    F.add("fit9.max", max(flow_rel), f"{SUMMARY} › coverage_final[*].flow.relmotion_fitsplit (max over steps)", "pct2")
    gain = [cov[s]["flow"]["gain_ic"] for s in cov]
    sfin = [cov[s]["flow"]["strain_finite_ic"] for s in cov]
    sic = [cov[s]["flow"]["strain_ic"] for s in cov]
    F.add("flow.gain.max", max(gain), f"{SUMMARY} › coverage_final[*].flow.gain_ic (max over steps)", "pct1")
    F.add("flow.strain.max_early", max(sic[:8]), f"{SUMMARY} › coverage_final[stem->b1..b7->b8].flow.strain_ic (max)", "pct2")
    F.add("flow.strain.last", sic[8], f"{SUMMARY} › coverage_final[b8->b9].flow.strain_ic", "pct0")
    F.add("flow.sfin.max", max(sfin), f"{SUMMARY} › coverage_final[*].flow.strain_finite_ic (max over steps)", "pct1")
    F.add("E.last", v("flow:ic_E_trace")[8], f"{SUMMARY} › final › flow:ic_E_trace[8][0] (mean over released images)", "f1")

    ke, ker, rig = v("flow:ke"), v("flow:ke_rigid"), v("flow:rigid_fraction_global")
    for i, s in enumerate(STEPS9):
        F.add(f"ke.{i}", ke[i], f"{SUMMARY} › final › flow:ke[{i}][0]", "f3" if ke[i] < 1 else "f2")
        F.add(f"rigid.{i}", rig[i], f"{SUMMARY} › final › flow:rigid_fraction_global[{i}][0]", "f2")
    F.add("rigid.final.min", min(rig), f"{SUMMARY} › final › flow:rigid_fraction_global (min over steps)", "f2")
    F.add("rigid.final.max", max(rig), f"{SUMMARY} › final › flow:rigid_fraction_global (max over steps)", "f2")
    F.add("ke.ratio", ke[8] / ke[7], f"{SUMMARY} › final › flow:ke[8][0] / flow:ke[7][0]", "x1")
    rel = v("flow:relmotion")
    T64 = jload(FIELDS / f"totals_{FINAL}.json")
    T64 = T64["steps"] if isinstance(T64, dict) else T64
    SP = "flow:split:relmotion_shift_slope_res:"
    sp = lambda k: [r[SP + k]["value"] for r in T64]
    tot, sh, sl, rs = sp("relmotion"), sp("relmotion_shift"), sp("relmotion_slope"), sp("relmotion_icres")
    shares = [[a / t, b / t, c / t] for a, b, c, t in zip(sh, sl, rs, tot)]
    closure = max(abs(a + b + c - t) / t for a, b, c, t in zip(sh, sl, rs, tot))
    F.add("split.closure", closure, f"data/fields/r20s0/totals_{FINAL}.json › [l].{SP}(shift + slope + icres - relmotion) / relmotion (max over steps)", "sci")
    F.add("split.supp.min", min(r[SP + "relmotion"]["mass_fraction"] for r in T64), f"data/fields/r20s0/totals_{FINAL}.json › [l].{SP}relmotion.mass_fraction (min)", "pct2")
    F.add("split.supp.max", max(r[SP + "relmotion"]["mass_fraction"] for r in T64), f"data/fields/r20s0/totals_{FINAL}.json › [l].{SP}relmotion.mass_fraction (max)", "pct2")
    AP = "diag:anchored:flow:split:relmotion_fit_res:"
    anch = T64[0][AP + "relmotion_fit"]["value"] / T64[0][AP + "relmotion"]["value"]
    F.add("anch.fit.stem", anch, f"data/fields/r20s0/totals_{FINAL}.json › [0].{AP}relmotion_fit / {AP}relmotion (diagnostic)", "pct1")
    F.add("slope.share.stem", shares[0][1], f"data/fields/r20s0/totals_{FINAL}.json › [0].{SP}relmotion_slope / {SP}relmotion", "pct1")
    F.add("relm.mass.min", min(mass("flow:relmotion")), f"{SUMMARY} › final › flow:relmotion[*][1] (min mass fraction)", "pct2")
    mres = v("median:flow:residual_fraction")
    F.add("resmed.min", min(mres), f"{SUMMARY} › final › median:flow:residual_fraction (min over steps)", "f2")
    F.add("resmed.max", max(mres), f"{SUMMARY} › final › median:flow:residual_fraction (max over steps)", "f2")
    msl, msh = v("median:flow:slope_fraction"), v("median:flow:shift_fraction")
    smass = [min(a, b) for a, b in zip(mass("median:flow:slope_fraction"), mass("flow:relmotion_shift"))]
    F.add("split.mass.min", min(smass), f"{SUMMARY} › final › median:flow:slope_fraction / flow:relmotion_shift [*][1] (min mass fraction)", "pct2")
    for i in range(9):
        F.add(f"slopemed.{i}", msl[i], f"{SUMMARY} › final › median:flow:slope_fraction[{i}][0]", "f2")
    F.add("slopemed.min", min(msl), f"{SUMMARY} › final › median:flow:slope_fraction (min over steps)", "f2")
    F.add("slopemed.max", max(msl), f"{SUMMARY} › final › median:flow:slope_fraction (max over steps)", "f2")
    F.add("slopeshare.max", max(x[1] for x in shares), f"data/fields/r20s0/totals_{FINAL}.json › [l].{SP}relmotion_slope / {SP}relmotion (max over steps)", "pct0")
    ho, hid, hco = v("median:flow:heldout_ic"), v("median:flow:heldout_identity"), v("median:flow:heldout_const")
    F.add("ho.min", min(ho[:8]), f"{SUMMARY} › final › median:flow:heldout_ic (min over stem->b1..b7->b8)", "f2")
    F.add("ho.max", max(ho[:8]), f"{SUMMARY} › final › median:flow:heldout_ic (max over stem->b1..b7->b8)", "f2")
    F.add("ho.last", ho[8], f"{SUMMARY} › final › median:flow:heldout_ic[8][0]", "f2")
    F.add("hco.max", max(hco[:8]), f"{SUMMARY} › final › median:flow:heldout_const (max over stem->b1..b7->b8)", "f2")
    F.add("hid.max", max(hid[:8]), f"{SUMMARY} › final › median:flow:heldout_identity (max over stem->b1..b7->b8)", "f2")
    nn, nv = v("direct:nn_log_ratio"), v("direct:near_vertex_to")
    for i in range(9):
        F.add(f"nn.{i}", nn[i], f"{SUMMARY} › final › direct:nn_log_ratio[{i}][0]", "signed2")
    F.add("nn.min_early", min(nn[:8]), f"{SUMMARY} › final › direct:nn_log_ratio (min over stem->b1..b7->b8)", "signed2")
    F.add("nn.max_early", max(nn[:8]), f"{SUMMARY} › final › direct:nn_log_ratio (max over stem->b1..b7->b8)", "signed2")
    F.add("nn.factor", math.exp(-nn[8]), f"exp(-{SUMMARY} › final › direct:nn_log_ratio[8][0])", "x1")
    F.add("vertex.last", nv[8], f"{SUMMARY} › final › direct:near_vertex_to[8][0]", "pct0")
    F.add("vertex.b8", nv[7], f"{SUMMARY} › final › direct:near_vertex_to[7][0] (share at b8)", "pct1")

    # coverage of every released field at the final checkpoint, straight from the site's field file
    fmeta = jload(FIELDS / "fields_meta.json")
    with np.load(FIELDS / f"depth_{FINAL}.npz") as z:
        fcov = [[float(np.isfinite(z[f["name"]][l]).mean()) for l in range(9)] for f in fmeta]
    flab = [f["label"] for f in fmeta]
    fgroup = [f.get("group", "") for f in fmeta]
    gi = flab.index("Largest log gain")
    F.add("cov.gain.max", max(fcov[gi]), f"data/fields/r20s0/depth_{FINAL}.npz › log_gain_max (share finite, max over steps)", "pct1")
    ei = [f["name"] for f in fmeta].index("E_trace")
    F.add("cov.E.last", fcov[ei][8], f"data/fields/r20s0/depth_{FINAL}.npz › E_trace[8] (share finite)", "pct0")
    F.add("cov.E.max_early", max(fcov[ei][:8]), f"data/fields/r20s0/depth_{FINAL}.npz › E_trace[0..7] (share finite, max)", "pct2")
    ti = [f["name"] for f in fmeta].index("tilt")
    F.add("cov.tilt.max", max(fcov[ti]), f"data/fields/r20s0/depth_{FINAL}.npz › tilt (share finite, max over steps)", "pct1")

    # robustness
    scales = [("k = 128", "scale:scale_k128_eps0.25.json", "scale_radius:scale_k128_eps0.25.json"),
              ("k = 256 (main)", "final", "radius_k256"),
              ("k = 512", "scale:scale_k512_eps0.25.json", "scale_radius:scale_k512_eps0.25.json"),
              ("k = 512, wider kernel", "scale:scale_k512_eps1.0.json", "scale_radius:scale_k512_eps1.0.json")]
    sc = []
    for name, key, rkey in scales:
        T = S[key]
        sc.append(dict(name=name, radius=S[rkey], slope=[r[0] for r in T["median:flow:slope_fraction"]],
                       heldout=[r[0] for r in T["median:flow:heldout_ic"]], shift=[r[0] for r in T["median:flow:shift_fraction"]]))
    r0 = np.array(S["radius_k256"])
    rch = [np.max(np.abs(np.array(S[k]) / r0 - 1)) for _, k2, k in scales if k != "radius_k256" and "eps1.0" not in k for _ in [0]]
    F.add("scale.radius.change", max(rch), f"{SUMMARY} › scale_radius:scale_k128/k512_eps0.25 vs radius_k256 (max relative change)", "pct0")
    def dmax(keys):
        return max(abs(a - b) for s_ in (sc[0], sc[2]) for key in keys for a, b in zip(s_[key], sc[1][key]) if a is not None and b is not None)
    F.add("scale.frac.change", dmax(("slope", "shift")), f"{SUMMARY} › scale:k128/k512_eps0.25 vs final: median slope and shift fractions (max |change|)", "f3")
    F.add("scale.ho.change", dmax(("heldout",)), f"{SUMMARY} › scale:k128/k512_eps0.25 vs final: median held-out score (max |change|)", "f2")
    F.add("scale.eps1.slope", max(abs(a - b) for a, b in zip(sc[3]["slope"], sc[1]["slope"])), f"{SUMMARY} › scale:scale_k512_eps1.0.json vs final: median:flow:slope_fraction (max |change|)", "f3")
    F.add("scale.eps1.ho", max(a - b for a, b in zip(sc[3]["heldout"], sc[1]["heldout"])), f"{SUMMARY} › scale:scale_k512_eps1.0.json minus final: median:flow:heldout_ic (max)", "f2")
    F.add("radius.ratio", rad[8] / rad[0], f"{SUMMARY} › diag_final[8].radius_q[1] / diag_final[0].radius_q[1]", "f0")
    F.add("scale.eps1.radius", np.max(np.array(S["scale_radius:scale_k512_eps1.0.json"]) / r0), f"{SUMMARY} › scale_radius:scale_k512_eps1.0.json / radius_k256 (max)", "x1")
    sub = S["subsample"]
    subj = jload(FIELDS / "extra" / "subsample.json")
    F.add("sub.n", subj["n_subsets"], "data/fields/r20s0/extra/subsample.json › n_subsets", "int")
    F.add("sub.size", subj["size"], "data/fields/r20s0/extra/subsample.json › size", "int")
    fr_keys = ["median:flow:slope_fraction", "median:flow:shift_fraction", "median:flow:heldout_ic", "median:flow:heldout_identity", "median:flow:heldout_const"]
    F.add("sub.frac.sd", max(r[1] for k in fr_keys for r in sub[k]), f"{SUMMARY} › subsample › median fractions and scores [*][1] (max sd)", "f3")
    F.add("sub.ke.sd", max(r[1] for r in sub["flow:ke"]), f"{SUMMARY} › subsample › flow:ke[*][1] (max sd)", "f3")
    F.add("sub.rigid.sd", max(r[1] for r in sub["flow:rigid_fraction_global"]), f"{SUMMARY} › subsample › flow:rigid_fraction_global[*][1] (max sd)", "f3")
    F.add("sub.ke.sd.last", sub["flow:ke"][8][1], f"{SUMMARY} › subsample › flow:ke[8][1]", "f3")
    s1 = S["seed1"]
    s1v = lambda key: [None if r[0] is None else float(r[0]) for r in s1[key]]
    F.add("seed1.ke.last", s1v("flow:ke")[8], f"{SUMMARY} › seed1 › flow:ke[8][0]", "f2")
    F.add("seed1.rigid.last", s1v("flow:rigid_fraction_global")[8], f"{SUMMARY} › seed1 › flow:rigid_fraction_global[8][0]", "f2")
    F.add("seed1.nn.last", s1v("direct:nn_log_ratio")[8], f"{SUMMARY} › seed1 › direct:nn_log_ratio[8][0]", "signed2")
    F.add("seed1.vertex.last", s1v("direct:near_vertex_to")[8], f"{SUMMARY} › seed1 › direct:near_vertex_to[8][0]", "pct0")

    depth = dict(steps=STEPS9, sites=SITES[:9], spectrum=spec, one_sided=os_, radius=rad, neff=neff,
                 strict_dim=[d["strict_dim"] for d in diag],
                 ke=ke, ke_rigid=ker, ke_nonrigid=v("flow:ke_nonrigid"), rigid=rig,
                 shares=shares, slope_med=msl, shift_med=msh, relmotion=rel,
                 ho=ho, hid=hid, hco=hco, nn=nn, near=nv, vfrom=v("direct:vertex_distance_from"), vto=v("direct:vertex_distance_to"),
                 cover=dict(labels=flab, groups=fgroup, values=fcov),
                 rules=dict(strict_tangent=strict_tan, strict_nb=strict_nb, flow_split=flow_rel, flow_gain=gain, flow_strain=sic, thick_tangent=thick_tan),
                 scales=sc, subsample={k: [r[1] for r in sub[k]] for k in ["flow:ke", "flow:rigid_fraction_global"] + fr_keys},
                 seed1=dict(ke=s1v("flow:ke"), rigid=s1v("flow:rigid_fraction_global"), slope=s1v("median:flow:slope_fraction"),
                            nn=s1v("direct:nn_log_ratio"), relmotion=s1v("flow:relmotion"), near=s1v("direct:near_vertex_to")))

    # ---------------- page 2: over training
    dot = S["depth_over_training"]
    M = {k: [dot[k][str(s)] for s in steps] for k in dot}
    kl = [r[8] for r in M["flow:ke"]]
    F.add("tr.ke.first", kl[0], f"{SUMMARY} › depth_over_training › flow:ke['0'][8]", "f2")
    F.add("tr.ke.min", min(kl), f"{SUMMARY} › depth_over_training › flow:ke[*][8] (min)", "f3")
    F.add("tr.ke.min.step", steps[int(np.argmin(kl))], f"{SUMMARY} › depth_over_training › flow:ke[*][8] (argmin step)", "int")
    F.add("tr.ke.last", kl[-1], f"{SUMMARY} › depth_over_training › flow:ke['64000'][8]", "f2")
    rl = [r[8] for r in M["flow:rigid_fraction_global"]]
    F.add("tr.rigid.first", rl[0], f"{SUMMARY} › depth_over_training › flow:rigid_fraction_global['0'][8]", "f2")
    F.add("tr.rigid.last", rl[-1], f"{SUMMARY} › depth_over_training › flow:rigid_fraction_global['64000'][8]", "f2")
    F.add("tr.rigid0.min", min(M["flow:rigid_fraction_global"][0]), f"{SUMMARY} › depth_over_training › flow:rigid_fraction_global['0'] (min over depth steps)", "f2")
    vl = [r[8] for r in M["direct:near_vertex_to"]]
    pre = [x for s, x in zip(steps, vl) if s < 1024]
    F.add("tr.vertex.pre1024", max(pre), f"{SUMMARY} › depth_over_training › direct:near_vertex_to[step < 1024][8] (max)", "pct1")
    F.add("tr.vertex.1024", vl[steps.index(1024)], f"{SUMMARY} › depth_over_training › direct:near_vertex_to['1024'][8]", "pct1")
    F.add("tr.vertex.2560", vl[steps.index(2560)], f"{SUMMARY} › depth_over_training › direct:near_vertex_to['2560'][8]", "pct0")
    F.add("tr.vertex.last", vl[-1], f"{SUMMARY} › depth_over_training › direct:near_vertex_to['64000'][8]", "pct0")
    nl = [r[8] for r in M["direct:nn_log_ratio"]]
    first_neg = next(s for s, x in zip(steps, nl) if x < 0)
    last_pos = max(s for s, x in zip(steps, nl) if x >= 0)
    F.add("tr.nn.firstneg", first_neg, f"{SUMMARY} › depth_over_training › direct:nn_log_ratio[*][8] (first step < 0)", "int")
    F.add("tr.nn.lastpos", last_pos, f"{SUMMARY} › depth_over_training › direct:nn_log_ratio[*][8] (last step >= 0)", "int")
    sl8 = [r[8] for r in M["median:flow:slope_fraction"]]
    F.add("tr.slope8.max", max(sl8), f"{SUMMARY} › depth_over_training › median:flow:slope_fraction[*][8] (max)", "f2")
    F.add("tr.slope8.argmax", steps[int(np.argmax(sl8))], f"{SUMMARY} › depth_over_training › median:flow:slope_fraction[*][8] (argmax step)", "int")
    F.add("tr.slope8.last", sl8[-1], f"{SUMMARY} › depth_over_training › median:flow:slope_fraction['64000'][8]", "f2")
    slall = np.array(M["median:flow:slope_fraction"])
    F.add("tr.slope.max", float(slall.max()), f"{SUMMARY} › depth_over_training › median:flow:slope_fraction (max over steps and depth)", "f2")
    F.add("tr.slope.max.step", steps[int(np.unravel_index(slall.argmax(), slall.shape)[0])], f"{SUMMARY} › depth_over_training › median:flow:slope_fraction (argmax step)", "int")
    trm = []
    for s in steps:
        rows = jload(FIELDS / f"totals_{s}.json")
        rows = rows["steps"] if isinstance(rows, dict) else rows
        trm.append([r["median:flow:slope_fraction"]["mass_fraction"] for r in rows])
    F.add("tr.split.mass.min", float(np.min(trm)), "data/fields/r20s0/totals_<step>.json › [l].median:flow:slope_fraction.mass_fraction (min over 40 checkpoints and 9 steps)", "pct2")
    covtr, fit9all = [], []
    for s in steps:
        c = jload(FIELDS / "extra" / f"coverage_{s}.json")
        covtr.append([c[l]["flow"]["gain_ic"] for l in range(9)])
        fit9all.append(min(c[l]["flow"]["relmotion_fitsplit"] for l in range(9)))
    F.add("fit9.min.all", min(fit9all), "data/fields/r20s0/extra/coverage_<step>.json › [l].flow.relmotion_fitsplit (min over 40 checkpoints and 9 steps)", "pct1")
    early = np.array([r for s, r in zip(steps, covtr) if s <= 16])
    F.add("tr.gain.early.max", float(early.max()), "data/fields/r20s0/extra/coverage_<step>.json (step <= 16) › [l].flow.gain_ic (max)", "pct0")
    F.add("tr.gain.late.max", float(np.max([r for s, r in zip(steps, covtr) if s >= 2560])), "data/fields/r20s0/extra/coverage_<step>.json (step >= 2560) › [l].flow.gain_ic (max)", "pct1")
    tc = S["training_clock"]
    tsteps = sorted(int(k) for k in tc)
    clock = dict(steps=tsteps, dt=[tc[str(s)]["dt"] for s in tsteps],
                 **{k.split(":")[-1]: [tc[str(s)][k] for s in tsteps] for k in ["flow:ke", "flow:rigid_fraction_global", "direct:nn_log_ratio", "flow:travel"]})
    kef, kelast = clock["ke"][0], clock["ke"][-1]
    ratios = [a / b for a, b in zip(kef, kelast)]
    F.add("tc.ratio.min", min(ratios), f"{SUMMARY} › training_clock['1'].flow:ke / ['64000'].flow:ke (min over sites)", "sci")
    F.add("tc.ratio.max", max(ratios), f"{SUMMARY} › training_clock['1'].flow:ke / ['64000'].flow:ke (max over sites)", "sci")
    peaks = [tsteps[int(np.argmax([row[i] for row in clock["ke"]]))] for i in range(10)]
    F.add("tc.peak.latest", max(peaks), f"{SUMMARY} › training_clock › flow:ke (argmax interval end step per site, max over sites)", "int")
    F.add("tc.peak.first.n", sum(p == tsteps[0] for p in peaks), f"{SUMMARY} › training_clock › flow:ke (sites whose argmax is the first interval)", "int")
    F.add("tc.rigid.stem", clock["rigid_fraction_global"][-1][0], f"{SUMMARY} › training_clock['64000'].flow:rigid_fraction_global[0]", "f2")
    F.add("tc.rigid.b9", clock["rigid_fraction_global"][-1][9], f"{SUMMARY} › training_clock['64000'].flow:rigid_fraction_global[9]", "f2")
    F.add("tc.dt.last", clock["dt"][-1], f"{SUMMARY} › training_clock['64000'].dt", "int")
    training = dict(steps=steps, steps9=STEPS9, M={k.split(":")[-1]: M[k] for k in M}, slope_mass=trm, gain_cov=covtr, clock=clock)

    # ---------------- page 3: calibration and the lens
    rep = jload(FIX / "rep_v4.json")
    ev = jload(FIX / "final_eval.json")
    reg = rep["regime"]["smooth_s0_k256_e0.25"]
    F.add("fx.radius", reg["radius_quantiles"][1], "data/fixture_matched/rep_v4.json › regime › smooth_s0_k256_e0.25.radius_quantiles[1]", "f2")
    F.add("fx.onesided", reg["one_sided_median"], "data/fixture_matched/rep_v4.json › regime › smooth_s0_k256_e0.25.one_sided_median", "f2")
    F.add("fx.radius.lo", reg["radius_quantiles"][0], "data/fixture_matched/rep_v4.json › regime › smooth_s0_k256_e0.25.radius_quantiles[0]", "f2")
    F.add("fx.radius.hi", reg["radius_quantiles"][2], "data/fixture_matched/rep_v4.json › regime › smooth_s0_k256_e0.25.radius_quantiles[2]", "f2")
    eve = {s: rep[f"edge_vs_endpoint_s{s}"] for s in (0, 1)}
    F.add("fx.anch.end", eve[0]["anch_endpoint"], "data/fixture_matched/rep_v4.json › edge_vs_endpoint_s0.anch_endpoint", "pct1")
    F.add("fx.ic.end", eve[0]["ic_endpoint"], "data/fixture_matched/rep_v4.json › edge_vs_endpoint_s0.ic_endpoint", "pct1")
    F.add("fx.anch.edge", eve[0]["anch_edge"], "data/fixture_matched/rep_v4.json › edge_vs_endpoint_s0.anch_edge", "pct1")
    F.add("fx.ic.edge", eve[0]["ic_edge"], "data/fixture_matched/rep_v4.json › edge_vs_endpoint_s0.ic_edge", "pct1")
    F.add("fx.anch.end.s1", eve[1]["anch_endpoint"], "data/fixture_matched/rep_v4.json › edge_vs_endpoint_s1.anch_endpoint", "pct1")
    F.add("fx.ic.end.s1", eve[1]["ic_endpoint"], "data/fixture_matched/rep_v4.json › edge_vs_endpoint_s1.ic_endpoint", "pct1")
    bm = rep["baselines_median"]
    nus = ["clean", "endpoint0.3", "endpoint0.6", "endpoint1.0", "endpoint1.5"]
    opt = [dict(name=n, heldout=ev[f"s0:smooth:{n}"]["heldout_ic_median"], oracle=ev[f"s0:smooth:{n}"]["oracle_median"],
                heldout_s1=ev[f"s1:smooth:{n}"]["heldout_ic_median"], oracle_s1=ev[f"s1:smooth:{n}"]["oracle_median"]) for n in nus]
    F.add("fx.ho.clean", opt[0]["heldout"], "data/fixture_matched/final_eval.json › s0:smooth:clean.heldout_ic_median", "f3")
    F.add("fx.or.clean", opt[0]["oracle"], "data/fixture_matched/final_eval.json › s0:smooth:clean.oracle_median", "f3")
    F.add("fx.ho.06", opt[2]["heldout"], "data/fixture_matched/final_eval.json › s0:smooth:endpoint0.6.heldout_ic_median", "f2")
    F.add("fx.or.06", opt[2]["oracle"], "data/fixture_matched/final_eval.json › s0:smooth:endpoint0.6.oracle_median", "f2")
    F.add("fx.ho.15", opt[4]["heldout"], "data/fixture_matched/final_eval.json › s0:smooth:endpoint1.5.heldout_ic_median", "f2")
    F.add("fx.or.15", opt[4]["oracle"], "data/fixture_matched/final_eval.json › s0:smooth:endpoint1.5.oracle_median", "f2")
    shuf = bm["smooth:shuffled"]
    F.add("fx.shuf.ho", shuf["heldout_ic"], "data/fixture_matched/rep_v4.json › baselines_median › smooth:shuffled.heldout_ic", "f2")
    F.add("fx.shuf.const", shuf["heldout_const"], "data/fixture_matched/rep_v4.json › baselines_median › smooth:shuffled.heldout_const", "f2")
    F.add("fx.clean.id", bm["smooth:clean"]["heldout_identity"], "data/fixture_matched/rep_v4.json › baselines_median › smooth:clean.heldout_identity", "f2")
    ct = rep["contrasts"]
    F.add("fx.rank8.true", ct["rank8:endpoint0.6"]["true_ls9_median"], "data/fixture_matched/rep_v4.json › contrasts › rank8:endpoint0.6.true_ls9_median", "f1")
    F.add("fx.rank8.est", ct["rank8:endpoint0.6"]["est_ls9_median"], "data/fixture_matched/rep_v4.json › contrasts › rank8:endpoint0.6.est_ls9_median", "f1")
    F.add("fx.rank8.loss", ct["rank8:endpoint0.6"]["rank_loss_frac"], "data/fixture_matched/rep_v4.json › contrasts › rank8:endpoint0.6.rank_loss_frac", "pct0")
    F.add("fx.rank8.loss.clean", ct["rank8:clean"]["rank_loss_frac"], "data/fixture_matched/rep_v4.json › contrasts › rank8:clean.rank_loss_frac", "pct0")
    F.add("fx.along.logJ", ct["weakcomp:along0.6"]["logJ"]["bias"], "data/fixture_matched/rep_v4.json › contrasts › weakcomp:along0.6.logJ.bias", "f2")
    F.add("fx.along.iclogJ", ct["weakcomp:along0.6"]["ic_logJ"]["bias"], "data/fixture_matched/rep_v4.json › contrasts › weakcomp:along0.6.ic_logJ.bias", "f2")
    F.add("fx.near.true", ct["nearsing:endpoint0.6"]["true_ls9_median"], "data/fixture_matched/rep_v4.json › contrasts › nearsing:endpoint0.6.true_ls9_median", "f1")
    F.add("fx.near.est", ct["nearsing:endpoint0.6"]["est_ls9_median"], "data/fixture_matched/rep_v4.json › contrasts › nearsing:endpoint0.6.est_ls9_median", "f1")
    tol = ev["tolerances"]
    F.add("tol.gain", tol["ic_log_gain_max"], "data/fixture_matched/final_eval.json › tolerances.ic_log_gain_max", "f1")
    variants = [k.split(":", 1)[1] for k in ev if k.startswith("s0:")]
    G = ev["gates"]
    for k, kind in [("heldout_min", "f1"), ("slope_gain_min", "f1"), ("cond_centred_max", "f0"), ("identity_gain_min", "f1"), ("identity_exact_min", "f2"), ("n_eff_per_dim_strain", "f0")]:
        F.add(f"gate.{k}", G[k], f"data/fixture_matched/final_eval.json › gates.{k}", kind)
    for sd in ("s0", "s1"):
        F.add(f"fx.gain.clean.{sd}", ev[f"{sd}:smooth:clean"]["ic_log_gain_max"]["coverage"], f"data/fixture_matched/final_eval.json › {sd}:smooth:clean.ic_log_gain_max.coverage", "pct1")
        F.add(f"fx.gain.10.{sd}", ev[f"{sd}:smooth:endpoint1.0"]["ic_log_gain_max"]["coverage"], f"data/fixture_matched/final_eval.json › {sd}:smooth:endpoint1.0.ic_log_gain_max.coverage", "pct1")
    for sd in ("s0", "s1"):
        pre = ev["pre_identity_gate"][f"{sd}:controls:identity_noise0.6"]["ic_log_gain_max"]
        post = ev[f"{sd}:controls:identity_noise0.6"]["ic_log_gain_max"]
        F.add(f"fx.idn.pre.cov.{sd}", pre["coverage"], f"data/fixture_matched/final_eval.json › pre_identity_gate › {sd}:controls:identity_noise0.6.ic_log_gain_max.coverage", "pct1")
        F.add(f"fx.idn.pre.fr.{sd}", pre["false_release"], f"data/fixture_matched/final_eval.json › pre_identity_gate › {sd}:controls:identity_noise0.6.ic_log_gain_max.false_release", "pct0")
        F.add(f"fx.idn.cov.{sd}", post["coverage"], f"data/fixture_matched/final_eval.json › {sd}:controls:identity_noise0.6.ic_log_gain_max.coverage", "pct2")
        F.add(f"fx.idn.fr.{sd}", post["false_release"], f"data/fixture_matched/final_eval.json › {sd}:controls:identity_noise0.6.ic_log_gain_max.false_release", "pct1")
    ftab = {}
    for f in ["ic_log_gain_max", "ic_E_trace", "ic_strain_sq", "ic_spin_sq", "slope_fraction", "heldout_ic"]:
        ftab[f] = {sd: [ev[f"{sd}:{vv}"][f] for vv in variants] for sd in ("s0", "s1")}
    fr_smooth = [ev[f"{sd}:smooth:{n}"][f].get("false_release") for sd in ("s0", "s1") for n in nus[1:4]
                 for f in ["ic_log_gain_max", "ic_E_trace", "ic_strain_sq", "ic_spin_sq"]]
    fr_smooth = [x for x in fr_smooth if x is not None]
    F.add("fx.fr.scatter.max", max(fr_smooth), "data/fixture_matched/final_eval.json › s0/s1:smooth:endpoint0.3/0.6/1.0.{4 ic fields}.false_release (max)", "pct0")
    hi15 = [ev[f"{sd}:smooth:endpoint1.5"][f] for sd in ("s0", "s1") for f in ["ic_log_gain_max", "ic_E_trace", "ic_strain_sq", "ic_spin_sq"]]
    F.add("fx.fr.15.max", max(r["false_release"] for r in hi15), "data/fixture_matched/final_eval.json › s0/s1:smooth:endpoint1.5.{4 ic fields}.false_release (max)", "pct0")
    F.add("fx.cov.15.max", max(r["coverage"] for r in hi15), "data/fixture_matched/final_eval.json › s0/s1:smooth:endpoint1.5.{4 ic fields}.coverage (max)", "pct0")
    cl = [ev[f"{sd}:smooth:clean"][f]["coverage"] for sd in ("s0", "s1") for f in ["ic_log_gain_max", "ic_E_trace", "ic_strain_sq", "ic_spin_sq"]]
    F.add("fx.cov.clean.min", min(cl), "data/fixture_matched/final_eval.json › s0/s1:smooth:clean.{4 ic fields}.coverage (min)", "pct0")
    fr_curved = [ev[f"{sd}:curved:clean"][f].get("false_release") for sd in ("s0", "s1") for f in ["ic_log_gain_max", "ic_E_trace", "ic_strain_sq", "ic_spin_sq"]]
    fr_curved = [x for x in fr_curved if x is not None]
    F.add("fx.fr.curved.min", min(fr_curved), "data/fixture_matched/final_eval.json › s0/s1:curved:clean.{ic fields}.false_release (min)", "pct0")
    F.add("fx.fr.curved.max", max(fr_curved), "data/fixture_matched/final_eval.json › s0/s1:curved:clean.{ic fields}.false_release (max)", "pct0")
    # lens checks over all 73 checkpoints
    lm = jload(LENS / "meta.json")
    worst, wsrc, acc, passed = 0.0, "", [], True
    for s in lm["steps"]:
        c = jload(LENS / f"checks_{s}.json")
        passed &= bool(c["passed"])
        acc.append(c["lens_top1_accuracy"])
        for part in "abc":
            items = [("", c["a"])] if part == "a" else list(c[part].items())
            for k, val in items:
                if val["max_abs"] > worst:
                    worst, wsrc = val["max_abs"], f"checks_{s}.json › {part}{'.' + k if k else ''}.max_abs"
    F.add("lens.n", len(lm["steps"]), "data/lens/r20s0/meta.json › steps (count)", "int")
    F.add("lens.worst", worst, f"data/lens/r20s0/{wsrc} (max over all checks and checkpoints)", "sci")
    F.add("lens.passed", 1.0 if passed else 0.0, "data/lens/r20s0/checks_*.json › passed (all)", text="all" if passed else "not all")
    a64 = acc[lm["steps"].index(FINAL)]
    for i, s in enumerate(SITES):
        F.add(f"acc.{s}", a64[i], f"data/lens/r20s0/checks_{FINAL}.json › lens_top1_accuracy[{i}]", "pct2" if i == 9 else "pct0")
    F.add("acc.model", jload(LENS / f"checks_{FINAL}.json")["model_top1_accuracy"], f"data/lens/r20s0/checks_{FINAL}.json › model_top1_accuracy", "pct2")
    F.add("acc.mid.min", min(a64[2:7]), f"data/lens/r20s0/checks_{FINAL}.json › lens_top1_accuracy[2..6] (min)", "pct0")
    F.add("acc.mid.max", max(a64[2:7]), f"data/lens/r20s0/checks_{FINAL}.json › lens_top1_accuracy[2..6] (max)", "pct0")
    calib = dict(real_b7=dict(spectrum=spec[7], radius=diag[7]["radius_q"], one_sided=os_[7], neff=diag[7]["n_eff_q"]),
                 fixture=dict(spectrum=reg["spectrum_median"], radius=reg["radius_quantiles"], one_sided=reg["one_sided_median"],
                              neff=reg["n_eff_median"], controls=rep["regime"]["controls_s0_k256_e0.25"]["spectrum_median"]),
                 eve=eve, opt=opt, baselines={k: bm[k] for k in bm}, variants=variants, ftab=ftab, tol=tol,
                 contrasts={k: {kk: ct[k][kk] for kk in ("rank_loss_frac", "true_ls9_median", "est_ls9_median")} | {"logJ_bias": (ct[k].get("logJ") or {}).get("bias"), "ic_logJ_bias": (ct[k].get("ic_logJ") or {}).get("bias")}
                            for k in ("rank8:clean", "rank8:endpoint0.6", "nearsing:clean", "nearsing:endpoint0.6", "weakcomp:endpoint0.6", "weakcomp:along0.6", "weakcomp:orth0.6")},
                 lens=dict(steps=lm["steps"], acc=acc, sites=SITES))
    return F, dict(depth=depth, training=training, calibration=calib)


def tojson(o):
    def clean(x):
        if isinstance(x, float):
            return None if not math.isfinite(x) else float(f"{x:.6g}")
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.floating,)):
            return clean(float(x))
        if isinstance(x, (np.integer,)):
            return int(x)
        return x
    return json.dumps(clean(o), separators=(",", ":")).replace("</", "<\\/")


def build(out):
    """Render the experiment pages into <out>/experiments/ and return the facts (for index.html)."""
    F, data = collect()
    src = HERE / "experiments" / "src"
    shell = (src / "_shell.html").read_text()
    man = jload(HERE / "experiments" / "manifest.json")
    for x in man["experiments"]:
        slug = Path(x["path"]).stem
        body = (src / f"{slug}.html").read_text()
        page = shell.replace("{{title}}", x["title"]).replace("{{body}}", body).replace("{{data}}", tojson(data[slug])).replace("{{slug}}", slug)
        page = F.fill(page, f"experiments/{slug}.html")
        (Path(out) / "experiments" / f"{slug}.html").write_text(page)
    return F


if __name__ == "__main__":
    F, _ = collect()
    for n, f in F.d.items():
        print(f"{n:28s} {f['text']:>14s}   {f['src']}")
