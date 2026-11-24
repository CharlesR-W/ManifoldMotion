import json, numpy as np, glob, os, sys
R="/home/ubuntu/code/ManifoldMotion/data/fields/r20s0"
STEPS=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128,256,512,1024,1280,2560,3840,5120,6400,7680,8960,11520,14080,17920,21760,25600,26880,33280,42240,51200,64000]
SITES=["stem"]+[f"b{i}" for i in range(1,10)]
meta=json.load(open(f"{R}/fields_meta.json")); names=[f['name'] for f in meta]
# verification
bad=[]
for s in STEPS:
    z=np.load(f"{R}/depth_{s}.npz")
    if set(z.files)!=set(names): bad.append((s,'names',sorted(set(names)^set(z.files))))
    for n in names:
        a=z[n]
        if a.shape!=(9,10000) or a.dtype!=np.float32: bad.append((s,n,a.shape,a.dtype))
    t=json.load(open(f"{R}/totals_{s}.json"))
    if len(t)!=9: bad.append((s,'totals',len(t)))
ntrain=len(glob.glob(f"{R}/extra/train_*.json"))
print("verify: bad", bad[:5], "n bad", len(bad), "train files", ntrain, "extra", sorted(os.listdir(f"{R}/extra"))[:3], len(os.listdir(f"{R}/extra")))
sp=np.load("/home/ubuntu/code/ManifoldMotion/data/fields/r20s0/depth_64000.npz")["speed"]
X=2*np.exp(np.load("/home/ubuntu/code/ManifoldMotion/data/lens/r20s0/logp_64000.npy").astype(np.float64)/2)
ref=4*np.arctan2(np.linalg.norm(X[:-1]-X[1:],axis=-1),np.linalg.norm(X[:-1]+X[1:],axis=-1))
print("speed vs make_speed_field max abs diff", float(np.max(np.abs(sp-ref))))

# ---------------- results
def tot(step, run="r20s0"):
    return json.load(open(f"/home/ubuntu/code/ManifoldMotion/data/fields/{run}/totals_{step}.json"))
def v(row,key):
    e=row.get(key)
    return (None,0.0) if e is None else (e['value'], e['mass_fraction'])
out={}
cov=json.load(open(f"{R}/extra/coverage_64000.json"))
out['coverage_final']={SITES[l]+'->'+SITES[l+1]:{m:{r:round(x,4) for r,x in c[m].items()} for m in c} for l,c in enumerate(cov)}
HEAD=["flow:ke","flow:ke_rigid","flow:ke_nonrigid","flow:rigid_fraction_global","flow:relmotion","flow:relmotion_shift","flow:relmotion_slope","flow:relmotion_icres",
      "diag:anchored:flow:relmotion_fit","diag:anchored:flow:relmotion_res","median:flow:heldout","median:flow:heldout_ic","diag:anchored:median:flow:heldout","median:flow:residual_fraction","median:flow:heldout_identity","median:flow:heldout_const","median:flow:slope_fraction","median:flow:shift_fraction","median:flow:intercept_rel",
      "flow:ic_log_gain_max","flow:ic_E_trace","flow:ic_strain_sq","flow:ic_spin_sq",
      "strict:fin_vol","strict:fin_shear","strict:rate_vol","strict:rate_shear","strict:logJ",
      "thick:tan_fraction","thick:tilt_max","direct:nn_log_ratio","direct:vertex_distance_from","direct:vertex_distance_to","direct:near_vertex_to"]
t=tot(64000)
out['final']={k:[v(t[l],k) for l in range(9)] for k in HEAD}
for f in sorted(glob.glob(f"{R}/extra/scale_*.json")):
    s=json.load(open(f)); out['scale:'+os.path.basename(f)]={k:[v(s['totals'][l],k) for l in range(9)] for k in HEAD}
    out['scale_radius:'+os.path.basename(f)]=[d['radius_q'][1] for d in s['diag']]
d0=json.load(open(f"{R}/extra/diag_64000.json")); out['radius_k256']=[d['radius_q'][1] for d in d0]
out['diag_final']=d0
if os.path.exists(f"{R}/extra/subsample.json"):
    s=json.load(open(f"{R}/extra/subsample.json"))
    def vals(l,k): return [r['totals'][l][k]['value'] for r in s['replicates'] if (r['totals'][l].get(k) or {}).get('value') is not None]
    out['subsample']={k:[(float(np.mean(vals(l,k))) if vals(l,k) else None, float(np.std(vals(l,k),ddof=1)) if len(vals(l,k))>1 else None) for l in range(9)] for k in HEAD}
p="/home/ubuntu/code/ManifoldMotion/data/fields/r20s1/totals_64000.json"
if os.path.exists(p):
    t1=json.load(open(p)); out['seed1']={k:[v(t1[l],k) for l in range(9)] for k in HEAD}
KT=["flow:ke","flow:rigid_fraction_global","flow:relmotion","median:flow:slope_fraction","median:flow:shift_fraction","median:flow:residual_fraction","median:flow:heldout_ic","direct:nn_log_ratio","direct:near_vertex_to","flow:ic_E_trace"]
out['depth_over_training']={k:{s:[v(tot(s)[l],k)[0] for l in range(9)] for s in STEPS} for k in KT}
tr={}
for f in sorted(glob.glob(f"{R}/extra/train_*.json")):
    s=json.load(open(f)); tr[s['step_to']]={k:[(r.get(k) or {}).get('value') for r in s['totals']] for k in ("flow:ke","flow:relmotion","flow:rigid_fraction_global","direct:nn_log_ratio","median:flow:heldout_ic","flow:travel")}
    tr[s['step_to']]['dt']=s['dt']
out['training_clock']=tr
def _js(x):
    if isinstance(x, dict): return {k: _js(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_js(v) for v in x]
    if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
    return x
json.dump(_js(out),open("/var/tmp/mmfx/results.json","w"),indent=1,allow_nan=False)
print("wrote results")
