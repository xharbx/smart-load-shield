"""
Phase 1 -- Contingency-screening dataset (LEAKAGE-FREE reformulation).

For each sample:
  INPUT  = base-case (pre-contingency) operating state + a contingency mask
           (which branches are tripped).  The base state is the SAME for every
           contingency at a given operating point; only the topology differs.
  TARGET = post-contingency outcome obtained from the AC power-flow ORACLE:
             - post-trip minimum bus voltage  (regression)
             - risk class  (Stable >=0.95 / Marginal [0.90,0.95) / Unstable <0.90 or diverged)
             - per-bus post-trip vulnerability  1[V_i < 0.95]
             - divergence flag

The post-contingency voltages are NOT in the input -> the model must genuinely
predict the nonlinear power-flow outcome of a contingency it has not solved.

Output: redesign/contingency_data.npz
"""
import numpy as np, pandapower as pp, time, os, sys, itertools
from copy import deepcopy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ieee26_bus import build_ieee26, add_bus27_pv

rng = np.random.RandomState(42)
OUT = os.path.join(os.path.dirname(__file__), 'contingency_data.npz')
TARGET = 30000

# ---- base grid ----
net0 = build_ieee26(); add_bus27_pv(net0, pv_mw=100.0, vm_pu=1.0)
pp.runpp(net0, enforce_q_lims=True, numba=False, max_iteration=50)
base_load_p = net0.load['p_mw'].copy(); base_load_q = net0.load['q_mvar'].copy()
pv_gen_idx = [i for i in net0.gen.index if 'PV' in str(net0.gen.at[i,'name']) or 'Bus 27' in str(net0.gen.at[i,'name'])]

bus_ids = sorted(net0.bus.index.tolist()); n_bus = len(bus_ids)
b2i = {b:i for i,b in enumerate(bus_ids)}

# branches = lines + trafos (each trippable). edge_pairs aligned to branch order.
branches = [('line', i) for i in net0.line.index] + [('trafo', i) for i in net0.trafo.index]
n_branch = len(branches)
edge_pairs = []
for typ, bi in branches:
    if typ == 'line':
        f, t = int(net0.line.at[bi,'from_bus']), int(net0.line.at[bi,'to_bus'])
    else:
        f, t = int(net0.trafo.at[bi,'hv_bus']), int(net0.trafo.at[bi,'lv_bus'])
    edge_pairs.append((b2i[f], b2i[t]))
print(f"grid: {n_bus} buses, {n_branch} branches ({len(net0.line)} lines + {len(net0.trafo)} trafos)")

def solve(load_s, pv_mw, trip=None):
    sim = deepcopy(net0)
    sim.load['p_mw'] = base_load_p * load_s; sim.load['q_mvar'] = base_load_q * load_s
    for gi in pv_gen_idx: sim.gen.at[gi,'p_mw'] = pv_mw
    if trip:
        for typ, bi in trip:
            if typ == 'line': sim.line.at[bi,'in_service'] = False
            else: sim.trafo.at[bi,'in_service'] = False
    try:
        pp.runpp(sim, enforce_q_lims=True, numba=False, max_iteration=30)
        if sim.res_bus['vm_pu'].isna().any(): return sim, False
        return sim, True
    except Exception:
        return sim, False

def node_feats(sim):
    f = np.zeros((n_bus, 4), np.float32)
    for b in bus_ids:
        i = b2i[b]
        f[i,0] = sim.res_bus.at[b,'vm_pu']; f[i,1] = sim.res_bus.at[b,'va_degree']/180.0
        f[i,2] = sim.res_bus.at[b,'p_mw']/100.0; f[i,3] = sim.res_bus.at[b,'q_mvar']/100.0
    return f

def base_edge_feats(sim):
    # [loading%, P_from] per branch at the BASE case (in-service filled later per contingency)
    f = np.zeros((n_branch, 2), np.float32)
    for k,(typ,bi) in enumerate(branches):
        if typ == 'line' and bi in sim.res_line.index:
            f[k,0] = (sim.res_line.at[bi,'loading_percent'] or 0)/100.0
            f[k,1] = (sim.res_line.at[bi,'p_from_mw'] or 0)/100.0
        elif typ == 'trafo' and bi in sim.res_trafo.index:
            f[k,0] = (sim.res_trafo.at[bi,'loading_percent'] or 0)/100.0
            f[k,1] = (sim.res_trafo.at[bi,'p_hv_mw'] or 0)/100.0
    return np.nan_to_num(f)

def risk_of(vmin, conv):
    if not conv: return 2
    if vmin < 0.90: return 2
    if vmin < 0.95: return 1
    return 0

def sample_contingency():
    # k tripped branches; weighted toward 1-2 with some 0 (base) and a tail of 3-4
    k = rng.choice([0,1,1,1,1,2,2,2,3,4], 1)[0]
    if k == 0: return []
    idx = rng.choice(n_branch, size=k, replace=False)
    return [branches[j] for j in idx]

# ---- generate ----
NF, EF, INSVC, OPC = [], [], [], []          # base node feats, base edge feats, in-service mask, (load,pv)
T_vmin, T_risk, T_vuln, T_div, T_vbus = [], [], [], [], []
base_cache = {}
# Fixed pool of operating points so each base solve is amortized over many
# contingencies (big speedup vs a fresh operating point per sample).
OP_SET = [(round(float(rng.uniform(0.6,1.9)),3), round(float(rng.uniform(10,200)),1))
          for _ in range(480)]
t0 = time.time(); n=0; nb=0
while n < TARGET:
    load_s, pv_mw = OP_SET[rng.randint(len(OP_SET))]
    key = (load_s, pv_mw)
    if key not in base_cache:
        sb, cb = solve(load_s, pv_mw, None)
        if not cb: continue
        base_cache[key] = (node_feats(sb), base_edge_feats(sb)); nb += 1
    nf, ef = base_cache[key]

    trip = sample_contingency()
    sim, conv = solve(load_s, pv_mw, trip)
    if conv:
        vbus = np.array([sim.res_bus.at[b,'vm_pu'] for b in bus_ids], np.float32)
        vmin = float(np.nanmin(vbus)); vuln = (vbus < 0.95).astype(np.float32)
    else:
        vbus = np.zeros(n_bus, np.float32); vmin = 0.0; vuln = np.ones(n_bus, np.float32)

    insvc = np.ones(n_branch, np.float32)
    for typ, bi in trip:
        k = branches.index((typ, bi)); insvc[k] = 0.0

    NF.append(nf); EF.append(ef); INSVC.append(insvc); OPC.append([load_s, pv_mw])
    T_vmin.append(vmin); T_risk.append(risk_of(vmin, conv)); T_vuln.append(vuln); T_div.append(0.0 if conv else 1.0)
    T_vbus.append(vbus)
    n += 1
    if n % 2000 == 0:
        r = np.array(T_risk)
        print(f"  {n}/{TARGET}  base_pts={nb}  S/M/U={np.sum(r==0)}/{np.sum(r==1)}/{np.sum(r==2)}  [{time.time()-t0:.0f}s]")

NF=np.array(NF,np.float32); EF=np.array(EF,np.float32); INSVC=np.array(INSVC,np.float32)
OPC=np.array(OPC,np.float32); T_vmin=np.array(T_vmin,np.float32); T_risk=np.array(T_risk,np.int64)
T_vuln=np.array(T_vuln,np.float32); T_div=np.array(T_div,np.float32); T_vbus=np.array(T_vbus,np.float32)
nmean=NF.reshape(-1,4).mean(0); nstd=NF.reshape(-1,4).std(0); nstd[nstd<1e-6]=1
emean=EF.reshape(-1,2).mean(0); estd=EF.reshape(-1,2).std(0); estd[estd<1e-6]=1
r=T_risk
print(f"\nDONE {n} samples in {time.time()-t0:.0f}s | S/M/U = {np.sum(r==0)}/{np.sum(r==1)}/{np.sum(r==2)} | diverged={int(T_div.sum())}")
np.savez_compressed(OUT,
    node_feats=NF, edge_feats=EF, in_service=INSVC, opcond=OPC,
    t_vmin=T_vmin, t_risk=T_risk, t_vuln=T_vuln, t_div=T_div, t_vbus=T_vbus,
    edge_pairs=np.array(edge_pairs,np.int32), bus_ids=np.array(bus_ids,np.int32),
    node_mean=nmean, node_std=nstd, edge_mean=emean, edge_std=estd,
    n_bus=np.int32(n_bus), n_branch=np.int32(n_branch))
print("saved", OUT, f"({os.path.getsize(OUT)/1e6:.1f} MB)")
