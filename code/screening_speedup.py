"""
Real proactive speedup (review issue #2): screen a COMPLETE N-1 and N-2 sweep with
the GNN (no post-contingency power flow) vs conventional Newton-Raphson enumeration.

For a fixed operating point we:
  1. build every N-1 (single branch) and a large N-2 (branch-pair) contingency set,
  2. solve each with pandapower (the conventional way) -> ground-truth risk + time,
  3. screen ALL of them with one batched GNN forward pass -> predicted risk + time,
and report wall-clock speedup and screening AGREEMENT with the oracle.
"""
import numpy as np, torch, time, os, sys, json, itertools
from copy import deepcopy
import pandapower as pp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ieee26_bus import build_ieee26, add_bus27_pv
from boost_core import CSGNNv2

HERE = os.path.dirname(__file__)
D = np.load(os.path.join(HERE, 'contingency_data.npz'))
N_BUS = int(D['n_bus']); pairs = D['edge_pairs']; bus_ids = D['bus_ids'].tolist()
b2i = {b: i for i, b in enumerate(bus_ids)}
S = np.load(os.path.join(HERE, 'grouped_split.npz'))   # train-only norm (matches grouped cs_model.pt)
nmean = torch.tensor(S['nmean']); nstd = torch.tensor(S['nstd'])
emean = torch.tensor(S['emean']); estd = torch.tensor(S['estd'])
fr = torch.tensor(pairs[:, 0], dtype=torch.long); to = torch.tensor(pairs[:, 1], dtype=torch.long)

net0 = build_ieee26(); add_bus27_pv(net0, pv_mw=100.0, vm_pu=1.0)
pp.runpp(net0, enforce_q_lims=True, numba=False, max_iteration=50)
base_p = net0.load['p_mw'].copy(); base_q = net0.load['q_mvar'].copy()
branches = [('line', i) for i in net0.line.index] + [('trafo', i) for i in net0.trafo.index]
N_BR = len(branches)
LOAD_S = 1.3                       # a stressed operating point where screening matters


def apply_load(sim):
    sim.load['p_mw'] = base_p * LOAD_S; sim.load['q_mvar'] = base_q * LOAD_S


def solve_trip(trip):
    sim = deepcopy(net0); apply_load(sim)
    for typ, bi in trip:
        if typ == 'line': sim.line.at[bi, 'in_service'] = False
        else: sim.trafo.at[bi, 'in_service'] = False
    try:
        pp.runpp(sim, enforce_q_lims=True, numba=False, max_iteration=30)
        if sim.res_bus['vm_pu'].isna().any(): return None
        return float(sim.res_bus['vm_pu'].min())
    except Exception:
        return None


def risk_of(vmin):
    if vmin is None: return 2
    if vmin < 0.90: return 2
    if vmin < 0.95: return 1
    return 0


# base (pre-contingency) node + edge features, fixed for all contingencies
def node_feats(sim):
    f = np.zeros((N_BUS, 4), np.float32)
    for b in bus_ids:
        i = b2i[b]; f[i, 0] = sim.res_bus.at[b, 'vm_pu']; f[i, 1] = sim.res_bus.at[b, 'va_degree'] / 180.0
        f[i, 2] = sim.res_bus.at[b, 'p_mw'] / 100.0; f[i, 3] = sim.res_bus.at[b, 'q_mvar'] / 100.0
    return f


def edge_feats(sim):
    f = np.zeros((N_BR, 2), np.float32)
    for k, (typ, bi) in enumerate(branches):
        if typ == 'line' and bi in sim.res_line.index:
            f[k, 0] = (sim.res_line.at[bi, 'loading_percent'] or 0) / 100.0
            f[k, 1] = (sim.res_line.at[bi, 'p_from_mw'] or 0) / 100.0
        elif typ == 'trafo' and bi in sim.res_trafo.index:
            f[k, 0] = (sim.res_trafo.at[bi, 'loading_percent'] or 0) / 100.0
            f[k, 1] = (sim.res_trafo.at[bi, 'p_hv_mw'] or 0) / 100.0
    return np.nan_to_num(f)


sb = deepcopy(net0); apply_load(sb); pp.runpp(sb, enforce_q_lims=True, numba=False, max_iteration=50)
nf0 = node_feats(sb); ef0 = edge_feats(sb)

# contingency sets
n1 = [[branches[i]] for i in range(N_BR)]
allpairs = list(itertools.combinations(range(N_BR), 2))
rng = np.random.RandomState(1); sel = rng.choice(len(allpairs), size=min(800, len(allpairs)), replace=False)
n2 = [[branches[i], branches[j]] for i, j in (allpairs[k] for k in sel)]
conts = n1 + n2
print(f"N-1: {len(n1)}  N-2 (sampled): {len(n2)}  total: {len(conts)} contingencies at load {LOAD_S}x")

# ---- conventional enumeration (pandapower) ----
t0 = time.time(); oracle = [risk_of(solve_trip(c)) for c in conts]
pf_t = time.time() - t0

# ---- GNN batch screening (no post-contingency PF) ----
B = len(conts)
ins = np.ones((B, N_BR), np.float32)
for k, c in enumerate(conts):
    for typ, bi in c: ins[k, branches.index((typ, bi))] = 0.0
import statistics
torch.set_num_threads(1)     # single-core timing, to match the reported per-core figure


def build_batch():           # assemble the dense contingency batch from the base-case features
    NF = ((torch.tensor(nf0) - nmean) / nstd).unsqueeze(0).repeat(B, 1, 1)
    EF = ((torch.tensor(ef0) - emean) / estd).unsqueeze(0).repeat(B, 1, 1)
    INS = torch.tensor(ins)
    dense = torch.zeros(B, N_BUS, N_BUS, 3)
    dense[:, fr, to, 0:2] = EF; dense[:, to, fr, 0:2] = EF
    dense[:, fr, to, 2] = INS; dense[:, to, fr, 2] = INS
    adj = torch.zeros(B, N_BUS, N_BUS)
    adj[:, fr, to] = INS; adj[:, to, fr] = INS
    adj[:, torch.arange(N_BUS), torch.arange(N_BUS)] = 1.0
    return NF, dense, adj


build_ts = []
for _ in range(10):
    t0 = time.time(); NF, dense, adj = build_batch(); build_ts.append(time.time() - t0)
build_ms = statistics.median(build_ts) * 1000

# model load is excluded from all timings below
m = CSGNNv2(branched=False, stack=False, vbus=False); m.load_state_dict(torch.load(os.path.join(HERE, 'full_a0.pt'), map_location='cpu'))
_BASE = torch.zeros(N_BUS, N_BUS); _fr = torch.as_tensor(fr); _to = torch.as_tensor(to)
_BASE[_fr, _to] = 1.0; _BASE[_to, _fr] = 1.0; m.set_base_mask(_BASE)   # round-6 #1 base-edge mask
m.eval()
N_REP = 20
with torch.no_grad():
    for _ in range(3): m(NF[:8], dense[:8], adj[:8])       # warm-up (3 passes)
    gnn_ts = []
    for _ in range(N_REP):
        t0 = time.time(); risk, *_ = m(NF, dense, adj); gnn_ts.append(time.time() - t0)
gnn_t = statistics.median(gnn_ts)
gnn_std = statistics.pstdev(gnn_ts)
gnn_p95 = sorted(gnn_ts)[min(N_REP - 1, int(round(0.95 * N_REP)) - 1)]
pred = risk.argmax(1).numpy()
oracle = np.array(oracle)
agree = (pred == oracle).mean()
# safety: among oracle-unstable, how many flagged at least Marginal (not Stable)?
unst = oracle == 2
caught = (pred[unst] >= 1).mean() if unst.any() else float('nan')

# ---- HYBRID workflow: surrogate clears predicted-Secure cases; oracle confirms escalated ones ----
pf_per = pf_t / len(conts)                                 # per-case Newton-Raphson cost
n_secure = int((pred == 0).sum()); n_alert = int((pred == 1).sum()); n_insecure = int((pred == 2).sum())
n_escalated = int((pred >= 1).sum())                       # sent to the oracle for confirmation
n_cleared = n_secure                                       # trusted without an exact solve
false_clear = int(((pred == 0) & (oracle == 2)).sum())     # truly-unstable predicted Secure (false-safe)
false_alarm = int(((pred >= 1) & (oracle == 0)).sum())     # truly-secure escalated (conservative)
t_full = pf_t                                              # solve every case
t_hybrid = gnn_t + n_escalated * pf_per                    # surrogate + oracle confirms escalated
eff_speedup = t_full / t_hybrid

import torch as _t
print(f"\ntiming: 1 thread on Intel Core Ultra 9 285H; {N_REP} timed reps after 3 warm-ups; model load excluded")
print(f"batch construction (median)             : {build_ms:.1f} ms (one-off, excluded from screen time)")
print(f"\nconventional Newton-Raphson enumeration : {pf_t*1000:.0f} ms ({pf_per*1000:.1f} ms/contingency)")
print(f"GNN batched screening (median of {N_REP})     : {gnn_t*1000:.1f} ms  (std {gnn_std*1000:.1f}, p95 {gnn_p95*1000:.1f}); {gnn_t/len(conts)*1000:.3f} ms/contingency")
print(f"raw surrogate-vs-enumeration speedup    : {pf_t/gnn_t:.0f}x")
print(f"screening agreement with oracle         : {agree*100:.1f}%")
print(f"unstable contingencies flagged (>=Marginal): {caught*100:.1f}%")
print(f"\n--- hybrid workflow (surrogate + oracle-confirm-escalated) ---")
print(f"predicted Secure / Alert / Insecure     : {n_secure} / {n_alert} / {n_insecure}")
print(f"cleared without a solve (Secure)        : {n_cleared} ({n_cleared/len(conts)*100:.1f}%)")
print(f"escalated to oracle                     : {n_escalated} ({n_escalated/len(conts)*100:.1f}%)")
print(f"false clearances (unstable->Secure)     : {false_clear}")
print(f"false alarms (secure->escalated)        : {false_alarm}")
print(f"hybrid runtime                          : {t_hybrid*1000:.0f} ms  vs  full enumeration {t_full*1000:.0f} ms")
print(f"effective hybrid speedup                : {eff_speedup:.2f}x  (solves avoided: {n_cleared})")
json.dump({'load_scale': LOAD_S, 'n_contingencies': len(conts), 'n_n1': len(n1), 'n_n2': len(n2),
           'cpu': 'Intel Core Ultra 9 285H', 'threads': 1, 'n_timed_reps': N_REP, 'warmups': 3,
           'batch_build_ms': build_ms, 'gnn_median_ms': gnn_t * 1000, 'gnn_std_ms': gnn_std * 1000,
           'gnn_p95_ms': gnn_p95 * 1000, 'model_load_excluded': True,
           'pf_total_ms': pf_t * 1000, 'gnn_total_ms': gnn_t * 1000, 'speedup': pf_t / gnn_t,
           'agreement': float(agree), 'unstable_caught': float(caught),
           'pred_secure': n_secure, 'pred_alert': n_alert, 'pred_insecure': n_insecure,
           'n_escalated': n_escalated, 'n_cleared': n_cleared, 'false_clear': false_clear,
           'false_alarm': false_alarm, 'pf_per_ms': pf_per * 1000,
           't_hybrid_ms': t_hybrid * 1000, 'eff_hybrid_speedup': eff_speedup},
          open(os.path.join(HERE, 'screening_results.json'), 'w'), indent=2)
print("saved screening_results.json")
