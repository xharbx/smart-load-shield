"""
IEEE 118-bus contingency-screening dataset (leakage-free formulation).

Same task definition, same features, same labels, same thresholds as the 27-bus
generator (`gen_contingency_data.py`); only the network and the operating-point
parameterisation change.  Every design choice that DIFFERS from the 27-bus run is
listed here so it can be stated in the paper and in the response letter.

NETWORK
  pandapower.networks.case118 -- the standard IEEE 118-bus test case (MATPOWER
  case118.m, from the University of Washington Power Systems Test Case Archive;
  a portion of the American Electric Power system, circa 1962).  Public,
  citable, and loaded verbatim from the library: no hand transcription.
  118 buses, 173 lines + 13 transformers = 186 trippable branches, 99 loads
  (4242 MW), 53 generators + slack, 14 shunts.
  NOTE pandapower indexes these buses 0..117; IEEE numbering is index + 1.

DIFFERENCES FROM THE 27-BUS SETUP, AND WHY
  1. VOLTAGE-SCHEDULE FLOOR (V_SET_FLOOR = 0.96 p.u.).
     case118 ships generator setpoints as low as 0.943 p.u. -- below the 0.95
     marginal threshold -- so the intact network would be labelled "Marginal"
     because of a scheduling choice rather than a security condition.  Flooring
     the schedule at 0.96 means a sub-0.95 reading can only come from a load bus,
     or from a generator that hit its Q limit and lost voltage control.  The
     latter is a genuine voltage-security event and is deliberately preserved.
     The 26-bus case needed no such floor (its setpoints are all >= 1.015).
     The CEILING is 1.04, strictly inside the 0.95-1.05 normal band.  It must be
     strictly inside: at dV = 0 the controlled buses sit at exactly their
     schedule, and the Unstable trigger V_max >= 1.05 is inclusive, so a 1.05
     ceiling would label every base case Unstable.
  1b. ADMISSION TEST ON THE OPERATING POINT.  A sampled operating point is kept
     only if its PRE-CONTINGENCY solution has V_max < 1.05.  Capping the
     schedule is NOT sufficient on its own: under high PV injection the network
     carries surplus reactive power, generators absorb down to their minimum Q
     limit, `enforce_q_lims` switches them PV->PQ, and their terminal voltage
     then floats ABOVE schedule (measured: 1.0638 at a 1.05 schedule).  Without
     the test, roughly half of all base cases breach the normal band before any
     contingency is applied, every contingency at those points inherits the
     breach, and the label becomes a function of the operating point alone --
     exactly the leakage this formulation exists to remove.  The low side is
     deliberately not screened; see the note in generate().
  2. PV AS A GRID-FOLLOWING INVERTER.  PV is a pandapower `sgen`: a constant-P,
     unity-power-factor injection (Q = 0, no voltage support).  NOTE that
     `runpp` treats an sgen as a fixed PQ injection -- the min/max_q_mvar fields
     are OPF metadata and are NOT enforced here, and `enforce_q_lims` applies
     only to `gen` elements.  So this is deliberately the most conservative
     grid-following model: the plant contributes no reactive support at all.
     Set PV_MODEL='gen' to recover the voltage-controlled behaviour used in the
     26-bus study.  This addresses the reviewers' question about inverter
     control mode.
  3. NO ADDED BUS.  The 26-bus study added Bus 27 for PV.  Here the three PV
     plants attach at EXISTING load buses, so the topology remains the published
     IEEE 118-bus system exactly (118 buses, 186 branches).
  4. GENERATOR VOLTAGE SCHEDULE IS PART OF THE OPERATING POINT.  A per-scenario
     offset dV is applied to every setpoint.  Without it V_min is pinned by a
     fixed generator setpoint and barely responds to load, which would collapse
     the class balance.

UNCHANGED FROM THE 27-BUS SETUP
  - Inputs: pre-contingency node state (Vm, Va, P, Q) + pre-contingency branch
    state (loading, P_from) + the in-service mask that encodes the contingency.
  - The post-contingency voltages are NEVER in the input.
  - Risk classes: Stable Vmin >= 0.95, Marginal [0.90, 0.95), Unstable < 0.90,
    Vmax >= 1.05, or power-flow divergence.  BOTH sides of Eq. (risk) are now
    enforced in code; see risk_of() and the ADMISSION TEST note below.
  - Per-bus vulnerability 1[V_i < 0.95]; divergence flag; contingency size k
    drawn from the same distribution.

Output npz has exactly the same keys as the 27-bus file, so every downstream
consumer works unchanged.
"""
import os
import time
from copy import deepcopy

import numpy as np
import pandapower as pp
import pandapower.networks as pn

# ---------------------------------------------------------------- config
SEED = 42
TARGET = 30000
N_OP = 480                      # operating points, amortising base solves
V_SET_FLOOR, V_SET_CEIL = 0.96, 1.04
V_HI = 1.05                     # normal-band upper limit; also the Unstable trigger
PV_MODEL = 'sgen'               # 'sgen' = grid-following (default) | 'gen' = voltage-controlled
NUMBA_PF = False                # power-flow numba backend; False matches the 27-bus generation
PV_BUSES = [59, 77, 10]         # pandapower indices = IEEE buses 60, 78, 11
PV_TOTAL_MAX = 640.0            # MW, ~15% of the 4242 MW system load
LOAD_LO, LOAD_HI = 0.55, 1.55
DV_LO, DV_HI = -0.03, 0.03
K_CHOICES = [0, 1, 1, 1, 1, 2, 2, 2, 3, 4]   # identical to the 27-bus generator


def build_net():
    """case118 with a floored voltage schedule and three grid-following PV plants."""
    net = pn.case118()
    net.gen['vm_pu'] = net.gen['vm_pu'].clip(lower=V_SET_FLOOR, upper=V_SET_CEIL)
    net.ext_grid['vm_pu'] = net.ext_grid['vm_pu'].clip(lower=V_SET_FLOOR, upper=V_SET_CEIL)
    per = PV_TOTAL_MAX / len(PV_BUSES)
    pv_idx = []
    for b in PV_BUSES:
        if PV_MODEL == 'sgen':
            i = pp.create_sgen(net, bus=b, p_mw=0.0, q_mvar=0.0,
                               max_p_mw=per, min_p_mw=0.0,
                               max_q_mvar=0.33 * per, min_q_mvar=-0.33 * per,
                               name=f"PV plant bus {b}", type='PV')
        else:
            i = pp.create_gen(net, bus=b, p_mw=0.0, vm_pu=1.0,
                              max_q_mvar=0.33 * per, min_q_mvar=-0.33 * per,
                              name=f"PV plant bus {b}", type='PV')
        pv_idx.append(i)
    return net, pv_idx


def risk_of(vmin, vmax, conv):
    """
    Eq. (risk) of the manuscript, BOTH sides enforced.

    Unstable   V_min < 0.90, V_max >= 1.05, nonconvergence, or islanding
    Marginal   0.90 <= V_min < 0.95 and V_max < 1.05
    Stable     V_min >= 0.95        and V_max < 1.05

    The overvoltage branch used to be missing here (and is still missing in the
    released 27-bus generator), so the code implemented a one-sided rule under a
    two-sided equation.  It is enforced now.  Reaching it meaningfully requires
    an admissible base case -- see the ADMISSION TEST in generate() -- otherwise
    the label is decided by the operating point rather than by the contingency.
    """
    if not conv:
        return 2
    if vmin < 0.90 or vmax >= V_HI:
        return 2
    if vmin < 0.95:
        return 1
    return 0


def generate(target=TARGET, out_path='contingency_data_118_v2.npz', seed=SEED,
             n_op=N_OP, progress_every=2000, checkpoint_every=None, verbose=True):
    rng = np.random.RandomState(seed)
    net0, pv_idx = build_net()
    pv_tbl_name = 'sgen' if PV_MODEL == 'sgen' else 'gen'

    base_p = net0.load['p_mw'].copy()
    base_q = net0.load['q_mvar'].copy()
    base_vg = net0.gen['vm_pu'].copy()
    base_vs = net0.ext_grid['vm_pu'].copy()

    bus_ids = sorted(net0.bus.index.tolist())
    n_bus = len(bus_ids)
    b2i = {b: i for i, b in enumerate(bus_ids)}

    branches = ([('line', i) for i in net0.line.index]
                + [('trafo', i) for i in net0.trafo.index])
    n_branch = len(branches)
    edge_pairs = []
    for typ, bi in branches:
        if typ == 'line':
            f, t = int(net0.line.at[bi, 'from_bus']), int(net0.line.at[bi, 'to_bus'])
        else:
            f, t = int(net0.trafo.at[bi, 'hv_bus']), int(net0.trafo.at[bi, 'lv_bus'])
        edge_pairs.append((b2i[f], b2i[t]))

    if verbose:
        print(f"grid: {n_bus} buses, {n_branch} branches "
              f"({len(net0.line)} lines + {len(net0.trafo)} trafos), "
              f"PV model={PV_MODEL} at buses {PV_BUSES}")

    def solve(load_s, dv, pv_mw, trip=None):
        sim = deepcopy(net0)
        sim.load['p_mw'] = base_p * load_s
        sim.load['q_mvar'] = base_q * load_s
        sim.gen['vm_pu'] = np.clip(base_vg + dv, V_SET_FLOOR, V_SET_CEIL)
        sim.ext_grid['vm_pu'] = np.clip(base_vs + dv, V_SET_FLOOR, V_SET_CEIL)
        tbl = getattr(sim, pv_tbl_name)
        for i in pv_idx:
            tbl.at[i, 'p_mw'] = pv_mw / len(pv_idx)
        if trip:
            for typ, bi in trip:
                if typ == 'line':
                    sim.line.at[bi, 'in_service'] = False
                else:
                    sim.trafo.at[bi, 'in_service'] = False
        try:
            pp.runpp(sim, enforce_q_lims=True, numba=NUMBA_PF, max_iteration=30)
            if sim.res_bus['vm_pu'].isna().any():
                return sim, False
            return sim, True
        except Exception:
            return sim, False

    def node_feats(sim):
        f = np.zeros((n_bus, 4), np.float32)
        vm = sim.res_bus['vm_pu']
        va = sim.res_bus['va_degree']
        pmw = sim.res_bus['p_mw']
        qmv = sim.res_bus['q_mvar']
        for b in bus_ids:
            i = b2i[b]
            f[i, 0] = vm.at[b]
            f[i, 1] = va.at[b] / 180.0
            f[i, 2] = pmw.at[b] / 100.0
            f[i, 3] = qmv.at[b] / 100.0
        return np.nan_to_num(f)

    def base_edge_feats(sim):
        f = np.zeros((n_branch, 2), np.float32)
        for k, (typ, bi) in enumerate(branches):
            if typ == 'line' and bi in sim.res_line.index:
                f[k, 0] = (sim.res_line.at[bi, 'loading_percent'] or 0) / 100.0
                f[k, 1] = (sim.res_line.at[bi, 'p_from_mw'] or 0) / 100.0
            elif typ == 'trafo' and bi in sim.res_trafo.index:
                f[k, 0] = (sim.res_trafo.at[bi, 'loading_percent'] or 0) / 100.0
                f[k, 1] = (sim.res_trafo.at[bi, 'p_hv_mw'] or 0) / 100.0
        return np.nan_to_num(f)

    def sample_contingency():
        k = rng.choice(K_CHOICES, 1)[0]
        if k == 0:
            return []
        idx = rng.choice(n_branch, size=k, replace=False)
        return [branches[j] for j in idx]

    # ------------------------------------------------------- operating points
    # ADMISSION TEST.  An operating point is admitted only if its
    # PRE-CONTINGENCY state solves and lies inside the normal band
    # (V_max < 1.05).  A post-contingency security screen presupposes an
    # admissible starting state: if the base case already breaches the limit,
    # every contingency at that point inherits the breach and the label becomes
    # a function of the operating point alone rather than of the contingency --
    # precisely the leakage this formulation exists to remove.
    #
    # The low side is deliberately NOT screened: a depressed base case still
    # leaves Marginal-vs-Unstable to be decided by the contingency, and the
    # published 27-bus corpus is built the same way.
    OP_SET, base_cache = [], {}
    n_tried = n_rej_pf = n_rej_hi = 0
    while len(OP_SET) < n_op:
        n_tried += 1
        op = (round(float(rng.uniform(LOAD_LO, LOAD_HI)), 3),
              round(float(rng.uniform(DV_LO, DV_HI)), 4),
              round(float(rng.uniform(0.0, PV_TOTAL_MAX)), 1))
        if op in base_cache:
            continue
        sb, cb = solve(op[0], op[1], op[2], None)
        if not cb:
            n_rej_pf += 1
            continue
        if float(np.nanmax(sb.res_bus['vm_pu'].values)) >= V_HI:
            n_rej_hi += 1
            continue
        base_cache[op] = (node_feats(sb), base_edge_feats(sb))
        OP_SET.append(op)
    if verbose:
        print(f"operating points: admitted {len(OP_SET)} of {n_tried} sampled "
              f"(rejected {n_rej_hi} for base V_max >= {V_HI}, "
              f"{n_rej_pf} for base non-convergence)")

    NF, EF, INSVC, OPC = [], [], [], []
    T_vmin, T_risk, T_vuln, T_div, T_vbus = [], [], [], [], []
    t0 = time.time()
    n = 0
    nb = len(base_cache)
    while n < target:
        op = OP_SET[rng.randint(len(OP_SET))]
        load_s, dv, pv_mw = op
        nf, ef = base_cache[op]

        trip = sample_contingency()
        sim, conv = solve(load_s, dv, pv_mw, trip)
        if conv:
            vbus = np.array([sim.res_bus.at[b, 'vm_pu'] for b in bus_ids], np.float32)
            vmin = float(np.nanmin(vbus))
            vmax = float(np.nanmax(vbus))
            vuln = (vbus < 0.95).astype(np.float32)
        else:
            vbus = np.zeros(n_bus, np.float32)
            vmin = 0.0
            vmax = 0.0
            vuln = np.ones(n_bus, np.float32)

        insvc = np.ones(n_branch, np.float32)
        for typ, bi in trip:
            insvc[branches.index((typ, bi))] = 0.0

        NF.append(nf); EF.append(ef); INSVC.append(insvc)
        OPC.append([load_s, pv_mw, dv])
        T_vmin.append(vmin); T_risk.append(risk_of(vmin, vmax, conv))
        T_vuln.append(vuln); T_div.append(0.0 if conv else 1.0); T_vbus.append(vbus)
        n += 1

        if verbose and progress_every and n % progress_every == 0:
            r = np.array(T_risk)
            print(f"  {n}/{target}  base_pts={nb}  "
                  f"S/M/U={int((r==0).sum())}/{int((r==1).sum())}/{int((r==2).sum())}  "
                  f"[{time.time()-t0:.0f}s]")
        if checkpoint_every and n % checkpoint_every == 0:
            _save(out_path.replace('.npz', '_partial.npz'), NF, EF, INSVC, OPC, T_vmin, T_risk, T_vuln,
                  T_div, T_vbus, edge_pairs, bus_ids, n_bus, n_branch)

    path = _save(out_path, NF, EF, INSVC, OPC, T_vmin, T_risk, T_vuln, T_div,
                 T_vbus, edge_pairs, bus_ids, n_bus, n_branch,
                 n_op_sampled=n_tried, n_op_rejected_high=n_rej_hi,
                 n_op_rejected_pf=n_rej_pf, v_set_ceil=V_SET_CEIL, v_hi=V_HI)
    if verbose:
        r = np.array(T_risk)
        vu = np.array(T_vuln)
        vb = np.array(T_vbus); dv_ = np.array(T_div)
        cv = dv_ < 0.5
        over = int(((vb.max(axis=1) >= V_HI) & cv).sum())
        print(f"\nDONE {n} samples in {time.time()-t0:.0f}s")
        print(f"  S/M/U = {int((r==0).sum())}/{int((r==1).sum())}/{int((r==2).sum())}"
              f"  diverged={int(dv_.sum())}")
        print(f"  contingency-induced overvoltage (V_max >= {V_HI}): {over}")
        print(f"  per-bus vulnerability positive rate: {vu.mean()*100:.1f}%")
        print(f"  saved {path} ({os.path.getsize(path)/1e6:.1f} MB)")
    return path


def _save(path, NF, EF, INSVC, OPC, T_vmin, T_risk, T_vuln, T_div, T_vbus,
          edge_pairs, bus_ids, n_bus, n_branch, **meta):
    NF = np.array(NF, np.float32); EF = np.array(EF, np.float32)
    INSVC = np.array(INSVC, np.float32); OPC = np.array(OPC, np.float32)
    T_vmin = np.array(T_vmin, np.float32); T_risk = np.array(T_risk, np.int64)
    T_vuln = np.array(T_vuln, np.float32); T_div = np.array(T_div, np.float32)
    T_vbus = np.array(T_vbus, np.float32)
    nmean = NF.reshape(-1, 4).mean(0); nstd = NF.reshape(-1, 4).std(0); nstd[nstd < 1e-6] = 1
    emean = EF.reshape(-1, 2).mean(0); estd = EF.reshape(-1, 2).std(0); estd[estd < 1e-6] = 1
    np.savez_compressed(
        path,
        node_feats=NF, edge_feats=EF, in_service=INSVC, opcond=OPC,
        t_vmin=T_vmin, t_risk=T_risk, t_vuln=T_vuln, t_div=T_div, t_vbus=T_vbus,
        edge_pairs=np.array(edge_pairs, np.int32), bus_ids=np.array(bus_ids, np.int32),
        node_mean=nmean, node_std=nstd, edge_mean=emean, edge_std=estd,
        n_bus=np.int32(n_bus), n_branch=np.int32(n_branch),
        **{k: np.float32(v) for k, v in meta.items()})
    return path


def make_grouped_split(npz_path, out_path='grouped_split_118_v2.npz', seed=42,
                       frac=(0.70, 0.15, 0.15), verbose=True):
    """
    Operating-point-disjoint split: an operating point contributes to exactly one
    of train / val / test, so the test set measures generalisation to UNSEEN
    operating conditions -- the same protocol as the 27-bus study.
    Normalisation statistics are computed on the TRAIN split only.
    """
    D = np.load(npz_path)
    op = D['opcond']
    keys = [tuple(np.round(r, 4)) for r in op]
    uniq = sorted(set(keys))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uniq))
    n_tr = int(frac[0] * len(uniq))
    n_va = int(frac[1] * len(uniq))
    grp = {}
    for j, p in enumerate(perm):
        grp[uniq[p]] = 0 if j < n_tr else (1 if j < n_tr + n_va else 2)
    tag = np.array([grp[k] for k in keys])
    idx_tr = np.where(tag == 0)[0]
    idx_va = np.where(tag == 1)[0]
    idx_te = np.where(tag == 2)[0]

    NF = D['node_feats'][idx_tr].reshape(-1, D['node_feats'].shape[-1])
    EF = D['edge_feats'][idx_tr].reshape(-1, D['edge_feats'].shape[-1])
    nmean, nstd = NF.mean(0), NF.std(0); nstd[nstd < 1e-6] = 1
    emean, estd = EF.mean(0), EF.std(0); estd[estd < 1e-6] = 1
    np.savez_compressed(out_path, idx_tr=idx_tr, idx_va=idx_va, idx_te=idx_te,
                        nmean=nmean, nstd=nstd, emean=emean, estd=estd,
                        n_op=np.int32(len(uniq)))
    if verbose:
        r = D['t_risk']
        print(f"grouped split: {len(uniq)} operating points -> "
              f"train {len(idx_tr)} / val {len(idx_va)} / test {len(idx_te)}")
        for nm, ix in (('train', idx_tr), ('val', idx_va), ('test', idx_te)):
            c = np.bincount(r[ix], minlength=3)
            print(f"   {nm:5s} S/M/U = {c[0]}/{c[1]}/{c[2]}")
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=int, default=TARGET)
    ap.add_argument('--n-op', type=int, default=N_OP)
    # v2 filenames: the v1 corpus is EVIDENCE for the response letter
    # (232 of 480 base cases above 1.05, 13,190 of 13,192 overvoltages inherited
    # rather than contingency-induced).  Never overwrite it.
    ap.add_argument('--out', type=str, default='contingency_data_118_v2.npz')
    ap.add_argument('--split-out', type=str, default='grouped_split_118_v2.npz')
    a = ap.parse_args()
    p = generate(target=a.target, out_path=a.out, n_op=a.n_op,
                 progress_every=max(1, a.target // 10))
    make_grouped_split(p, a.split_out)
