"""
Training + evaluation for the IEEE 118-bus contingency-screening surrogate.

Recipe is the 27-bus one (150-epoch warmup-cosine, label smoothing, class
weights, EMA, risk-gated checkpoint selection on zero false-safe); only the
batch assembly differs, because the dense edge/adjacency tensors are built per
batch instead of for the whole dataset (see core118.py).

Metrics deliberately go beyond the 27-bus set, to pre-empt reviewer questions:
  * PR-AUC alongside ROC-AUC for the per-bus vulnerability head.  On 118 buses
    most buses are non-vulnerable under most contingencies, so ROC-AUC alone
    flatters the model.  This is reviewer R4 #3.
  * A screening benchmark that reports the speed-up BOTH with and without
    feature preprocessing, and that SCORES the sweep, which is reviewer R1 #6.
  * An explicit breakdown of the Unstable class into voltage violation,
    islanding, and non-islanded divergence, which is reviewer R1 #4.
"""
import json
import math
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from core118 import CSGNN118, build_graph_ctx, cells_and_adj


# ============================================================================
# Data
# ============================================================================
def build_data(npz_path, split_path, dev='cpu'):
    """Compact tensors only; dense per-batch structures are built in `batch()`."""
    D = np.load(npz_path)
    S = np.load(split_path)
    n_bus = int(D['n_bus'])
    ctx = build_graph_ctx(D['edge_pairs'], n_bus)

    nmean = torch.tensor(S['nmean']).float()
    nstd = torch.tensor(S['nstd']).float()
    emean = torch.tensor(S['emean']).float()
    estd = torch.tensor(S['estd']).float()

    NF = ((torch.tensor(D['node_feats']) - nmean) / nstd).float().to(dev)
    EF = ((torch.tensor(D['edge_feats']) - emean) / estd).float().to(dev)
    INS = torch.tensor(D['in_service']).float().to(dev)

    d = dict(
        NF=NF, EF=EF, INS=INS, ctx=ctx, n_bus=n_bus, dev=dev,
        Tr=torch.tensor(D['t_risk']).to(dev),
        Tv=torch.tensor(D['t_vmin']).float().to(dev),
        Tu=torch.tensor(D['t_vuln']).float().to(dev),
        Td=torch.tensor(D['t_div']).float().to(dev),
        Tb=torch.tensor(D['t_vbus']).float().to(dev) if 't_vbus' in D else None,
        # V_max target for the high-side head.  Diverged scenarios store an
        # all-zero voltage vector, so this is 0.0 there exactly as t_vmin is,
        # and the loss masks them out the same way.
        Tx=(torch.tensor(D['t_vbus']).float().max(dim=1).values.to(dev)
            if 't_vbus' in D else None),
        idx_tr=torch.tensor(S['idx_tr'].astype('int64')).to(dev),
        idx_va=torch.tensor(S['idx_va'].astype('int64')).to(dev),
        idx_te=torch.tensor(S['idx_te'].astype('int64')).to(dev),
        opcond=D['opcond'],
        # Kept so that inference-time feature extraction uses the SAME statistics
        # the model was trained with.  Feeding raw features to a model trained on
        # normalised ones is silent and produces meaningless predictions.
        nmean=nmean.to(dev), nstd=nstd.to(dev),
        emean=emean.to(dev), estd=estd.to(dev),
    )
    cnt = torch.bincount(d['Tr'][d['idx_tr']], minlength=3).float()
    d['cw'] = (cnt.sum() / (3 * cnt.clamp(min=1))).to(dev)
    return d


def batch(d, b):
    """Assemble one batch: node features + per-cell edge features + adjacency."""
    cell_feat, adj = cells_and_adj(d['EF'][b], d['INS'][b], d['ctx'])
    return d['NF'][b], cell_feat, adj


# ============================================================================
# Evaluation
# ============================================================================
@torch.no_grad()
def evaluate(net, d, idx, amp=False, bs=1024, full=False):
    net.eval()
    P, LOG, VMp, VMt, Dp, Dt, VUp, VUt = [], [], [], [], [], [], [], []
    for s0 in range(0, len(idx), bs):
        b = idx[s0:s0 + bs]
        nf, cf, adj = batch(d, b)
        with torch.autocast('cuda', enabled=amp):
            _o = net(nf, cf, adj)
            r, vm, vu, dv = _o[0], _o[1], _o[2], _o[3]
        r = r.float()
        P.append(r.argmax(1))
        LOG.append(r)
        conv = d['Td'][b] == 0
        VMp.append(vm.float()[conv]); VMt.append(d['Tv'][b][conv])
        Dp.append(torch.sigmoid(dv.float())); Dt.append(d['Td'][b])
        VUp.append(torch.sigmoid(vu.float())[conv]); VUt.append(d['Tu'][b][conv])

    P = torch.cat(P); R = d['Tr'][idx]; LOG = torch.cat(LOG)
    acc = (P == R).float().mean().item()
    fs_n = int((((R == 2) & (P == 0))).sum())
    fs = fs_n / max(1, int((R == 2).sum()))
    vmp, vmt = torch.cat(VMp), torch.cat(VMt)
    vmae = (vmp - vmt).abs().mean().item() if len(vmp) else float('nan')
    ss_res = ((vmp - vmt) ** 2).sum().item()
    ss_tot = ((vmt - vmt.mean()) ** 2).sum().item()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    dp, dt = torch.cat(Dp), torch.cat(Dt)
    dpred = (dp > 0.5).float()
    tp = float(((dpred == 1) & (dt == 1)).sum()); fp = float(((dpred == 1) & (dt == 0)).sum())
    fn = float(((dpred == 0) & (dt == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else float('nan')
    rec = tp / (tp + fn) if tp + fn else float('nan')
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and prec + rec else float('nan')

    out = dict(acc=acc, false_safe=fs, false_safe_n=fs_n,
               n_unstable=int((R == 2).sum()),
               vmin_mae=vmae, vmin_r2=r2, div_f1=f1,
               div_precision=prec, div_recall=rec, n=len(idx))
    if not full:
        return out

    # ---- per-class recall + confusion
    conf = np.zeros((3, 3), int)
    for t in range(3):
        for p in range(3):
            conf[t, p] = int(((R == t) & (P == p)).sum())
    out['confusion'] = conf.tolist()
    out['per_class_recall'] = [float(conf[t, t] / max(1, conf[t].sum())) for t in range(3)]

    # ---- calibration on the risk head
    prob = torch.softmax(LOG, 1)
    conf_max, pred = prob.max(1)
    correct = (pred == R).float()
    bins = torch.linspace(0, 1, 16, device=prob.device)
    ece = 0.0
    for i in range(15):
        m = (conf_max > bins[i]) & (conf_max <= bins[i + 1])
        if m.any():
            ece += (m.float().mean() * (correct[m].mean() - conf_max[m].mean()).abs()).item()
    out['ece'] = ece
    onehot = F.one_hot(R, 3).float()
    out['brier'] = ((prob - onehot) ** 2).sum(1).mean().item()
    out['nll'] = F.nll_loss(torch.log(prob.clamp(min=1e-12)), R).item()

    # ---- vulnerability head: ROC-AUC *and* PR-AUC (reviewer R4 #3)
    # NOTE this positive rate is over CONVERGED scenarios only, because the
    # vulnerability target is undefined when the power flow diverged.  The
    # dataset-summary figure is over all scenarios, so the two differ.
    vup = torch.cat(VUp).reshape(-1).cpu().numpy()
    vut = torch.cat(VUt).reshape(-1).cpu().numpy()
    out['vuln_positive_rate_converged'] = float(vut.mean())
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        if vut.min() != vut.max():
            out['vuln_roc_auc'] = float(roc_auc_score(vut, vup))
            out['vuln_pr_auc'] = float(average_precision_score(vut, vup))
    except Exception:
        out['vuln_roc_auc'] = out['vuln_pr_auc'] = float('nan')
    try:
        from sklearn.metrics import roc_auc_score
        yn = dt.cpu().numpy(); pn_ = dp.cpu().numpy()
        if yn.min() != yn.max():
            out['div_roc_auc'] = float(roc_auc_score(yn, pn_))
    except Exception:
        pass
    return out


# ============================================================================
# Training
# ============================================================================
def train_eval(d, seed=0, alpha=0.0, branched=False, stack=False, vbus=False, vmax=False,
               hidden=128, heads=4, n_gat=3, epochs=150, bs=256, lr=3e-3, ls=0.05,
               amp=True, verbose=True, eval_bs=1024, log_every=10):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = d['dev']
    amp = amp and str(dev).startswith('cuda')
    vbus = vbus and d.get('Tb') is not None
    vmax = vmax and d.get('Tx') is not None

    m = CSGNN118(hidden=hidden, heads=heads, n_gat=n_gat, branched=branched,
                 stack=stack, vbus=vbus, vmax=vmax).to(dev)
    m.set_graph(d['ctx'])
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    warm = max(1, epochs // 30)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: min((e + 1) / warm, 1.0)
        * 0.5 * (1 + math.cos(math.pi * max(0, e - warm) / max(1, epochs - warm))))
    ema = torch.optim.swa_utils.AveragedModel(
        m, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    idx_tr = d['idx_tr']
    best, best_state = -1.0, None
    stack_warm = max(1, epochs // 15)
    n_bad = 0
    t0 = time.time()

    for ep in range(epochs):
        m.train()
        beta = min(1.0, (ep + 1) / stack_warm) if stack else 1.0
        perm = idx_tr[torch.randperm(len(idx_tr), device=dev)]
        for s0 in range(0, len(perm), bs):
            b = perm[s0:s0 + bs]
            nf, cf, adj = batch(d, b)
            with torch.autocast('cuda', enabled=amp):
                out = m(nf, cf, adj, alpha=alpha, beta=beta)
                r, vm, vu, dv, vb = out[:5]
                vx = out[5] if len(out) > 5 else None
                conv = d['Td'][b] == 0
                l_risk = F.cross_entropy(r, d['Tr'][b], weight=d['cw'], label_smoothing=ls)
                l_vm = F.smooth_l1_loss(vm[conv], d['Tv'][b][conv]) if conv.any() else r.sum() * 0
                l_vu = (F.binary_cross_entropy_with_logits(vu[conv], d['Tu'][b][conv])
                        if conv.any() else r.sum() * 0)
                l_dv = F.binary_cross_entropy_with_logits(dv, d['Td'][b])
                l_vb = (F.smooth_l1_loss(vb[conv], d['Tb'][b][conv])
                        if (vbus and conv.any()) else r.sum() * 0)
                # same weight as the V_min head: the criterion is two-sided,
                # so the supervision is symmetric
                l_vx = (F.smooth_l1_loss(vx[conv], d['Tx'][b][conv])
                        if (vx is not None and conv.any()) else r.sum() * 0)
                loss = (l_risk + 0.5 * l_vm + 0.5 * l_vx
                        + 0.3 * l_vu + 0.3 * l_dv + 0.3 * l_vb)
            if not torch.isfinite(loss):
                n_bad += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0)
            scaler.step(opt); scaler.update()
            ema.update_parameters(m)
        sch.step()

        if ep >= epochs // 3 and (ep % 5 == 0 or ep == epochs - 1):
            for cand in (m, ema.module):
                r = evaluate(cand, d, d['idx_va'], amp=amp, bs=eval_bs)
                if r['false_safe'] == 0.0 and r['acc'] > best:
                    best = r['acc']
                    best_state = {k: v.clone() for k, v in cand.state_dict().items()}
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            r = evaluate(m, d, d['idx_va'], amp=amp, bs=eval_bs)
            print(f"  ep{ep:3d}  val acc {r['acc']*100:5.2f}%  fs {r['false_safe']*100:5.2f}%  "
                  f"best@fs0 {max(best,0)*100:5.2f}%  [{time.time()-t0:.0f}s]")

    if best_state is None:
        best_state = max(((evaluate(c, d, d['idx_va'], amp=amp, bs=eval_bs)['acc'], i, c)
                          for i, c in enumerate((m, ema.module))), key=lambda t: t[0])[2].state_dict()
        best_state = {k: v.clone() for k, v in best_state.items()}
    ema.module.load_state_dict(best_state)
    te = evaluate(ema.module, d, d['idx_te'], amp=amp, bs=eval_bs, full=True)
    te['params'] = sum(p.numel() for p in m.parameters())
    te['epochs'] = epochs
    te['batch_size'] = bs
    te['train_seconds'] = time.time() - t0
    te['nonfinite_steps_skipped'] = n_bad
    return te, ema.module


# ============================================================================
# Screening benchmark + sweep scoring
#   reviewer R1 #6: is preprocessing inside the reported speed-up?
# ============================================================================
@torch.no_grad()
def screening_benchmark(net, d, gen_module, n_repeat=3, verbose=True,
                        load_s=1.0, dv=0.0, pv_mw=300.0):
    """
    Time AND score a full N-1 sweep.

    ORACLE    : one AC power flow per contingency.  Also supplies ground truth.
    SURROGATE : (a) forward pass only, (b) end to end, including the base power
                flow, feature extraction and -- critically -- the SAME
                normalisation the model was trained with.

    Reporting both timings is the honest answer to "is preprocessing inside the
    speed-up?".  Scoring the sweep is what backs the claim that the screen flags
    every unstable contingency it screens: `sweep_false_safe` counts the truly
    unstable contingencies the surrogate cleared as Stable.
    """
    import pandapower as pp
    net0, pv_idx = gen_module.build_net()
    branches = ([('line', i) for i in net0.line.index]
                + [('trafo', i) for i in net0.trafo.index])
    n_branch = len(branches)
    bus_ids = sorted(net0.bus.index.tolist())
    base_p = net0.load['p_mw'].copy()
    base_q = net0.load['q_mvar'].copy()
    base_vg = net0.gen['vm_pu'].copy()
    base_vs = net0.ext_grid['vm_pu'].copy()
    pv_tbl = 'sgen' if gen_module.PV_MODEL == 'sgen' else 'gen'
    floor, ceil = gen_module.V_SET_FLOOR, gen_module.V_SET_CEIL
    nb_flag = getattr(gen_module, 'NUMBA_PF', False)

    def prep(sim):
        sim.load['p_mw'] = base_p * load_s
        sim.load['q_mvar'] = base_q * load_s
        sim.gen['vm_pu'] = np.clip(base_vg + dv, floor, ceil)
        sim.ext_grid['vm_pu'] = np.clip(base_vs + dv, floor, ceil)
        tbl = getattr(sim, pv_tbl)
        for i in pv_idx:
            tbl.at[i, 'p_mw'] = pv_mw / len(pv_idx)
        return sim

    def trip(sim, k):
        typ, bi = branches[k]
        if typ == 'line':
            sim.line.at[bi, 'in_service'] = False
        else:
            sim.trafo.at[bi, 'in_service'] = False
        return sim

    # ---- oracle: one solve per contingency (timed, and used as ground truth)
    t_oracle, truth = [], None
    for _ in range(n_repeat):
        lab = np.zeros(n_branch, np.int64)
        t0 = time.time()
        for k in range(n_branch):
            sim = trip(prep(deepcopy(net0)), k)
            try:
                pp.runpp(sim, enforce_q_lims=True, numba=nb_flag, max_iteration=30)
                conv = not sim.res_bus['vm_pu'].isna().any()
                vmin = float(sim.res_bus['vm_pu'].min()) if conv else 0.0
                vmax = float(sim.res_bus['vm_pu'].max()) if conv else 0.0
            except Exception:
                conv, vmin, vmax = False, 0.0, 0.0
            lab[k] = gen_module.risk_of(vmin, vmax, conv)
        t_oracle.append(time.time() - t0)
        if truth is None:
            truth = lab
    t_or = float(np.median(t_oracle))

    # ---- surrogate: base solve + features + normalisation + batched forward
    dev = d['dev']
    t_e2e, t_fwd, pred = [], [], None
    for _ in range(n_repeat):
        t0 = time.time()
        sim = prep(deepcopy(net0))
        pp.runpp(sim, enforce_q_lims=True, numba=nb_flag, max_iteration=30)
        nf_np = np.zeros((len(bus_ids), 4), np.float32)
        for ii, bb in enumerate(bus_ids):
            nf_np[ii, 0] = sim.res_bus.at[bb, 'vm_pu']
            nf_np[ii, 1] = sim.res_bus.at[bb, 'va_degree'] / 180.0
            nf_np[ii, 2] = sim.res_bus.at[bb, 'p_mw'] / 100.0
            nf_np[ii, 3] = sim.res_bus.at[bb, 'q_mvar'] / 100.0
        ef_np = np.zeros((n_branch, 2), np.float32)
        for k, (typ, bi) in enumerate(branches):
            if typ == 'line' and bi in sim.res_line.index:
                ef_np[k, 0] = (sim.res_line.at[bi, 'loading_percent'] or 0) / 100.0
                ef_np[k, 1] = (sim.res_line.at[bi, 'p_from_mw'] or 0) / 100.0
            elif typ == 'trafo' and bi in sim.res_trafo.index:
                ef_np[k, 0] = (sim.res_trafo.at[bi, 'loading_percent'] or 0) / 100.0
                ef_np[k, 1] = (sim.res_trafo.at[bi, 'p_hv_mw'] or 0) / 100.0
        # SAME normalisation as training. This is preprocessing, so it is inside
        # the end-to-end clock.
        NF1 = (torch.tensor(np.nan_to_num(nf_np), device=dev) - d['nmean']) / d['nstd']
        EF1 = (torch.tensor(np.nan_to_num(ef_np), device=dev) - d['emean']) / d['estd']
        NF = NF1.unsqueeze(0).expand(n_branch, -1, -1).contiguous()
        EF = EF1.unsqueeze(0).expand(n_branch, -1, -1).contiguous()
        INS = torch.ones(n_branch, n_branch, device=dev)
        INS[torch.arange(n_branch), torch.arange(n_branch)] = 0.0
        t_prep = time.time()
        cf, adj = cells_and_adj(EF, INS, d['ctx'])
        net.eval()
        r = net(NF, cf, adj)[0]
        if str(dev).startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()
        t_e2e.append(t1 - t0)
        t_fwd.append(t1 - t_prep)
        if pred is None:
            pred = r.float().argmax(1).cpu().numpy()

    n_unstable = int((truth == 2).sum())
    false_safe = int(((truth == 2) & (pred == 0)).sum())
    out = dict(
        n_contingencies=n_branch,
        oracle_s=t_or,
        surrogate_end_to_end_s=float(np.median(t_e2e)),
        surrogate_forward_only_s=float(np.median(t_fwd)),
        sweep_accuracy=float((pred == truth).mean()),
        sweep_n_unstable_true=n_unstable,
        sweep_n_flagged_not_stable=int((pred != 0).sum()),
        sweep_false_safe=false_safe,
        sweep_operating_point=dict(load_scale=load_s, schedule_shift=dv, pv_mw=pv_mw),
    )
    out['speedup_end_to_end'] = out['oracle_s'] / out['surrogate_end_to_end_s']
    out['speedup_forward_only'] = out['oracle_s'] / out['surrogate_forward_only_s']
    if verbose:
        print(f"  N-1 sweep of {n_branch} contingencies (load {load_s}x, PV {pv_mw:.0f} MW)")
        print(f"    oracle (pandapower)     : {out['oracle_s']*1000:8.1f} ms")
        print(f"    surrogate, end to end   : {out['surrogate_end_to_end_s']*1000:8.1f} ms"
              f"   -> {out['speedup_end_to_end']:6.1f}x")
        print(f"    surrogate, forward only : {out['surrogate_forward_only_s']*1000:8.1f} ms"
              f"   -> {out['speedup_forward_only']:6.1f}x")
        print(f"    sweep accuracy          : {out['sweep_accuracy']*100:.1f}%  "
              f"({n_unstable} truly unstable, {false_safe} cleared as Stable)")
    return out


# ============================================================================
# Unstable-class decomposition (reviewer R1 #4)
# ============================================================================
def unstable_breakdown(npz_path, gen_module, verbose=True):
    """
    Split the Unstable class into (a) converged with V_min < 0.90, (b) divergence
    where the trip set ISLANDS part of the network, and (c) divergence with the
    network still fully connected.  Distinguishing (b) from (c) is exactly what
    R1 #4 asked for: physical instability versus numerical solver failure.

    The islanding test applies each scenario's ACTUAL trip set.  With k up to 4,
    branches that island nothing on their own can island jointly, so testing only
    the single-branch islanders would misfile those as "still connected".  Trip
    sets are cached, so the divergent scenarios cost far fewer topology checks
    than there are scenarios.
    """
    import pandapower.topology as top
    D = np.load(npz_path)
    risk, div, ins = D['t_risk'], D['t_div'], D['in_service']
    net0, _ = gen_module.build_net()
    branches = ([('line', i) for i in net0.line.index]
                + [('trafo', i) for i in net0.trafo.index])

    def islands(trip_idx):
        sim = deepcopy(net0)
        for k in trip_idx:
            typ, bi = branches[k]
            if typ == 'line':
                sim.line.at[bi, 'in_service'] = False
            else:
                sim.trafo.at[bi, 'in_service'] = False
        try:
            return len(top.unsupplied_buses(sim)) > 0
        except Exception:
            return False

    unst = risk == 2
    diverged = unst & (div == 1)
    cache, n_isl = {}, 0
    for i in np.where(diverged)[0]:
        key = tuple(np.where(ins[i] == 0)[0].tolist())
        if key not in cache:
            cache[key] = islands(key)
        n_isl += bool(cache[key])

    solo = sum(1 for k in range(len(branches)) if islands((k,)))
    # Split the converged violations by SIDE.  Eq. (risk) is two-sided, so the
    # overvoltage branch has to be reported separately -- it is the direct answer
    # to R1 #5, which asked whether the overvoltage case is ever exercised.
    vbus = D['t_vbus']
    v_hi = float(D['v_hi']) if 'v_hi' in D else 1.05
    conv_unst = unst & (div == 0)
    vmin_s = vbus.min(axis=1)
    vmax_s = vbus.max(axis=1)
    n_under = int((conv_unst & (vmin_s < 0.90)).sum())
    n_over = int((conv_unst & (vmax_s >= v_hi)).sum())
    n_both = int((conv_unst & (vmin_s < 0.90) & (vmax_s >= v_hi)).sum())

    out = dict(
        n_unstable=int(unst.sum()),
        voltage_violation=int(conv_unst.sum()),
        undervoltage=n_under,
        overvoltage=n_over,
        both_sides=n_both,
        overvoltage_threshold=v_hi,
        divergence_islanding=int(n_isl),
        divergence_connected=int(diverged.sum()) - int(n_isl),
        n_islanding_branches_solo=int(solo),
        n_branches=len(branches),
        distinct_trip_sets_evaluated=len(cache),
    )
    if verbose:
        print(f"  Unstable = {out['n_unstable']}: "
              f"{out['voltage_violation']} voltage violation "
              f"({n_under} undervoltage, {n_over} overvoltage, {n_both} both), "
              f"{out['divergence_islanding']} islanding divergence, "
              f"{out['divergence_connected']} connected divergence")
    return out
