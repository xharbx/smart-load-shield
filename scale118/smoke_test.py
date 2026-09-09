"""
Local smoke test: run the ENTIRE 118-bus pipeline end to end on a small dataset.

This is not a convergence test -- 600 samples and 4 epochs prove nothing about
accuracy.  It proves the things that would otherwise waste a Colab session:
shapes line up, the grouped split populates all three classes, the training loop
steps without NaN, every metric comes back finite, and the two reviewer-driven
analyses (screening speed-up, Unstable-class decomposition) actually run.
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gen118
import train118
from train118 import build_data, train_eval, evaluate, screening_benchmark, unstable_breakdown

SCRATCH = os.environ.get('SMOKE_DIR', os.path.join(HERE, '_smoke'))
os.makedirs(SCRATCH, exist_ok=True)
NPZ = os.path.join(SCRATCH, 'smoke_118.npz')
SPLIT = os.path.join(SCRATCH, 'smoke_split_118.npz')

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


print("=" * 72)
print("SMOKE TEST: full 118-bus pipeline on a small sample")
print("=" * 72)

# ---------------------------------------------------------------- 1. generate
print("\n1. dataset generation")
t0 = time.time()
gen118.generate(target=600, out_path=NPZ, n_op=60, progress_every=200)
gen_s = time.time() - t0
D = np.load(NPZ)
check("npz written", os.path.exists(NPZ))
check("n_bus == 118", int(D['n_bus']) == 118, f"got {int(D['n_bus'])}")
check("n_branch == 186", int(D['n_branch']) == 186, f"got {int(D['n_branch'])}")
check("node_feats shape", D['node_feats'].shape == (600, 118, 4), str(D['node_feats'].shape))
check("edge_feats shape", D['edge_feats'].shape == (600, 186, 2), str(D['edge_feats'].shape))
check("t_vuln shape", D['t_vuln'].shape == (600, 118), str(D['t_vuln'].shape))
check("opcond carries (load, pv, dv)", D['opcond'].shape == (600, 3), str(D['opcond'].shape))
check("no NaN in features",
      np.isfinite(D['node_feats']).all() and np.isfinite(D['edge_feats']).all())
c = np.bincount(D['t_risk'], minlength=3)
check("all three risk classes present", (c > 0).all(), f"S/M/U = {c[0]}/{c[1]}/{c[2]}")
print(f"     generation rate: {600/gen_s:.1f} samples/s  ->  30k in {30000/(600/gen_s)/60:.0f} min")

# ---------------------------------------------------------------- 2. split
print("\n2. grouped (operating-point-disjoint) split")
gen118.make_grouped_split(NPZ, SPLIT)
S = np.load(SPLIT)
tr, va, te = S['idx_tr'], S['idx_va'], S['idx_te']
check("split covers every sample", len(tr) + len(va) + len(te) == 600,
      f"{len(tr)}+{len(va)}+{len(te)}")
check("splits are disjoint",
      len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0 and len(set(va) & set(te)) == 0)
ops = [tuple(np.round(r, 4)) for r in D['opcond']]
op_tr = {ops[i] for i in tr}; op_te = {ops[i] for i in te}
check("operating points are disjoint across train/test", len(op_tr & op_te) == 0,
      f"overlap={len(op_tr & op_te)}")
check("test split non-empty and has >1 class",
      len(te) > 0 and len(np.unique(D['t_risk'][te])) > 1)

# ---------------------------------------------------------------- 3. build_data
print("\n3. tensor assembly")
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
d = build_data(NPZ, SPLIT, dev=dev)
check("device", True, dev)
check("NF normalised finite", torch.isfinite(d['NF']).all().item())
check("EF normalised finite", torch.isfinite(d['EF']).all().item())
check("class weights finite", torch.isfinite(d['cw']).all().item(), str(d['cw'].tolist()))
nf, cf, adj = train118.batch(d, d['idx_tr'][:16])
check("batch node feats", tuple(nf.shape) == (16, 118, 4), str(tuple(nf.shape)))
check("batch cell feats", cf.shape[0] == 16 and cf.shape[2] == 3, str(tuple(cf.shape)))
check("batch adjacency", tuple(adj.shape) == (16, 118, 118), str(tuple(adj.shape)))
check("adjacency symmetric", torch.equal(adj, adj.transpose(1, 2)))
check("adjacency has self-loops", bool((adj[:, torch.arange(118), torch.arange(118)] == 1).all()))
n_cell = d['ctx']['n_cell']
check("parallel branches collapsed", n_cell == 179, f"n_cell={n_cell} from 186 branches")

# ---------------------------------------------------------------- 4. train
print("\n4. training loop (4 epochs -- shape/stability check only)")
res, model = train_eval(d, epochs=4, bs=64, verbose=True, log_every=1, amp=False)
check("no non-finite steps skipped", res['nonfinite_steps_skipped'] == 0,
      f"skipped={res['nonfinite_steps_skipped']}")
for k in ('acc', 'vmin_mae', 'vmin_r2', 'ece', 'brier', 'nll'):
    v = res.get(k, None)
    check(f"metric '{k}' finite", v is not None and np.isfinite(v), f"{v}")
check("confusion matrix totals match test size",
      sum(sum(r) for r in res['confusion']) == len(te))
check("vulnerability PR-AUC reported",
      'vuln_pr_auc' in res and np.isfinite(res['vuln_pr_auc']),
      f"ROC-AUC={res.get('vuln_roc_auc')} PR-AUC={res.get('vuln_pr_auc')} "
      f"pos_rate={res.get('vuln_positive_rate_converged')}")
check("params == 163592", res['params'] == 163592, str(res['params']))

# ---------------------------------------------------------------- 5. screening
print("\n5. screening benchmark (reviewer R1 #6)")
sb = screening_benchmark(model, d, gen118, n_repeat=1)
check("sweep is scored against the oracle",
      'sweep_false_safe' in sb and 'sweep_accuracy' in sb,
      f"acc={sb.get('sweep_accuracy')} false_safe={sb.get('sweep_false_safe')}")
check("oracle time positive", sb['oracle_s'] > 0)
check("end-to-end speedup finite", np.isfinite(sb['speedup_end_to_end']))
check("forward-only speedup >= end-to-end",
      sb['speedup_forward_only'] >= sb['speedup_end_to_end'])

# ---------------------------------------------------------------- 6. breakdown
print("\n6. Unstable-class decomposition (reviewer R1 #4)")
ub = unstable_breakdown(NPZ, gen118)
print("     " + json.dumps(ub))
check("breakdown sums to the Unstable count",
      ub['voltage_violation'] + ub['divergence_islanding'] + ub['divergence_connected']
      == ub['n_unstable'])
check("solo-islanding branch count matches probe", ub['n_islanding_branches_solo'] == 9,
      f"got {ub['n_islanding_branches_solo']}")

# ---------------------------------------------------------------- done
print("\n" + "=" * 72)
print("SMOKE TEST " + ("PASSED" if ok else "FAILED"))
print("=" * 72)
sys.exit(0 if ok else 1)
