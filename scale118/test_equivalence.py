"""
EQUIVALENCE GATE.

Proves that the memory-efficient 118-bus model (core118.CSGNN118) computes
EXACTLY the same function as the deployed 27-bus model (boost_core.CSGNNv2).
If this does not pass, no number produced by the 118-bus run can be compared
with a number in the paper, and nothing downstream is worth running.

Four tests:
  T1  Graph context on the real 27-bus topology is the identity mapping, and the
      reconstructed adjacency / base mask match boost_core's dense construction
      cell for cell.
  T2  Real deployed weights (full_a0.pt) on real 27-bus data: all five outputs
      agree to < 1e-5 in float32, dropout off.
  T3  All four architecture variants (branched / stack / vbus / plain) with
      random weights agree to < 1e-5 -- exercises every code path, including the
      ones full_a0.pt does not contain.
  T4  Parallel-branch semantics on a synthetic network: the dense reference is
      ill-defined there (last write wins), so instead we assert the three
      properties we deliberately chose -- OR adjacency, averaged features, and
      correct reduction to the single-branch case.
"""
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# boost_core.py, the 27-bus reference implementation, lives in code/.
CODE = os.path.abspath(os.path.join(HERE, '..', 'code'))
sys.path.insert(0, HERE)
sys.path.insert(0, CODE)

import core118
from core118 import CSGNN118, build_graph_ctx, cells_and_adj, dense_from_compact
import boost_core
from boost_core import CSGNNv2

TOL = 1e-5
torch.manual_seed(0)
np.random.seed(0)

DATA = os.path.join(CODE, 'contingency_data.npz')
SPLIT = os.path.join(CODE, 'grouped_split.npz')
CKPT = os.path.join(CODE, 'full_a0.pt')

results = []


def report(name, ok, detail=""):
    results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


# ---------------------------------------------------------------- T1
def t1_graph_ctx():
    print("\nT1. graph context vs dense construction (real 27-bus topology)")
    D = np.load(DATA)
    ep = D['edge_pairs']
    N = int(D['n_bus'])
    ctx = build_graph_ctx(ep, N)

    report("n_cell == n_branch (no parallel branches on 27-bus)",
           ctx['n_cell'] == len(ep), f"n_cell={ctx['n_cell']} n_branch={len(ep)}")
    identity = torch.equal(ctx['cell_of_branch'], torch.arange(len(ep)))
    report("cell mapping is the identity", identity)

    B = 64
    ef = torch.tensor(D['edge_feats'][:B]).float()
    ins = torch.tensor(D['in_service'][:B]).float()

    cell_feat, adj_fast = cells_and_adj(ef, ins, ctx)
    dense, adj_ref, base_ref = dense_from_compact(ef, ins, ep, N)

    report("adjacency matches dense construction",
           torch.equal(adj_fast, adj_ref),
           f"maxdiff={float((adj_fast - adj_ref).abs().max()):.2e}")

    # cell_feat should equal the dense value at each branch cell
    fr = torch.tensor(ep[:, 0].astype(np.int64))
    to = torch.tensor(ep[:, 1].astype(np.int64))
    dense_at_cells = dense[:, fr, to, :]
    report("cell features match dense cell values",
           torch.allclose(cell_feat, dense_at_cells, atol=1e-7),
           f"maxdiff={float((cell_feat - dense_at_cells).abs().max()):.2e}")

    base_fast = torch.zeros(N, N)
    base_fast[ctx['cell_i'], ctx['cell_j']] = 1.0
    base_fast[ctx['cell_j'], ctx['cell_i']] = 1.0
    report("base mask matches", torch.equal(base_fast, base_ref))
    return ctx, D


# ---------------------------------------------------------------- T2
def t2_real_weights(ctx, D):
    print("\nT2. deployed weights (full_a0.pt) on real 27-bus data")
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    if not isinstance(sd, dict) or 'node_enc.0.weight' not in sd:
        sd = sd.get('state_dict', sd)
    cfg = dict(branched=('gat_r.W.weight' in sd),
               vbus=any(k.startswith('vbus_h') for k in sd),
               uncert=('log_s' in sd))
    cfg['stack'] = sd['risk_h.0.weight'].shape[1] > 384
    cfg['n_gat'] = (sum(1 for k in sd if k.startswith('gats.') and k.endswith('.W.weight'))
                    + (1 if cfg['branched'] else 0))
    print(f"     checkpoint config: {cfg}")

    N = int(D['n_bus'])
    ep = D['edge_pairs']
    S = np.load(SPLIT)
    idx = S['idx_te'][:96].astype(np.int64)

    nf = ((torch.tensor(D['node_feats'][idx]) - torch.tensor(S['nmean']))
          / torch.tensor(S['nstd'])).float()
    ef = ((torch.tensor(D['edge_feats'][idx]) - torch.tensor(S['emean']))
          / torch.tensor(S['estd'])).float()
    ins = torch.tensor(D['in_service'][idx]).float()

    ref = CSGNNv2(**cfg).eval()
    fast = CSGNN118(**cfg).eval()
    ref.load_state_dict(sd)
    fast.load_state_dict(sd)
    fast.set_graph(ctx)

    dense, adj, base = dense_from_compact(ef, ins, ep, N)
    ref.set_base_mask(base)
    cell_feat, adj2 = cells_and_adj(ef, ins, ctx)

    with torch.no_grad():
        o_ref = ref(nf, dense, adj)
        o_fast = fast(nf, cell_feat, adj2)

    names = ['risk', 'vmin', 'vuln', 'div', 'vbus']
    worst = 0.0
    for nm, a, b in zip(names, o_ref, o_fast):
        if a is None and b is None:
            report(f"output {nm}: both None (head absent)", True)
            continue
        d = float((a - b).abs().max())
        worst = max(worst, d)
        report(f"output {nm} within {TOL:g}", d < TOL, f"maxdiff={d:.3e}")
    # the decision that actually matters
    same_pred = torch.equal(o_ref[0].argmax(1), o_fast[0].argmax(1))
    report("identical predicted risk class on all 96 samples", same_pred)
    print(f"     worst absolute difference across all outputs: {worst:.3e}")


# ---------------------------------------------------------------- T3
def t3_all_variants(ctx, D):
    print("\nT3. all architecture variants, random weights")
    N = int(D['n_bus'])
    ep = D['edge_pairs']
    B = 32
    nf = torch.randn(B, N, 4)
    ef = torch.randn(B, len(ep), 2)
    ins = (torch.rand(B, len(ep)) > 0.15).float()

    dense, adj, base = dense_from_compact(ef, ins, ep, N)
    cell_feat, adj2 = cells_and_adj(ef, ins, ctx)

    for branched in (False, True):
        for stack in (False, True):
            for vbus in (False, True):
                cfg = dict(branched=branched, stack=stack, vbus=vbus, n_gat=3)
                torch.manual_seed(7)
                ref = CSGNNv2(**cfg).eval()
                torch.manual_seed(7)
                fast = CSGNN118(**cfg).eval()
                fast.load_state_dict(ref.state_dict())
                ref.set_base_mask(base)
                fast.set_graph(ctx)
                with torch.no_grad():
                    a = ref(nf, dense, adj)
                    b = fast(nf, cell_feat, adj2)
                d = max(float((x - y).abs().max())
                        for x, y in zip(a, b) if x is not None and y is not None)
                report(f"branched={int(branched)} stack={int(stack)} vbus={int(vbus)}",
                       d < TOL, f"maxdiff={d:.3e}")


# ---------------------------------------------------------------- T4
def t4_parallel_branches():
    print("\nT4. parallel-branch semantics (synthetic 5-bus network)")
    #        branches: 0:(0,1) 1:(1,2) 2:(1,2) parallel 3:(2,3) 4:(3,4)
    ep = np.array([[0, 1], [1, 2], [1, 2], [2, 3], [3, 4]], np.int64)
    N = 5
    ctx = build_graph_ctx(ep, N)
    report("parallel pair collapsed into one cell",
           ctx['n_cell'] == 4, f"n_cell={ctx['n_cell']} (expected 4)")
    report("cell 1 holds two branches",
           float(ctx['branch_per_cell'][1]) == 2.0)

    ef = torch.tensor([[[1., 10.], [2., 20.], [4., 40.], [3., 30.], [5., 50.]]])
    # trip ONE of the two parallel branches (branch index 2)
    ins = torch.tensor([[1., 1., 0., 1., 1.]])
    cell_feat, adj = cells_and_adj(ef, ins, ctx)

    report("corridor stays adjacent when one parallel circuit trips",
           float(adj[0, 1, 2]) == 1.0 and float(adj[0, 2, 1]) == 1.0)
    exp = torch.tensor([3.0, 30.0, 0.5])          # mean of (2,20,1) and (4,40,0)
    report("parallel cell features are averaged",
           torch.allclose(cell_feat[0, 1], exp, atol=1e-6),
           f"got {cell_feat[0,1].tolist()} expected {exp.tolist()}")

    # trip BOTH parallel circuits -> corridor must go down
    ins2 = torch.tensor([[1., 0., 0., 1., 1.]])
    _, adj2 = cells_and_adj(ef, ins2, ctx)
    report("corridor drops when both parallel circuits trip",
           float(adj2[0, 1, 2]) == 0.0)

    # a network without parallels must reduce exactly to the dense reference
    ep2 = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], np.int64)
    ctx2 = build_graph_ctx(ep2, N)
    ef2 = torch.randn(8, 4, 2)
    ins2 = (torch.rand(8, 4) > 0.3).float()
    cf2, adjf = cells_and_adj(ef2, ins2, ctx2)
    _, adjr, _ = dense_from_compact(ef2, ins2, ep2, N)
    report("no-parallel network reduces exactly to dense reference",
           torch.equal(adjf, adjr))


if __name__ == "__main__":
    print("=" * 72)
    print("EQUIVALENCE GATE: CSGNN118 (fast) vs CSGNNv2 (deployed)")
    print("=" * 72)
    ctx, D = t1_graph_ctx()
    t2_real_weights(ctx, D)
    t3_all_variants(ctx, D)
    t4_parallel_branches()

    print("\n" + "=" * 72)
    npass = sum(1 for _, ok in results if ok)
    nfail = len(results) - npass
    print(f"RESULT: {npass} passed, {nfail} failed")
    print("=" * 72)
    if nfail:
        for nm, ok in results:
            if not ok:
                print("  FAILED:", nm)
        sys.exit(1)
    print("Gate PASSED -- the fast model is numerically identical to the deployed one.")
