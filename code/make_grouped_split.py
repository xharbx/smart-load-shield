"""Reproduce the grouped-by-operating-point split + train-only normalisation used by
train_cs_grouped.py, and save it to grouped_split.npz so baselines/calibration/figures
all share the identical split and stats. Deterministic (RandomState(0)) -> matches the
indices cs_model_grouped.pt was trained on."""
import numpy as np, os
HERE = os.path.dirname(__file__)
D = np.load(os.path.join(HERE, 'contingency_data.npz'))
OPC = D['opcond']
uniq, inv = np.unique(OPC, axis=0, return_inverse=True)
n_op = len(uniq)
rng = np.random.RandomState(0)
op_perm = rng.permutation(n_op)
o1, o2 = int(0.70 * n_op), int(0.85 * n_op)
tr_ops = set(op_perm[:o1].tolist()); va_ops = set(op_perm[o1:o2].tolist()); te_ops = set(op_perm[o2:].tolist())
idx_tr = np.where(np.isin(inv, list(tr_ops)))[0]
idx_va = np.where(np.isin(inv, list(va_ops)))[0]
idx_te = np.where(np.isin(inv, list(te_ops)))[0]
assert tr_ops.isdisjoint(te_ops) and tr_ops.isdisjoint(va_ops) and va_ops.isdisjoint(te_ops)
# train-only normalisation stats (per-feature)
raw_nf = D['node_feats'].astype('float32'); raw_ef = D['edge_feats'].astype('float32')
nmean = raw_nf[idx_tr].reshape(-1, 4).mean(0); nstd = raw_nf[idx_tr].reshape(-1, 4).std(0) + 1e-6
emean = raw_ef[idx_tr].reshape(-1, 2).mean(0); estd = raw_ef[idx_tr].reshape(-1, 2).std(0) + 1e-6
np.savez(os.path.join(HERE, 'grouped_split.npz'),
         idx_tr=idx_tr, idx_va=idx_va, idx_te=idx_te,
         nmean=nmean, nstd=nstd, emean=emean, estd=estd, n_op=n_op)
print(f"n_op={n_op}  OPs tr/va/te={len(tr_ops)}/{len(va_ops)}/{len(te_ops)}")
print(f"scenarios tr/va/te={len(idx_tr)}/{len(idx_va)}/{len(idx_te)}")
print("saved grouped_split.npz")
