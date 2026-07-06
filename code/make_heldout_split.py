"""Round-7 #6: held-out-CONTINGENCY split (branch-identity generalization test).

Distinguishes physics-generalization from branch-identity memorization. We hold out
20% of branches (H); every sample that trips at least one held-out branch is the
GENERALIZATION test set (never seen during training). The remaining samples (whose
tripped branches are all "seen", plus the base k=0 cases) form the trainable pool,
split sample-wise into train / val / in-distribution test.

Design (reviewed, Fable-5):
  * Operating points are SHARED across train and held-out test on purpose -- this
    isolates the branch-identity variable (OP-generalization is already covered by the
    main grouped-by-OP split; the two splits are complementary factorial cells).
  * The in-distribution test is a SAMPLE-level slice of the trainable pool, so it shares
    OPs with train exactly the way the held-out test does => the ID-vs-heldout gap is a
    clean single-variable (branch novelty) measurement from the SAME trained weights.
  * Checkpoint selection uses ID val only; the held-out set is touched once, at final eval.
  * NEVER compare these numbers to the deployed 98.12% (different split protocol / train size).

Saves heldout_split.npz with the build_data keys (idx_tr/idx_va/idx_te + train-only norm)
PLUS idx_heldout, the held-out branch list H, and per-sample trip count k so the notebook
can produce the per-branch table and the pure-N-1 breakout.
"""
import numpy as np, os
HERE = os.path.dirname(__file__)
D = np.load(os.path.join(HERE, 'contingency_data.npz'))
INS = D['in_service']; n_branch = int(D['n_branch'])
tripped = (INS == 0); k = tripped.sum(1).astype('int64')     # trips per sample
M = INS.shape[0]

# ---- hold out 20% of branches (fixed seed 7; draw disclosed in the paper) ----
HOLD_SEED = 7
rng_h = np.random.RandomState(HOLD_SEED)
H = np.sort(rng_h.permutation(n_branch)[:int(0.2 * n_branch)]).astype('int64')   # 9 branches

test_heldout = tripped[:, H].any(1)          # >=1 held-out branch tripped -> generalization test
trainable = ~test_heldout                    # all tripped branches seen (incl. base k=0)
idx_heldout = np.where(test_heldout)[0]

# ---- sample-wise split of the trainable pool: 75 train / 10 val / 15 in-dist test ----
SPLIT_SEED = 0
pool = np.where(trainable)[0]
rng_s = np.random.RandomState(SPLIT_SEED)
perm = rng_s.permutation(len(pool))
n1 = int(0.75 * len(pool)); n2 = int(0.85 * len(pool))
idx_tr = np.sort(pool[perm[:n1]])
idx_va = np.sort(pool[perm[n1:n2]])
idx_te = np.sort(pool[perm[n2:]])            # in-distribution reference test (seen branches)

assert set(idx_tr) & set(idx_heldout) == set()      # held-out never in train
assert set(idx_va) & set(idx_heldout) == set()
assert set(idx_te) & set(idx_heldout) == set()

# ---- train-only normalization (per feature) ----
raw_nf = D['node_feats'].astype('float32'); raw_ef = D['edge_feats'].astype('float32')
nmean = raw_nf[idx_tr].reshape(-1, 4).mean(0); nstd = raw_nf[idx_tr].reshape(-1, 4).std(0) + 1e-6
emean = raw_ef[idx_tr].reshape(-1, 2).mean(0); estd = raw_ef[idx_tr].reshape(-1, 2).std(0) + 1e-6

np.savez(os.path.join(HERE, 'heldout_split.npz'),
         idx_tr=idx_tr, idx_va=idx_va, idx_te=idx_te, idx_heldout=idx_heldout,
         nmean=nmean, nstd=nstd, emean=emean, estd=estd,
         H=H, k=k, hold_seed=HOLD_SEED, split_seed=SPLIT_SEED)

R = D['t_risk']
def sm(ix): return f"{int((R[ix]==0).sum())}/{int((R[ix]==1).sum())}/{int((R[ix]==2).sum())}"
print(f"held-out branches H ({len(H)}): {H.tolist()}")
print(f"train/val/in-dist-test/HELD-OUT = {len(idx_tr)}/{len(idx_va)}/{len(idx_te)}/{len(idx_heldout)}")
print(f"risk S/M/U   train={sm(idx_tr)}  in-dist-test={sm(idx_te)}  HELD-OUT={sm(idx_heldout)}")
n1h = int(((k == 1) & test_heldout).sum())
print(f"pure N-1 held-out (single tripped branch, held-out): {n1h}")
print("saved heldout_split.npz")
