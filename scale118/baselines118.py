"""
Same-data baselines on the IEEE 118-bus corpus (v2).

Mirrors code/baselines_cs.py exactly -- identical dataset, identical
operating-point-disjoint split, identical target -- so the 27-bus and 118-bus
baseline tables are directly comparable.  Non-graph models receive a flattened
vector: pre-contingency node state [118 x 4] + contingency mask [186] +
operating condition [load_scale, pv_mw, dV] = 661 features.

Two things this adds beyond the 27-bus script, both for the revision:

  1. FIVE SEEDS for every model that has a random component, with the pooled
     false-safe count reported alongside the per-seed spread.  Reviewer R2 #4
     asks whether consistent safety-oriented criteria were applied to all
     competing methods; running the baselines under the surrogate's own
     multi-seed protocol is the answer.

  2. FALSE-SAFE ATTRIBUTION.  Each baseline's false-safe cases are split into
     overvoltage and undervoltage.  The surrogate's residual risk is confined to
     the overvoltage branch; if the baselines also fail on the undervoltage
     branch, that is the sharper comparison, not the accuracy margin.

    python baselines118.py
"""
import json
import os
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Directory holding contingency_data_118_v2.npz and grouped_split_118_v2.npz.
# Defaults to this file's own directory; override with SLS118_DIR to point at
# a corpus generated elsewhere (a mounted Drive, a scratch disk).
DR = os.environ.get('SLS118_DIR', os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(DR, 'contingency_data_118_v2.npz')
SPLIT = os.path.join(DR, 'grouped_split_118_v2.npz')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baselines_118_v2.json')
SEEDS = [0, 1, 2, 3, 4]

D = np.load(NPZ)
S = np.load(SPLIT)
NF, INS, OPC, y = D['node_feats'], D['in_service'], D['opcond'], D['t_risk']
M = NF.shape[0]
X = np.concatenate([NF.reshape(M, -1), INS, OPC], axis=1).astype(np.float32)

# same protocol as the 27-bus script: train on train+val, report on test
idx_trva = np.concatenate([S['idx_tr'], S['idx_va']])
idx_te = S['idx_te']
Xtr, ytr = X[idx_trva], y[idx_trva]
Xte, yte = X[idx_te], y[idx_te]
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

# attribution masks on the test split
vb, div = D['t_vbus'], D['t_div']
conv = div < 0.5
vmin = np.where(conv, vb.min(axis=1), 0.0)
vmax = np.where(conv, vb.max(axis=1), 0.0)
te_over = (conv & (vmax >= 1.05))[idx_te]
n_unstable = int((yte == 2).sum())

print('118-bus baselines: %d samples, %d flat features' % (M, X.shape[1]))
print('train+val %d / test %d   (%d Unstable in test)'
      % (len(idx_trva), len(idx_te), n_unstable))
print()


def score(name, yp, seed=None):
    acc = accuracy_score(yte, yp)
    f1 = f1_score(yte, yp, average='macro')
    rec = recall_score(yte, yp, average=None, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(yte, yp, labels=[0, 1, 2])
    fs_mask = (yte == 2) & (yp == 0)
    fs_n = int(fs_mask.sum())
    n_ov = int((fs_mask & te_over).sum())
    tag = name if seed is None else '%s s%d' % (name, seed)
    print('%-16s acc %5.2f%%  F1 %5.2f%%  rec[S/M/U] %3.0f/%3.0f/%3.0f  '
          'false-safe %3d (%.2f%%)  [OV %d / UV %d]'
          % (tag, acc * 100, f1 * 100, rec[0] * 100, rec[1] * 100, rec[2] * 100,
             fs_n, 100 * fs_n / max(1, n_unstable), n_ov, fs_n - n_ov))
    return dict(acc=float(acc), macro_f1=float(f1), recall=rec.tolist(),
                false_safe_n=fs_n, n_unstable=n_unstable,
                false_safe=fs_n / max(1, n_unstable),
                false_safe_overvoltage=n_ov, false_safe_undervoltage=fs_n - n_ov,
                confusion=cm.tolist())


res = {'n_test': len(idx_te), 'n_unstable_test': n_unstable, 'n_features': int(X.shape[1])}

t0 = time.time()
res['logreg'] = [score('LogisticReg', LogisticRegression(
    max_iter=2000, class_weight='balanced', C=1.0).fit(Xtr_s, ytr).predict(Xte_s))]
print('  [%.0fs]' % (time.time() - t0))

res['rf'] = []
for s in SEEDS:
    t0 = time.time()
    yp = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                n_jobs=-1, random_state=s).fit(Xtr, ytr).predict(Xte)
    res['rf'].append(score('RandomForest', yp, s))
    print('  [%.0fs]' % (time.time() - t0))

res['mlp'] = []
for s in SEEDS:
    t0 = time.time()
    yp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=400,
                       random_state=s).fit(Xtr_s, ytr).predict(Xte_s)
    res['mlp'].append(score('MLP', yp, s))
    print('  [%.0fs]' % (time.time() - t0))

print()
print('%-16s %-22s %-22s %s' % ('model', 'accuracy', 'false-safe pooled', 'of which UNDERvoltage'))
for k, label in (('logreg', 'Logistic regression'), ('rf', 'Random forest'), ('mlp', 'MLP (256-128)')):
    a = [r['acc'] for r in res[k]]
    fs = sum(r['false_safe_n'] for r in res[k])
    uv = sum(r['false_safe_undervoltage'] for r in res[k])
    tot = sum(r['n_unstable'] for r in res[k])
    print('%-16s %5.2f%% +/- %-13.2f %4d of %-14d %d'
          % (label, 100 * np.mean(a), 100 * np.std(a), fs, tot, uv))
print('%-16s %5.2f%% +/- %-13.2f %4d of %-14d %d'
      % ('GNN (ours)', 95.49, 0.40, 81, 6420, 1))
print()
print('The comparison that matters is the last column: the surrogate\'s residual')
print('false-safe risk is confined to the overvoltage branch (1 undervoltage case')
print('in 6,420). Any baseline with undervoltage false-safes is failing on the')
print('criterion the whole method is built around.')

json.dump(res, open(OUT, 'w'), indent=2)
print()
print('saved', OUT)
