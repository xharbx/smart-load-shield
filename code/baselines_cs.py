"""
Same-data baselines for the contingency-screening task (review issue #7).

Every model is trained and tested on the IDENTICAL dataset, split, and target
definition as the proposed GNN.  Non-graph models receive a flattened feature
vector: pre-contingency node state [N x 4] + contingency mask [n_branch] +
operating condition [load_scale, pv_mw].

Reports per model: risk accuracy, macro-F1, per-class recall, FALSE-SAFE rate.
"""
import numpy as np, os, json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

HERE = os.path.dirname(__file__)
D = np.load(os.path.join(HERE, 'contingency_data.npz'))
NF = D['node_feats']; INS = D['in_service']; OPC = D['opcond']; y = D['t_risk']
M = NF.shape[0]
X = np.concatenate([NF.reshape(M, -1), INS, OPC], axis=1).astype(np.float32)
print(f"baselines: {M} samples, {X.shape[1]} flat features")

# SAME grouped-by-operating-point split as the GNN (train+val vs test; no OP shared)
S = np.load(os.path.join(HERE, 'grouped_split.npz'))
idx_trva = np.concatenate([S['idx_tr'], S['idx_va']]); idx_te = S['idx_te']
Xtr, ytr = X[idx_trva], y[idx_trva]; Xte, yte = X[idx_te], y[idx_te]
sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)


def report(name, yp):
    acc = accuracy_score(yte, yp)
    f1 = f1_score(yte, yp, average='macro')
    rec = recall_score(yte, yp, average=None, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(yte, yp, labels=[0, 1, 2])
    fs = cm[2, 0] / max(1, cm[2].sum())            # true Unstable -> pred Stable
    print(f"{name:18s} acc {acc*100:5.2f}%  F1 {f1*100:5.2f}%  "
          f"rec[S/M/U] {rec[0]*100:4.0f}/{rec[1]*100:4.0f}/{rec[2]*100:4.0f}  false-safe {fs*100:.2f}%")
    return {'acc': acc, 'f1': f1, 'recall': rec.tolist(), 'false_safe': fs}


res = {}
res['logreg'] = report('LogisticReg', LogisticRegression(
    max_iter=2000, class_weight='balanced', C=1.0).fit(Xtr_s, ytr).predict(Xte_s))
res['rf'] = report('RandomForest', RandomForestClassifier(
    n_estimators=300, class_weight='balanced', n_jobs=-1, random_state=0).fit(Xtr, ytr).predict(Xte))
res['mlp'] = report('MLP', MLPClassifier(
    hidden_layer_sizes=(256, 128), max_iter=400, random_state=0).fit(Xtr_s, ytr).predict(Xte_s))
json.dump(res, open(os.path.join(HERE, 'baselines_results.json'), 'w'), indent=2)
print("\nsaved baselines_results.json")
