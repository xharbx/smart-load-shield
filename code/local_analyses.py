"""Round-7 local analyses on the DEPLOYED model (full_a0, masked coupled) + grouped-split test.
No retrain. Computes:
  #11 calibration: overall ECE + classwise (one-vs-rest) ECE + reliability-diagram bins (-> reliability_cs.png)
  #12 cross-head consistency: P(risk=Stable & Vmin<0.95), P(pdiv>0.5 & risk!=Unstable), vuln<->Vmin agreement
  #13 boundary Vmin MAE: MAE/max/p95/bias in [0.88,0.92] & [0.93,0.97] (+ overall)
  #2  subgroup nonconvergence recall: islanding vs connected-collapse; classify the missed cases
Saves local_analyses_results.json + reliability_cs.png.
"""
import json, numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({                       # academic house style (matches make_figs_cs.py / Fig 7)
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13, 'legend.fontsize': 10,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'axes.linewidth': 1.0,
    'savefig.dpi': 200, 'savefig.bbox': 'tight'})
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import boost_core as B

DEV = 'cpu'
d = B.build_data('contingency_data.npz', 'grouped_split.npz', DEV)
m = B.CSGNNv2(node_dim=4, edge_dim=3, hidden=128, heads=4, branched=False, stack=False, vbus=False)
m.load_state_dict(torch.load('full_a0.pt', map_location='cpu')); m.set_base_mask(d['BASE']); m.eval()

D = np.load('contingency_data.npz')
bus_ids = D['bus_ids']; pairs = D['edge_pairs']; n_bus = int(D['n_bus'])
slack_idx = int(np.where(bus_ids == 1)[0][0])          # slack generator is Bus 1

idx = d['idx_te']
P, PROB, VM, DV, VU = [], [], [], [], []
with torch.no_grad():
    for s in range(0, len(idx), 4096):
        b = idx[s:s+4096]
        r, vm, vu, dv, vb = m(d['NF'][b], d['DENSE'][b], d['ADJ'][b])
        P.append(r.argmax(1)); PROB.append(torch.softmax(r, 1)); VM.append(vm)
        DV.append(torch.sigmoid(dv)); VU.append(torch.sigmoid(vu))
P = torch.cat(P).numpy(); PROB = torch.cat(PROB).numpy(); VM = torch.cat(VM).numpy()
DV = torch.cat(DV).numpy(); VU = torch.cat(VU).numpy()
R = d['Tr'][idx].numpy(); Td = d['Td'][idx].numpy(); Tv = d['Tv'][idx].numpy()
INS_te = D['in_service'][idx.numpy()]
conv = (Td == 0)
out = {}

# ---------- #11 calibration ----------
def ece_bins(conf, correct, nb=15):
    edges = np.linspace(0, 1, nb + 1); e = 0.0; rows = []
    for i in range(nb):
        msk = (conf > edges[i]) & (conf <= edges[i + 1]); n = int(msk.sum())
        if n == 0:
            rows.append((0.5 * (edges[i] + edges[i + 1]), np.nan, 0.0, 0)); continue
        acc_b = float(correct[msk].mean()); cf_b = float(conf[msk].mean())
        e += (n / len(conf)) * abs(acc_b - cf_b)
        rows.append((cf_b, acc_b, n / len(conf), n))
    return e, rows

topconf = PROB.max(1); toppred = PROB.argmax(1); correct = (toppred == R).astype(float)
ece, rel_rows = ece_bins(topconf, correct)
classwise = {}
for c in range(3):
    ec, _ = ece_bins(PROB[:, c], (R == c).astype(float))
    classwise[['Stable', 'Marginal', 'Unstable'][c]] = ec
out['calibration'] = dict(ece=ece, classwise_ece=classwise, classwise_mean=float(np.mean(list(classwise.values()))))

# reliability diagram (house style: #1f77b4 bars, grey dashed diagonal, grid; no population markers)
edges = np.linspace(0, 1, 16); centers = 0.5 * (edges[:-1] + edges[1:])
accs_bin = [r[1] for r in rel_rows]                  # NaN where the bin is empty
xs = [c for c, a in zip(centers, accs_bin) if not np.isnan(a)]
ys = [a for a in accs_bin if not np.isnan(a)]
fig, ax = plt.subplots(figsize=(4.3, 4.0))
ax.grid(alpha=0.3); ax.set_axisbelow(True)
ax.plot([0, 1], [0, 1], '--', color='gray', lw=1.3, label='perfect calibration')
ax.bar(xs, ys, width=(1 / 15) * 0.9, color='#1f77b4', alpha=0.85, edgecolor='#0d3b5c', lw=0.5, label='model accuracy')
ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy'); ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.0)
ax.legend(loc='upper left', frameon=False)
fig.tight_layout(); fig.savefig('reliability_cs.png'); plt.close(fig)

# ---------- #12 cross-head consistency ----------
n = len(R)
c1 = int(((P == 0) & (VM < 0.95)).sum())                 # risk says Stable but Vmin head < 0.95
c2 = int(((DV > 0.5) & (P != 2)).sum())                  # nonconv fires but risk not Unstable
vuln_any = VU.max(1) > 0.5                                # any bus predicted vulnerable (<0.95)
vmin_low = VM < 0.95
agree = float((vuln_any == vmin_low).mean())             # the two "some bus < 0.95" signals agree
out['cross_head'] = dict(
    n=n,
    p_stable_and_vmin_low=[c1, c1 / n],
    p_div_and_not_unstable=[c2, c2 / n],
    vuln_vmin_agreement=agree,
    corr_vmin_minvuln=float(np.corrcoef(VM, VU.max(1))[0, 1]))

# ---------- #13 boundary Vmin MAE ----------
def band_stats(lo, hi):
    msk = conv & (Tv >= lo) & (Tv < hi); e = np.abs(VM[msk] - Tv[msk])
    if len(e) == 0: return dict(n=0)
    return dict(n=int(msk.sum()), mae=float(e.mean()), max=float(e.max()),
                p95=float(np.percentile(e, 95)), bias=float((VM[msk] - Tv[msk]).mean()))
eall = np.abs(VM[conv] - Tv[conv])
out['boundary_vmin'] = dict(
    overall=dict(n=int(conv.sum()), mae=float(eall.mean()), max=float(eall.max()),
                 p95=float(np.percentile(eall, 95)), bias=float((VM[conv] - Tv[conv]).mean())),
    band_0p88_0p92=band_stats(0.88, 0.92),
    band_0p93_0p97=band_stats(0.93, 0.97))

# ---------- #2 subgroup nonconvergence recall (islanding vs connected-collapse) ----------
def is_islanding(insvc_row):
    live = pairs[insvc_row == 1]                         # in-service branches
    if len(live) == 0: return True
    A = csr_matrix((np.ones(len(live) * 2),
                    (np.r_[live[:, 0], live[:, 1]], np.r_[live[:, 1], live[:, 0]])),
                   shape=(n_bus, n_bus))
    ncomp, lab = connected_components(A, directed=False)
    return bool((lab != lab[slack_idx]).any())           # some bus not in slack's component

div_mask = Td == 1
div_pos = np.where(div_mask)[0]
isl = np.array([is_islanding(INS_te[i]) for i in div_pos])
pred_div = DV[div_pos] > 0.5                             # nonconv head fires
def rec(sub):
    return dict(n=int(sub.sum()), caught=int((pred_div & sub).sum()),
                missed=int((~pred_div & sub).sum()),
                recall=float((pred_div & sub).sum() / max(1, sub.sum())))
missed_pos = div_pos[~pred_div]
out['nonconv_subgroup'] = dict(
    n_nonconvergent=int(div_mask.sum()),
    islanding=rec(isl), collapse=rec(~isl),
    n_missed=int((~pred_div).sum()),
    missed_islanding=int(isl[~pred_div].sum()),
    missed_collapse=int((~isl[~pred_div]).sum()),
    missed_true_risk=[int(R[i]) for i in missed_pos])     # all should be 2 (still escalated by risk head)

json.dump(out, open('local_analyses_results.json', 'w'), indent=2)
print(json.dumps(out, indent=2))
print('\nsaved local_analyses_results.json + reliability_cs.png')
