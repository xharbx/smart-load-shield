"""
Figures for the contingency-screening paper (no embedded titles, per house style).
  confusion_cs.png     - risk confusion matrix on the test set
  vmin_parity_cs.png   - predicted vs true POST-contingency V_min (exact oracle)
  baselines_cs.png     - GNN vs same-data LR/RF/MLP: accuracy + false-safe
Run AFTER train_cs.py (needs cs_model.pt) and baselines_cs.py.
"""
import numpy as np, torch, os, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from model_cs import ContingencyGNN

# academic house style (matches paper/analyses.py line-plot figures)
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.linewidth': 1.0,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

HERE = os.path.dirname(__file__)
OUT = os.path.join(os.path.dirname(HERE), 'paper', 'CJECE_IEEE')   # write straight into paper dir
D = np.load(os.path.join(HERE, 'contingency_data.npz'))
N_BUS = int(D['n_bus']); pairs = torch.tensor(D['edge_pairs'], dtype=torch.long)
fr, to = pairs[:, 0], pairs[:, 1]
S = np.load(os.path.join(HERE, 'grouped_split.npz'))   # grouped-by-OP split + train-only norm
nmean = torch.tensor(S['nmean']); nstd = torch.tensor(S['nstd'])
emean = torch.tensor(S['emean']); estd = torch.tensor(S['estd'])
NF = (torch.tensor(D['node_feats']) - nmean) / nstd
EFb = (torch.tensor(D['edge_feats']) - emean) / estd
INS = torch.tensor(D['in_service']); T_risk = torch.tensor(D['t_risk'])
T_vmin = torch.tensor(D['t_vmin']); T_div = torch.tensor(D['t_div'])
idx_te = torch.tensor(S['idx_te'].astype('int64'))


def build_dense(b):
    B = len(b); ef = EFb[b]; ins = INS[b]
    dense = torch.zeros(B, N_BUS, N_BUS, 3)
    dense[:, fr, to, 0:2] = ef; dense[:, to, fr, 0:2] = ef
    dense[:, fr, to, 2] = ins; dense[:, to, fr, 2] = ins
    adj = torch.zeros(B, N_BUS, N_BUS)
    adj[:, fr, to] = ins; adj[:, to, fr] = ins
    adj[:, torch.arange(N_BUS), torch.arange(N_BUS)] = 1.0
    return dense, adj


T_vuln = torch.tensor(D['t_vuln'])
m = ContingencyGNN(); m.load_state_dict(torch.load(os.path.join(HERE, 'cs_model.pt'))); m.eval()
P, VM, VU = [], [], []
with torch.no_grad():
    for s in range(0, len(idx_te), 256):
        b = idx_te[s:s + 256]; dense, adj = build_dense(b)
        risk, vmin, vuln, *_ = m(NF[b], dense, adj)
        P.append(risk.argmax(1)); VM.append(vmin); VU.append(vuln)
P = torch.cat(P).numpy(); VMp = torch.cat(VM).numpy(); VUp = torch.cat(VU).numpy()
R = T_risk[idx_te].numpy(); VMt = T_vmin[idx_te].numpy(); conv = T_div[idx_te].numpy() == 0
VUt = T_vuln[idx_te].numpy()
VBt = D['t_vbus'][idx_te.numpy()]   # per-bus voltages, aligned to current (grouped) idx_te order

# 1) confusion matrix
cm = np.zeros((3, 3), int)
for t, p in zip(R, P): cm[t, p] += 1
fig, ax = plt.subplots(figsize=(3.8, 3.4))
im = ax.imshow(cm, cmap='Blues')
labs = ['Stable', 'Marginal', 'Unstable']
ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
ax.set_xticklabels(labs); ax.set_yticklabels(labs)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'confusion_cs.png'), dpi=170); plt.close(fig)

# 2) post-contingency Vmin parity (converged only; exact oracle labels)
fig, ax = plt.subplots(figsize=(4.3, 4.0))
ax.grid(alpha=0.3); ax.set_axisbelow(True)
ax.scatter(VMt[conv], VMp[conv], s=8, alpha=0.25, color='#1f77b4', edgecolors='none')
lo, hi = 0.5, 1.02
ax.plot([lo, hi], [lo, hi], '--', color='gray', lw=1.3)
mae = np.abs(VMp[conv] - VMt[conv]).mean()
ss_res = ((VMt[conv] - VMp[conv]) ** 2).sum(); ss_tot = ((VMt[conv] - VMt[conv].mean()) ** 2).sum()
r2 = 1 - ss_res / ss_tot
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel('True post-contingency $V_{\\min}$ (p.u.)')
ax.set_ylabel('Predicted $V_{\\min}$ (p.u.)')
ax.text(0.05, 0.93, f'MAE = {mae:.3f}\n$R^2$ = {r2:.3f}', transform=ax.transAxes, va='top',
        bbox=dict(boxstyle='round', fc='white', ec='gray'))
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'vmin_parity_cs.png')); plt.close(fig)

# 3) baselines comparison
csr = json.load(open(os.path.join(HERE, 'cs_results.json')))
bl = json.load(open(os.path.join(HERE, 'baselines_results.json')))
names = ['LogReg', 'RandForest', 'MLP', 'GNN (ours)']
accs = [bl['logreg']['acc'], bl['rf']['acc'], bl['mlp']['acc'], csr['test_acc']]
fss = [bl['logreg']['false_safe'], bl['rf']['false_safe'], bl['mlp']['false_safe'], csr['false_safe']]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.2))
cols = ['#888', '#888', '#888', '#1f77b4']
a1.bar(names, [a * 100 for a in accs], color=cols); a1.set_ylabel('Test accuracy (%)'); a1.set_ylim(60, 100)
a2.bar(names, [f * 100 for f in fss], color=cols); a2.set_ylabel('False-safe rate (%)')
for a in (a1, a2):
    a.set_xticklabels(names, rotation=20, ha='right')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'baselines_cs.png'), dpi=170); plt.close(fig)

# 4) per-bus vulnerability vs TRUE post-contingency bus voltage (scatter; matches Fig.4 style)
from sklearn.metrics import roc_auc_score
vu_auc = roc_auc_score(VUt[conv].ravel(), VUp[conv].ravel())
xb = VBt[conv].ravel(); yv = VUp[conv].ravel()
keep = xb > 0.3                                   # drop divergence sentinels if any slipped in
xb, yv = xb[keep], yv[keep]
# subsample for a clean, non-saturated scatter
rs = np.random.RandomState(0)
if len(xb) > 12000:
    sel = rs.choice(len(xb), 12000, replace=False); xb, yv = xb[sel], yv[sel]
fig, ax = plt.subplots(figsize=(4.9, 4.0))
ax.grid(alpha=0.3); ax.set_axisbelow(True)
ax.scatter(xb, yv, s=8, alpha=0.2, color='#1f77b4', edgecolors='none')
ax.axvline(0.95, ls='--', color='gray', lw=1.3)
ax.set_xlabel('True post-contingency bus voltage (p.u.)')
ax.set_ylabel('Predicted per-bus vulnerability')
ax.set_xlim(0.55, 1.02); ax.set_ylim(-0.02, 1.02)
ax.text(0.05, 0.5, f'ROC-AUC = {vu_auc:.3f}', transform=ax.transAxes, va='center',
        bbox=dict(boxstyle='round', fc='white', ec='gray'))
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'vuln_cs.png'), pad_inches=0.12); plt.close(fig)

print(f"figures written to {OUT}")
print(f"  vuln_cs.png  per-bus ROC-AUC {vu_auc:.3f}")
print(f"  confusion_cs.png  test acc {(P==R).mean()*100:.2f}%")
print(f"  vmin_parity_cs.png  MAE {mae:.4f}  R2 {r2:.3f}")
print(f"  baselines_cs.png  accs {[round(a,3) for a in accs]}")
