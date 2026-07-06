"""Deploy full_a0 as the paper model: compute ALL test metrics + regenerate confusion / vmin-parity /
vuln figures into paper/CJECE_IEEE, and write cs_results_full_a0.json. Grouped-split test."""
import os, json, numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import boost_core as B

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(os.path.dirname(HERE), 'paper', 'CJECE_IEEE')
plt.rcParams.update({'font.size': 11, 'savefig.dpi': 200, 'savefig.bbox': 'tight'})
d = B.build_data('contingency_data.npz', 'grouped_split.npz', 'cpu')
m = B.CSGNNv2(node_dim=4, edge_dim=3, hidden=128, heads=4, branched=False, stack=False, vbus=False)
m.load_state_dict(torch.load('full_a0.pt', map_location='cpu'))
m.set_base_mask(d['BASE'])          # round-6 #1: masked model needs the base-edge mask at inference
m.eval()
idx = d['idx_te']
P, prob, VM, VU = [], [], [], []
with torch.no_grad():
    for s in range(0, len(idx), 4096):
        b = idx[s:s+4096]
        r, vm, vu, dv, vb = m(d['NF'][b], d['DENSE'][b], d['ADJ'][b])
        P.append(r.argmax(1)); prob.append(torch.softmax(r, 1)); VM.append(vm); VU.append(torch.sigmoid(vu))
P = torch.cat(P); prob = torch.cat(prob).numpy(); VM = torch.cat(VM); VU = torch.cat(VU); R = d['Tr'][idx]
conf = torch.zeros(3, 3, dtype=torch.long)
for t, p in zip(R, P): conf[t, p] += 1
cm = conf.numpy()
acc = (P == R).float().mean().item()
recall = [cm[c, c] / max(1, cm[c].sum()) for c in range(3)]
false_safe = cm[2, 0] / max(1, cm[2].sum())
f1s = []
for k in range(3):
    tp = cm[k, k]; fp = cm[:, k].sum() - tp; fn = cm[k, :].sum() - tp
    pr = tp/max(1, tp+fp); rc = tp/max(1, tp+fn); f1s.append(2*pr*rc/max(1e-9, pr+rc))
macro_f1 = float(np.mean(f1s))
conv = (d['Td'][idx] == 0).numpy()
vt = d['Tv'][idx].numpy()[conv]; vp = VM.numpy()[conv]
vmae = float(np.abs(vt-vp).mean()); r2 = float(1 - ((vt-vp)**2).sum()/((vt-vt.mean())**2).sum())
r2fit = float(np.corrcoef(vt, vp)[0, 1] ** 2)      # fit R^2 (shown on the parity figure); r2 = identity R^2
vln_t = d['Tu'][idx].numpy()[conv].ravel(); vln_p = VU.numpy()[conv].ravel()
vuln_auc = float(roc_auc_score(vln_t, vln_p)); vuln_prauc = float(average_precision_score(vln_t, vln_p))
# calibration
Y = R.numpy(); bins = np.linspace(0, 1, 16); cf = prob.max(1); pl = prob.argmax(1); cor = (pl == Y)
ece = mce = mce_g = 0.0
for i in range(15):
    msk = (cf > bins[i]) & (cf <= bins[i+1]); n = int(msk.sum())
    if n:
        gap = abs(cor[msk].mean()-cf[msk].mean()); ece += msk.mean()*gap; mce = max(mce, gap)
        if n >= 20: mce_g = max(mce_g, gap)   # guarded MCE: ignore near-empty bins (raw MCE is a 1-sample artifact)
Yoh = np.eye(3)[Y]; brier = float(np.mean(np.sum((prob-Yoh)**2, 1))); nll = float(-np.mean(np.log(prob[np.arange(len(Y)), Y]+1e-12)))
roc = [float(roc_auc_score((Y == c).astype(int), prob[:, c])) for c in range(3)]
prc = [float(average_precision_score((Y == c).astype(int), prob[:, c])) for c in range(3)]

json.dump(dict(model='full_a0', test_acc=acc, macro_f1=macro_f1, recall=recall, false_safe=false_safe,
               confusion=cm.tolist(), n_test=int(len(idx)), n_unstable=int(cm[2].sum()), umarg=int(cm[2, 1]),
               vmin_mae=vmae, vmin_r2=r2, vmin_r2_fit=r2fit, vuln_auc=vuln_auc, vuln_prauc=vuln_prauc,
               ece=ece, mce=mce, mce_guarded=mce_g, brier=brier, nll=nll, roc_auc=roc, pr_auc=prc,
               nonconv=B.divergence_report(m, d, idx)),
          open('cs_results_full_a0.json', 'w'), indent=2)

# ---- figures ----
names = ['Stable', 'Marginal', 'Unstable']
fig, ax = plt.subplots(figsize=(4.2, 3.8))
im = ax.imshow(cm, cmap='Blues')
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12,
                color='white' if cm[i, j] > cm.max()/2 else 'black')
ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(names); ax.set_yticklabels(names)
ax.set_xlabel('Predicted'); ax.set_ylabel('True'); fig.tight_layout()
fig.savefig(os.path.join(OUT, 'confusion_cs.png'), dpi=170); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.3, 4.0)); ax.grid(alpha=0.3); ax.set_axisbelow(True)
ax.scatter(vt, vp, s=8, alpha=0.25, color='#1f77b4', edgecolors='none')
ax.plot([0.5, 1.02], [0.5, 1.02], '--', color='gray', lw=1.3, label='ideal $y=x$')
slope, intc = np.polyfit(vt, vp, 1)
r2fit = float(np.corrcoef(vt, vp)[0, 1] ** 2)      # coefficient of determination of the fit
xr = np.array([vt.min(), vt.max()])
ax.plot(xr, slope*xr + intc, '-', color='#d62728', lw=1.6, label='least-squares fit')
ax.set_xlim(0.5, 1.02); ax.set_ylim(0.5, 1.02)
ax.set_xlabel('True post-contingency $V_{\\min}$ (p.u.)'); ax.set_ylabel('Predicted $V_{\\min}$ (p.u.)')
ax.text(0.05, 0.95, f'$R^2$ = {r2fit:.2f}\nMAE = {vmae:.3f} p.u.', transform=ax.transAxes,
        va='top', bbox=dict(boxstyle='round', fc='white', ec='gray'))
ax.legend(loc='lower right', fontsize=9, framealpha=0.9); fig.tight_layout()
print(f'vmin fit: slope {slope:.3f} intc {intc:+.3f} R2_fit {r2fit:.3f}')
fig.savefig(os.path.join(OUT, 'vmin_parity_cs.png')); plt.close(fig)

# vuln: predicted vulnerability vs true post-contingency bus voltage (converged only)
VBt = d['t_vbus'][idx.numpy()][conv].ravel() if False else None
import numpy as _np
Vb = np.load('contingency_data.npz')['t_vbus'][idx.numpy()][conv].ravel()
fig, ax = plt.subplots(figsize=(4.3, 4.0)); ax.grid(alpha=0.3); ax.set_axisbelow(True)
ax.scatter(Vb, vln_p, s=4, alpha=0.15, color='#1f77b4', edgecolors='none')
ax.axvline(0.95, ls='--', color='gray', lw=1.2); ax.set_xlim(0.6, 1.05); ax.set_ylim(-0.02, 1.02)
ax.set_xlabel('True post-contingency bus voltage (p.u.)'); ax.set_ylabel('Predicted vulnerability')
ax.text(0.05, 0.5, f'ROC-AUC = {vuln_auc:.3f}\nPR-AUC = {vuln_prauc:.3f}', transform=ax.transAxes, va='center',
        bbox=dict(boxstyle='round', fc='white', ec='gray')); fig.tight_layout()
fig.savefig(os.path.join(OUT, 'vuln_cs.png')); plt.close(fig)

print(f"acc {acc*100:.2f}%  F1 {macro_f1*100:.2f}%  fs {false_safe*100:.3f}%  recall {[round(r*100,1) for r in recall]}")
print(f"confusion {cm.tolist()}  U->M {cm[2,1]}")
print(f"vmin MAE {vmae:.4f} R2 {r2:.3f}  vuln AUC {vuln_auc:.3f} PR {vuln_prauc:.3f}")
print(f"ECE {ece:.4f} MCE {mce:.3f} Brier {brier:.3f} NLL {nll:.3f} ROC {[round(x,3) for x in roc]} PR {[round(x,3) for x in prc]}")
print("wrote confusion_cs.png, vmin_parity_cs.png, vuln_cs.png + cs_results_full_a0.json")
