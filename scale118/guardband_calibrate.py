"""
Calibrate the overvoltage guard band on VALIDATION, then report it on TEST.

The guard band escalates any contingency whose PREDICTED V_max exceeds
1.05 - delta to the exact solver. An earlier version of this analysis picked
delta by looking at the test set, which is selection on test and would not
survive review.

delta is now set by SPLIT-CONFORMAL calibration: it is the (1 - alpha) quantile
of the V_max under-prediction residual on the held-out VALIDATION split, with
alpha = 0.001. Clearing only when predicted V_max + delta < 1.05 therefore
controls the probability that a true V_max crosses the limit unseen. It is a
stated rule applied to held-out data, not a tuned number, and no test data
enters the choice. The test split is scored once at that fixed delta.

    python guardband_calibrate.py
"""
import json
import sys

import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train118 import build_data, batch
from core118 import CSGNN118

# Directory holding contingency_data_118_v2.npz and grouped_split_118_v2.npz.
# Defaults to this file's own directory; override with SLS118_DIR to point at
# a corpus generated elsewhere (a mounted Drive, a scratch disk).
DR = os.environ.get('SLS118_DIR', os.path.dirname(os.path.abspath(__file__)))
V_HI = 1.05

d = build_data(DR + r'\contingency_data_118_v2.npz',
               DR + r'\grouped_split_118_v2.npz', dev='cpu')
raw = np.load(DR + r'\contingency_data_118_v2.npz')
vb, div, risk = raw['t_vbus'], raw['t_div'], raw['t_risk']
conv = div < 0.5
tvmax = np.where(conv, vb.max(axis=1), 0.0)


@torch.no_grad()
def run(model, idx, bs=512):
    p, vx = [], []
    for i in range(0, len(idx), bs):
        nf, cf, adj = batch(d, idx[i:i + bs])
        o = model(nf, cf, adj)
        p.append(torch.softmax(o[0].float(), 1).numpy())
        vx.append(o[5].float().numpy())
    return np.concatenate(p), np.concatenate(vx)


models = []
for s in range(5):
    m = CSGNN118(hidden=128, heads=4, n_gat=3, branched=False, stack=False,
                 vbus=False, vmax=True)
    m.set_graph(d['ctx'])
    m.load_state_dict(torch.load(DR + ('/model_118_v2_vmax_seed%d.pt' % s),
                                 map_location='cpu'))
    m.eval()
    models.append(m)


def ensemble(idx):
    P, X = [], []
    for m in models:
        p, x = run(m, idx)
        P.append(p); X.append(x)
    return np.stack(P).mean(0), np.stack(X).mean(0)


def score(idx, delta):
    prob, vx = CACHE[id(idx)]
    pred = prob.argmax(1)
    y = risk[idx]
    esc = vx >= (V_HI - delta)
    eff = np.where(esc, 2, pred)              # escalated -> not cleared
    fs = int(((y == 2) & (eff == 0)).sum())
    return fs, float(esc.mean()), int((y == 2).sum())


va, te = d['idx_va'], d['idx_te']
CACHE = {id(va): ensemble(va), id(te): ensemble(te)}

# SPLIT-CONFORMAL CALIBRATION.
# delta is not tuned. It is the (1 - alpha) quantile of the V_max
# UNDER-PREDICTION residual measured on the held-out validation split, so that
# clearing only when predicted V_max + delta < 1.05 controls the probability
# that the true V_max crosses the limit unseen. Nothing about the test split
# enters the choice.
ALPHA = 0.001
prob_va, xva = CACHE[id(va)]
res = (tvmax[va] - xva)[conv[va]]
delta = float(np.percentile(res, 100 * (1 - ALPHA)))
print('calibrating on VALIDATION (n=%d, %d converged, %d unstable)'
      % (len(va), int(conv[va].sum()), int((risk[va] == 2).sum())))
print('  V_max under-prediction residual: p90 %+.5f  p99 %+.5f  p99.9 %+.5f  max %+.5f'
      % (np.percentile(res, 90), np.percentile(res, 99),
         np.percentile(res, 99.9), res.max()))
print('  delta = p%.1f quantile = %.4f p.u.   (alpha = %.3f)'
      % (100 * (1 - ALPHA), delta, ALPHA))
fs_va, esc_va, _ = score(va, delta)
print('  on validation itself: %d false clearances, %.1f%% escalated'
      % (fs_va, 100 * esc_va))

print()
print('applying that FIXED delta to TEST, scored once:')
fs, esc, n_un = score(te, delta)
print('  test unstable cases      : %d' % n_un)
print('  false clearances         : %d' % fs)
print('  escalated to exact solver: %.1f%% of the screened set' % (100 * esc))

print()
print('for reference, test behaviour across delta (NOT used for selection):')
for dl in (0.0, 0.002, 0.003, delta, 0.005, 0.006):
    f, e, _ = score(te, dl)
    tag = '   <-- calibrated on validation' if abs(dl - delta) < 1e-9 else ''
    print('  delta %.4f  false-safe %3d  escalation %5.1f%%%s' % (dl, f, 100 * e, tag))

json.dump(dict(delta=delta, alpha=ALPHA, test_false_safe=fs, test_escalation=esc,
               test_unstable=n_un, calibrated_on='validation split',
               rule='split-conformal: delta = (1-alpha) quantile of the V_max '
                    'under-prediction residual on validation',
               note='no test data enters the choice of delta; test scored once'),
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'guardband_118_v2.json'), 'w'),
          indent=2)
print('\nsaved guardband_118_v2.json')
