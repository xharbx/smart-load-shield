"""
Apple-to-apple GNN architecture baselines for contingency screening (review issue #7).

Same dataset, same grouped-by-operating-point split, same risk targets and class
weights as the deployed model -- only the graph architecture changes:

  GCN         : symmetric-normalised mean aggregation (Kipf & Welling)
  GraphSAGE   : mean-neighbour aggregation with self transform
  GAT (node)  : multi-head graph attention on node features only (no edge fusion)
  --- reference: our full model = GAT + edge-flow fusion (reported separately, 97.67%)

All consume node features [V,theta,P,Q] over the post-contingency adjacency A_c
(with self-loops); risk-only head; identical width/depth/optimiser/epochs/EMA and the
same zero-false-safe-gated checkpoint rule. Reports acc / macro-F1 / false-safe.

Run: python baselines_gnn.py            (full)
     python baselines_gnn.py 5          (smoke test: 5 epochs)
"""
import sys, os, json, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import boost_core as B

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
SEEDS = [0, 1, 2] if EPOCHS >= 150 else [0]
HID, HEADS, NGAT, BS, LR, WD, LS = 128, 4, 3, 1024, 3e-3, 1e-4, 0.05


# ---- conv layers (dense [B,N,N] adjacency, self-loops already on the diagonal) ----
class GCN(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__(); self.lin = nn.Linear(d_in, d_out)

    def forward(self, x, adj):
        deg = adj.sum(-1, keepdim=True).clamp(min=1)          # A already has self-loops
        an = adj / deg.sqrt() / deg.transpose(1, 2).sqrt()    # D^-1/2 A D^-1/2
        return an @ self.lin(x)


class SAGE(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__(); self.lw = nn.Linear(d_in, d_out); self.nw = nn.Linear(d_in, d_out)

    def forward(self, x, adj):
        deg = adj.sum(-1, keepdim=True).clamp(min=1)
        nb = (adj @ x) / deg                                  # mean of neighbours (incl self)
        return self.lw(x) + self.nw(nb)


class SimpleGNN(nn.Module):
    """Risk-only GNN with a swappable conv; node features + adjacency, mean pooling."""
    def __init__(self, conv, node_dim=4, hidden=HID):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(node_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        if conv == 'gat':
            self.convs = nn.ModuleList([B.GAT(hidden, hidden, HEADS, 0.15) for _ in range(NGAT)])
        elif conv == 'gcn':
            self.convs = nn.ModuleList([GCN(hidden, hidden) for _ in range(NGAT)])
        elif conv == 'sage':
            self.convs = nn.ModuleList([SAGE(hidden, hidden) for _ in range(NGAT)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(NGAT)])
        self.head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.GELU(), nn.Dropout(0.15), nn.Linear(64, 3))

    def forward(self, nf, adj):
        h = self.enc(nf)
        for c, n in zip(self.convs, self.norms):
            h = n(h + F.gelu(c(h, adj)))
        return self.head(torch.cat([h.mean(1), h.max(1).values], -1))


@torch.no_grad()
def evaluate(net, d, idx):
    net.eval(); P = []
    for s in range(0, len(idx), 4096):
        b = idx[s:s+4096]; P.append(net(d['NF'][b], d['ADJ'][b]).argmax(1))
    P = torch.cat(P); R = d['Tr'][idx]
    acc = (P == R).float().mean().item()
    um = (R == 2); fs = (((R == 2) & (P == 0)).sum() / um.sum().clamp(min=1)).item()
    cm = torch.zeros(3, 3, dtype=torch.long)
    for t, p in zip(R, P): cm[t, p] += 1
    f1 = []
    for k in range(3):
        tp = cm[k, k].item(); fp = cm[:, k].sum().item()-tp; fn = cm[k].sum().item()-tp
        pr = tp/max(1, tp+fp); rc = tp/max(1, tp+fn); f1.append(2*pr*rc/max(1e-9, pr+rc))
    return acc, fs, float(np.mean(f1))


def train_one(conv, d, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = SimpleGNN(conv).to(d['dev'])
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    warm = max(1, EPOCHS // 30)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: min((e+1)/warm, 1.0) *
          0.5*(1+math.cos(math.pi*max(0, e-warm)/max(1, EPOCHS-warm))))
    ema = torch.optim.swa_utils.AveragedModel(m, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    itr = d['idx_tr']; best, best_state = -1.0, None
    for ep in range(EPOCHS):
        m.train(); perm = itr[torch.randperm(len(itr), device=d['dev'])]
        for s in range(0, len(perm), BS):
            b = perm[s:s+BS]
            loss = F.cross_entropy(m(d['NF'][b], d['ADJ'][b]), d['Tr'][b], weight=d['cw'], label_smoothing=LS)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0); opt.step(); ema.update_parameters(m)
        sch.step()
        if ep >= EPOCHS // 3 and (ep % 5 == 0 or ep == EPOCHS-1):
            for cand in (m, ema.module):
                acc, fs, _ = evaluate(cand, d, d['idx_va'])
                if fs == 0.0 and acc > best:
                    best = acc; best_state = {k: v.clone() for k, v in cand.state_dict().items()}
    if best_state is None:
        best_state = max(((evaluate(c, d, d['idx_va'])[0], c) for c in (m, ema.module)),
                         key=lambda t: t[0])[1].state_dict()
        best_state = {k: v.clone() for k, v in best_state.items()}
    ema.module.load_state_dict(best_state)
    return evaluate(ema.module, d, d['idx_te'])


def main():
    d = B.build_data('contingency_data.npz', 'grouped_split.npz', DEV)
    print(f"device {DEV}  epochs {EPOCHS}  seeds {SEEDS}")
    res = {}
    for conv in ('gcn', 'sage', 'gat'):
        accs, fss, f1s = [], [], []
        for sd in SEEDS:
            a, fs, f1 = train_one(conv, d, sd); accs.append(a); fss.append(fs); f1s.append(f1)
            print(f"  {conv:5s} seed{sd}: acc {a*100:.2f}%  fs {fs*100:.3f}%  F1 {f1*100:.2f}%")
        res[conv] = dict(acc=float(np.mean(accs)), acc_std=float(np.std(accs)),
                         false_safe=float(np.mean(fss)), macro_f1=float(np.mean(f1s)),
                         acc_all=accs, fs_all=fss)
        print(f"{conv.upper():10s} acc {np.mean(accs)*100:.2f}+-{np.std(accs)*100:.2f}%  "
              f"fs {np.mean(fss)*100:.3f}%  F1 {np.mean(f1s)*100:.2f}%")
    json.dump(res, open(os.path.join(os.path.dirname(__file__), 'baselines_gnn_results.json'), 'w'), indent=2)
    print("saved baselines_gnn_results.json")


if __name__ == '__main__':
    main()
