"""
Core model + training for the "make the full multi-task model beat risk-only" experiment
(Fable-5 plan). Self-contained: only needs torch + numpy. Inlined into the Colab notebook.

Key ideas:
1. alpha-scaled gradient routing. The auxiliary heads (V_min, vulnerability, divergence)
   train on trunk features whose GRADIENT into the trunk is multiplied by `alpha`. At
   alpha=0 the trunk trains identically to a risk-only model (aux heads become passive
   probes) => the full 4-output model provably matches risk-only risk accuracy.
   (v2 fix: aux pooling now uses its own gate `ga_a`, so alpha=0 is EXACTLY clean --
   previously aux losses still updated the shared pooling gate.)
2. aux-as-FEATURES stacking (`stack=True`). The risk label is a deterministic threshold
   of (V_min, divergence) in the data generator, so the aux heads' predictions -- trained
   on the strictly richer CONTINUOUS labels -- are fed (detached) into the risk head as
   explicit margin-to-threshold features. Positive transfer flows through features, not
   gradients, so the alpha=0 guarantee on the trunk is preserved.
3. dense per-bus voltage aux (`vbus=True`): a per-node regression head on t_vbus; the
   min over predicted bus voltages is a second, structurally-grounded V_min estimate
   that also enters the stack features.
4. post-hoc stacker (fit_stacker/evaluate_stacked): a tiny fs-gated logistic fusion of
   [risk logits, V_min margins, p_div, vuln stats] fitted on half the val split and
   accepted only if it improves the held-out other half at 0 false-safe.
Recipe: 150-epoch warmup-cosine, label smoothing, EMA, risk-gated checkpoint selection.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- GAT (identical to model_cs.py, inlined so the notebook needs no extra upload) ----
class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.15):
        super().__init__()
        self.heads, self.d = heads, out_dim // heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * self.d, 1, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.heads, self.d)
        hi = h.unsqueeze(2).expand(B, N, N, self.heads, self.d)
        hj = h.unsqueeze(1).expand(B, N, N, self.heads, self.d)
        e = F.leaky_relu(self.attn(torch.cat([hi, hj], -1)).squeeze(-1), 0.2)
        e = e.masked_fill((adj == 0).unsqueeze(-1), float('-inf'))
        a = self.drop(torch.nan_to_num(F.softmax(e, dim=2), 0))
        return (a.unsqueeze(-1) * hj).sum(2).reshape(B, N, -1)


# ---- alpha-scaled gradient routing ----
class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.alpha, None


def gs(x, alpha):
    return GradScale.apply(x, alpha) if alpha < 1.0 else x


class CSGNNv2(nn.Module):
    """Multi-task contingency-screening GNN with alpha-routing + optional branched risk trunk.
    Aux heads output LOGITS (BCEWithLogits) so mixed precision is safe."""
    def __init__(self, node_dim=4, edge_dim=3, hidden=128, heads=4, dropout=0.15,
                 n_gat=3, branched=True, uncert=False, stack=False, vbus=False):
        super().__init__()
        # base-topology edge mask (round-6 #1): 1 at physical-branch cells, 0 at non-branch
        # pairs, so padding cells never contribute through the encoder bias. Set via
        # set_base_mask() once the topology (N, edge_pairs) is known. Non-persistent so it
        # is rebuilt from data on load rather than baked into the checkpoint.
        self.register_buffer('base_mask', None, persistent=False)
        self.branched, self.uncert, self.stack, self.vbus = branched, uncert, stack, vbus
        self.node_enc = nn.Sequential(nn.Linear(node_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.edge_enc = nn.Sequential(nn.Linear(edge_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        n_sh = (n_gat - 1) if branched else n_gat
        self.gats = nn.ModuleList([GAT(hidden, hidden, heads, dropout) for _ in range(n_sh)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_sh)])
        if branched:
            self.gat_r = GAT(hidden, hidden, heads, dropout)
            self.norm_r = nn.LayerNorm(hidden)
            self.ga_r = nn.Linear(hidden, 1)
        self.ga = nn.Linear(hidden, 1)        # risk pooling gate
        self.ga_a = nn.Linear(hidden, 1)      # SEPARATE aux pooling gate: alpha=0 is exactly clean
        rd = hidden * 3
        n_sf = (9 if vbus else 6) if stack else 0   # stacked aux-prediction features into risk head
        self.risk_h = nn.Sequential(nn.Linear(rd + n_sf, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 3))
        self.vmin_h = nn.Sequential(nn.Linear(rd, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.vuln_h = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1))   # logits
        self.div_h = nn.Sequential(nn.Linear(rd, 64), nn.GELU(), nn.Linear(64, 1))         # logits
        if vbus:
            self.vbus_h = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1))  # per-bus V (pu)
        if uncert:
            self.log_s = nn.Parameter(torch.zeros(4))

    def set_base_mask(self, mask):
        """Fixed (N,N) base-topology mask: 1 at physical-branch cells (both directions,
        tripped branches INCLUDED, no self-loops), 0 elsewhere. Excludes non-branch/self
        pairs from the edge-feature sum so padding cells contribute exactly nothing
        (round-6 #1). Idempotent; re-registers so device moves carry it."""
        self.register_buffer('base_mask', mask.float(), persistent=False)

    def _pool(self, h, ga):
        a = torch.softmax(ga(h).squeeze(-1), 1).unsqueeze(-1)
        return torch.cat([h.mean(1), h.max(1).values, (h * a).sum(1)], -1)

    def forward(self, nf, ed, adj, alpha=1.0, beta=1.0):
        em = self.edge_enc(ed)
        if self.base_mask is not None:                       # zero non-branch pairs (round-6 #1)
            em = em * self.base_mask.unsqueeze(0).unsqueeze(-1)
        h = self.fuse(torch.cat([self.node_enc(nf), em.sum(2)], -1))
        for g, n in zip(self.gats, self.norms):
            h = n(h + g(h, adj))
        # ---- aux heads first (on alpha-scaled features + their own pooling gate) ----
        ha = gs(h, alpha)
        pa = self._pool(ha, self.ga_a)
        vmin = self.vmin_h(pa).squeeze(-1)
        vuln = self.vuln_h(ha).squeeze(-1)
        div = self.div_h(pa).squeeze(-1)
        vb = (self.vbus_h(ha).squeeze(-1) + 1.0) if self.vbus else None   # predict around 1.0 pu
        # ---- risk head, optionally consuming DETACHED aux predictions as features ----
        if self.branched:
            hr = self.norm_r(h + self.gat_r(h, adj))
            pr = self._pool(hr, self.ga_r)
        else:
            pr = self._pool(h, self.ga)
        if self.stack:
            feats = [vmin, (vmin - 0.95) * 20., (vmin - 0.90) * 20., torch.sigmoid(div),
                     torch.sigmoid(vuln).mean(1), torch.sigmoid(vuln).amax(1)]
            if vb is not None:
                vbm = vb.amin(1)
                feats += [vbm, (vbm - 0.95) * 20., (vbm - 0.90) * 20.]
            sf = torch.stack(feats, -1).detach() * beta     # detached: no risk->aux/trunk gradient
            risk = self.risk_h(torch.cat([pr, sf], -1))
        else:
            risk = self.risk_h(pr)
        return risk, vmin, vuln, div, vb


def build_data(npz_path, split_path, dev):
    """Load data, apply train-only normalisation from the grouped split, precompute dense tensors."""
    D = np.load(npz_path); S = np.load(split_path)
    N = int(D['n_bus']); pairs = torch.tensor(D['edge_pairs'], dtype=torch.long)
    fr, to = pairs[:, 0], pairs[:, 1]
    NF = ((torch.tensor(D['node_feats']) - torch.tensor(S['nmean'])) / torch.tensor(S['nstd'])).float().to(dev)
    EF = ((torch.tensor(D['edge_feats']) - torch.tensor(S['emean'])) / torch.tensor(S['estd'])).float()
    INS = torch.tensor(D['in_service']).float()
    M = NF.shape[0]
    DENSE = torch.zeros(M, N, N, 3); ADJ = torch.zeros(M, N, N)
    DENSE[:, fr, to, :2] = EF; DENSE[:, to, fr, :2] = EF
    DENSE[:, fr, to, 2] = INS; DENSE[:, to, fr, 2] = INS
    ADJ[:, fr, to] = INS; ADJ[:, to, fr] = INS
    ADJ[:, torch.arange(N), torch.arange(N)] = 1.0
    BASE = torch.zeros(N, N)                      # base physical-branch mask (round-6 #1)
    BASE[fr, to] = 1.0; BASE[to, fr] = 1.0        # both directions, no self-loops, tripped branches kept
    d = dict(
        NF=NF, DENSE=DENSE.to(dev), ADJ=ADJ.to(dev), BASE=BASE.to(dev),
        Tr=torch.tensor(D['t_risk']).to(dev), Tv=torch.tensor(D['t_vmin']).float().to(dev),
        Tu=torch.tensor(D['t_vuln']).float().to(dev), Td=torch.tensor(D['t_div']).float().to(dev),
        Tb=torch.tensor(D['t_vbus']).float().to(dev) if 't_vbus' in D else None,
        idx_tr=torch.tensor(S['idx_tr'].astype('int64')).to(dev),
        idx_va=torch.tensor(S['idx_va'].astype('int64')).to(dev),
        idx_te=torch.tensor(S['idx_te'].astype('int64')).to(dev), dev=dev,
    )
    cnt = torch.bincount(d['Tr'][d['idx_tr']], minlength=3).float()
    d['cw'] = (cnt.sum() / (3 * cnt)).to(dev)
    return d


@torch.no_grad()
def evaluate(net, d, idx, amp=False):
    net.eval()
    P, VMp, VMt, Dp, Dt = [], [], [], [], []
    for s0 in range(0, len(idx), 4096):
        b = idx[s0:s0 + 4096]
        with torch.autocast('cuda', enabled=amp):
            r, vm, vu, dv, vb = net(d['NF'][b], d['DENSE'][b], d['ADJ'][b])
        P.append(r.argmax(1)); conv = d['Td'][b] == 0
        VMp.append(vm[conv].float()); VMt.append(d['Tv'][b][conv])
        Dp.append((torch.sigmoid(dv) > 0.5).float()); Dt.append(d['Td'][b])
    P = torch.cat(P); R = d['Tr'][idx]
    acc = (P == R).float().mean().item()
    um = (R == 2)
    fs = (((R == 2) & (P == 0)).sum() / um.sum().clamp(min=1)).item()
    vmae = (torch.cat(VMp) - torch.cat(VMt)).abs().mean().item() if VMp else float('nan')
    dacc = (torch.cat(Dp) == torch.cat(Dt)).float().mean().item()
    return dict(acc=acc, false_safe=fs, vmin_mae=vmae, div_acc=dacc)


@torch.no_grad()
def divergence_report(net, d, idx, amp=False):
    """Round-6 #4: full nonconvergence-head metrics on one checkpoint (positive class = PF
    nonconvergence, Td==1). Precision/recall/F1 + confusion counts in torch; ROC-AUC/PR-AUC
    via sklearn if available. false_nonconvergent = predicted-diverge-but-converged (FP);
    false_convergent = predicted-converge-but-diverged (FN, the safety-relevant miss)."""
    net.eval(); Pr, Yt = [], []
    for s0 in range(0, len(idx), 4096):
        b = idx[s0:s0 + 4096]
        with torch.autocast('cuda', enabled=amp):
            _, _, _, dv, _ = net(d['NF'][b], d['DENSE'][b], d['ADJ'][b])
        Pr.append(torch.sigmoid(dv).float()); Yt.append(d['Td'][b].float())
    p = torch.cat(Pr); y = torch.cat(Yt); pred = (p > 0.5).float()
    tp = float(((pred == 1) & (y == 1)).sum()); fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum()); tn = float(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float('nan')
    rec = tp / (tp + fn) if tp + fn else float('nan')
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float('nan')
    out = dict(acc=(tp + tn) / max(1.0, tp + tn + fp + fn), precision=prec, recall=rec, f1=f1,
               tp=tp, fp=fp, fn=fn, tn=tn, n_pos=int(tp + fn), n_neg=int(fp + tn),
               false_nonconvergent=int(fp), false_convergent=int(fn),
               roc_auc=float('nan'), pr_auc=float('nan'))
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        yn, pn = y.cpu().numpy(), p.cpu().numpy()
        if yn.min() != yn.max():                 # both classes present
            out['roc_auc'] = float(roc_auc_score(yn, pn))
            out['pr_auc'] = float(average_precision_score(yn, pn))
    except Exception:
        pass
    return out


def train_eval(d, seed=0, alpha=0.1, branched=True, uncert=False, zero_aux=False,
               stack=False, vbus=False,
               hidden=128, heads=4, n_gat=3, epochs=150, bs=1024, lr=3e-3, ls=0.05,
               amp=True, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = d['dev']
    amp = amp and dev == 'cuda'
    vbus = vbus and d.get('Tb') is not None
    m = CSGNNv2(hidden=hidden, heads=heads, n_gat=n_gat, branched=branched, uncert=uncert,
                stack=stack, vbus=vbus).to(dev)
    m.set_base_mask(d['BASE'])                    # round-6 #1: mask non-branch edge cells
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    warm = max(1, epochs // 30)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: min((e + 1) / warm, 1.0) * 0.5 * (1 + math.cos(math.pi * max(0, e - warm) / max(1, epochs - warm))))
    ema = torch.optim.swa_utils.AveragedModel(
        m, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    idx_tr = d['idx_tr']
    best, best_state = -1.0, None

    stack_warm = max(1, epochs // 15)   # let aux heads converge before the risk head trusts them
    for ep in range(epochs):
        m.train()
        beta = min(1.0, (ep + 1) / stack_warm) if stack else 1.0
        perm = idx_tr[torch.randperm(len(idx_tr), device=dev)]
        for s0 in range(0, len(perm), bs):
            b = perm[s0:s0 + bs]
            with torch.autocast('cuda', enabled=amp):
                r, vm, vu, dv, vb = m(d['NF'][b], d['DENSE'][b], d['ADJ'][b], alpha=alpha, beta=beta)
                conv = d['Td'][b] == 0
                lr_ = F.cross_entropy(r, d['Tr'][b], weight=d['cw'], label_smoothing=ls)
                lv = F.smooth_l1_loss(vm[conv], d['Tv'][b][conv]) if conv.any() else r.sum() * 0
                # round-6 #5: vulnerability is undefined on nonconvergent cases (no solved V);
                # supervise it on converged scenarios only (like V_min / V_bus above).
                lvu = (F.binary_cross_entropy_with_logits(vu[conv], d['Tu'][b][conv])
                       if conv.any() else r.sum() * 0)
                ld = F.binary_cross_entropy_with_logits(dv, d['Td'][b])   # nonconvergence: ALL cases
                lvb = (F.smooth_l1_loss(vb[conv], d['Tb'][b][conv])
                       if (vbus and conv.any()) else r.sum() * 0)
                if zero_aux:
                    loss = lr_
                elif uncert:
                    s = m.log_s
                    loss = (torch.exp(-s[0]) * lr_ + 0.5 * torch.exp(-s[1]) * lv
                            + torch.exp(-s[2]) * lvu + torch.exp(-s[3]) * ld + 0.5 * s.sum())
                else:
                    loss = lr_ + 0.5 * lv + 0.3 * lvu + 0.3 * ld + 0.3 * lvb
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0)
            scaler.step(opt); scaler.update()
            ema.update_parameters(m)
        sch.step()
        if ep >= epochs // 3 and (ep % 5 == 0 or ep == epochs - 1):
            # select on the better of the raw and EMA weights (EMA lags on short runs), gated on 0 false-safe
            for cand in (m, ema.module):
                r = evaluate(cand, d, d['idx_va'], amp=amp)
                if r['false_safe'] == 0.0 and r['acc'] > best:
                    best = r['acc']; best_state = {k: v.clone() for k, v in cand.state_dict().items()}
            if verbose:
                print(f"  ep{ep} val acc(raw) {evaluate(m, d, d['idx_va'], amp=amp)['acc']*100:.2f}% best@fs0 {best*100:.2f}%")
    if best_state is None:
        # no zero-false-safe checkpoint yet (e.g. a very short run): take whichever weights give best val acc
        best_state = max(((evaluate(c, d, d['idx_va'], amp=amp)['acc'], c) for c in (m, ema.module)),
                         key=lambda t: t[0])[1].state_dict()
        best_state = {k: v.clone() for k, v in best_state.items()}
    ema.module.load_state_dict(best_state)
    te = evaluate(ema.module, d, d['idx_te'], amp=amp)
    te['params'] = sum(p.numel() for p in m.parameters())
    return te, ema.module


# ---- post-hoc fs-gated logistic fusion of risk logits with the aux predictions ----
@torch.no_grad()
def collect_feats(net, d, idx, amp=False, logits_only=False):
    """Per-sample fusion features: [risk logits(3)] (+ [vmin, margins, p_div, vuln stats, vbus_min...])."""
    net.eval(); out = []
    for s0 in range(0, len(idx), 4096):
        b = idx[s0:s0 + 4096]
        with torch.autocast('cuda', enabled=amp):
            r, vm, vu, dv, vb = net(d['NF'][b], d['DENSE'][b], d['ADJ'][b])
        f = [r.float()]
        if not logits_only:
            vm = vm.float()
            f += [vm.unsqueeze(-1), ((vm - 0.95) * 20.).unsqueeze(-1), ((vm - 0.90) * 20.).unsqueeze(-1),
                  torch.sigmoid(dv).float().unsqueeze(-1),
                  torch.sigmoid(vu).float().mean(1, keepdim=True), torch.sigmoid(vu).float().amax(1, keepdim=True)]
            if vb is not None:
                vbm = vb.float().amin(1)
                f += [vbm.unsqueeze(-1), ((vbm - 0.95) * 20.).unsqueeze(-1), ((vbm - 0.90) * 20.).unsqueeze(-1)]
        out.append(torch.cat(f, -1))
    return torch.cat(out)


def _acc_fs(pred, y):
    acc = (pred == y).float().mean().item()
    um = (y == 2)
    fs = (((y == 2) & (pred == 0)).sum() / um.sum().clamp(min=1)).item()
    return acc, fs


def fit_stacker(net, d, amp=False, iters=400, lr=0.05, seed=0, logits_only=False):
    """Tiny linear fusion head. Fitted on HALF of val; accepted only if it improves risk
    accuracy on the other (held-out) half of val at 0 false-safe. Initialised at identity
    on the logit block, so rejecting it exactly recovers the raw model. For the risk-only
    reference use logits_only=True (plain affine recalibration -- fairness control)."""
    idx = d['idx_va']
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(idx), generator=g).to(idx.device)
    fit, sel = idx[perm[:len(idx) // 2]], idx[perm[len(idx) // 2:]]
    Xf = collect_feats(net, d, fit, amp, logits_only); yf = d['Tr'][fit]
    Xs = collect_feats(net, d, sel, amp, logits_only); ys = d['Tr'][sel]
    mu = Xf.mean(0); sd = Xf.std(0).clamp(min=1e-4)
    mu[:3] = 0.; sd[:3] = 1.                        # keep logits unscaled for identity init
    W = torch.zeros(Xf.shape[1], 3, device=Xf.device)
    W[:3] = torch.eye(3, device=Xf.device)
    W.requires_grad_(True); b = torch.zeros(3, device=Xf.device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    Zf = (Xf - mu) / sd
    for _ in range(iters):
        loss = F.cross_entropy(Zf @ W + b, yf, weight=d['cw']) + 1e-3 * (W[3:] ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        a_st, fs_st = _acc_fs(((Xs - mu) / sd @ W + b).argmax(1), ys)
        a_raw, _ = _acc_fs(Xs[:, :3].argmax(1), ys)
    st = dict(W=W.detach(), b=b.detach(), mu=mu, sd=sd, logits_only=logits_only,
              accepted=bool(fs_st == 0.0 and a_st > a_raw), val_acc=a_st, val_raw=a_raw)
    return st


@torch.no_grad()
def evaluate_stacked(net, d, idx, st, amp=False):
    """Test metrics with the fusion head; falls back to raw argmax if the stacker was rejected."""
    X = collect_feats(net, d, idx, amp, st['logits_only']); y = d['Tr'][idx]
    pred = ((X - st['mu']) / st['sd'] @ st['W'] + st['b']).argmax(1) if st['accepted'] else X[:, :3].argmax(1)
    acc, fs = _acc_fs(pred, y)
    return dict(acc=acc, false_safe=fs, stacker_used=st['accepted'])
