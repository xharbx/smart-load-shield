"""
Memory-efficient core for scaling the contingency-screening surrogate to the
IEEE 118-bus system.

WHY THIS FILE EXISTS
--------------------
`boost_core.py` materialises, for the WHOLE dataset, a dense edge tensor
[M, N, N, 3] and a dense adjacency [M, N, N].  At N=27 that is 262 MB + 87 MB
and fits on any GPU.  At N=118 the same tensors are 5.0 GB + 1.7 GB, and the
attention layer's [B, N, N, heads, 2d] concatenation adds ~900 MB per layer at
batch 64.  A naive port therefore OOMs before the first step.

Three changes fix it, and all three are EXACTLY equivalent to the original --
this is the whole point, and `test_equivalence.py` proves it numerically:

  1. SPARSE EDGE ENCODING.  The original computes `edge_enc` on every one of the
     N^2 dense cells, multiplies by a base mask that zeroes non-branch cells,
     then sums over j.  Only branch cells survive, so we instead evaluate
     `edge_enc` on the ~186 real branch cells and `index_add` the result onto
     both endpoints.  Identical output, O(n_branch) instead of O(N^2) memory.

  2. FACTORISED ATTENTION.  The original builds [B,N,N,heads,2d] to compute
     `attn(cat([h_i, h_j]))`.  Because `attn` is linear and bias-free,
        attn([h_i; h_j]) = a_src . h_i + a_dst . h_j,
     with `a_src`/`a_dst` the two halves of the SAME weight matrix.  We compute
     two [B,N,heads] vectors and broadcast-add, then aggregate with an einsum
     that never materialises the product.  Identical output, O(N^2 * heads)
     instead of O(N^2 * 2d) memory -- a 16x reduction at heads=4, d=32.

  3. PER-BATCH DENSE ADJACENCY.  Built on the fly from the compact in-service
     vector instead of being precomputed for all M samples.

PARALLEL BRANCHES
-----------------
case118 has 7 pairs of buses joined by two branches each; the 26/27-bus case has
none.  The original dense code would silently let one parallel branch overwrite
the other (last write wins), which is both arbitrary and physically wrong.  Here
a graph EDGE is a unique bus pair ("cell") and parallel branches are aggregated
onto it: features are averaged, and the pair is adjacent if ANY of its branches
is in service.  On a network without parallel branches every cell holds exactly
one branch, so this reduces EXACTLY to the original behaviour -- which is why
the 27-bus equivalence test is a valid gate for the 118-bus code path.

Contingencies are still defined per BRANCH (186 of them), so tripping one of two
parallel circuits is a genuinely milder, and correctly represented, event.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Graph context: branch list -> unique undirected cells
# ============================================================================
def build_graph_ctx(edge_pairs, n_bus):
    """
    edge_pairs : [n_branch, 2] int array of (from_node, to_node) INDEX pairs
                 (already 0..n_bus-1, i.e. positions in the node ordering).

    Returns a dict with
      cell_of_branch : [n_branch] long, which cell each branch belongs to
      cell_i, cell_j : [n_cell] long, the two endpoints of each cell
      branch_per_cell: [n_cell] float, how many branches share the cell
      n_cell         : int
    Cells are emitted in order of first appearance so that, on a network with no
    parallel branches, cell k IS branch k and the mapping is the identity.
    """
    ep = np.asarray(edge_pairs).astype(np.int64)
    order, cell_of_branch = {}, np.zeros(len(ep), np.int64)
    ci, cj = [], []
    for k, (f, t) in enumerate(ep):
        key = (int(min(f, t)), int(max(f, t)))
        if key not in order:
            order[key] = len(ci)
            ci.append(key[0])
            cj.append(key[1])
        cell_of_branch[k] = order[key]
    n_cell = len(ci)
    cnt = np.zeros(n_cell, np.float32)
    for c in cell_of_branch:
        cnt[c] += 1.0
    return dict(
        cell_of_branch=torch.tensor(cell_of_branch, dtype=torch.long),
        cell_i=torch.tensor(ci, dtype=torch.long),
        cell_j=torch.tensor(cj, dtype=torch.long),
        branch_per_cell=torch.tensor(cnt, dtype=torch.float32),
        n_cell=n_cell,
        n_bus=int(n_bus),
        n_branch=len(ep),
    )


def cells_and_adj(ef, ins, ctx):
    """
    Compact per-branch tensors -> per-cell features and dense adjacency.

    ef  : [B, n_branch, 2]  base-case (loading, P_from) per branch
    ins : [B, n_branch]     in-service flag per branch (the contingency)

    Returns
      cell_feat : [B, n_cell, 3]  averaged (loading, P_from, in-service) per cell
      adj       : [B, N, N]       1 where a cell has ANY branch in service, plus
                                  self-loops -- matching the original construction
    """
    B = ef.shape[0]
    dev = ef.device
    cob = ctx['cell_of_branch'].to(dev)
    ci, cj = ctx['cell_i'].to(dev), ctx['cell_j'].to(dev)
    cnt = ctx['branch_per_cell'].to(dev)
    N, n_cell = ctx['n_bus'], ctx['n_cell']

    feat = torch.cat([ef, ins.unsqueeze(-1)], -1)                  # [B,n_branch,3]
    csum = torch.zeros(B, n_cell, 3, device=dev, dtype=feat.dtype)
    csum.index_add_(1, cob, feat)
    cell_feat = csum / cnt.view(1, -1, 1)                          # mean over parallels

    isum = torch.zeros(B, n_cell, device=dev, dtype=ins.dtype)
    isum.index_add_(1, cob, ins)
    live = (isum > 0).to(ef.dtype)                                 # OR over parallels

    adj = torch.zeros(B, N, N, device=dev, dtype=ef.dtype)
    adj[:, ci, cj] = live
    adj[:, cj, ci] = live
    adj[:, torch.arange(N, device=dev), torch.arange(N, device=dev)] = 1.0
    return cell_feat, adj


def dense_from_compact(ef, ins, edge_pairs, n_bus):
    """
    Reference implementation: reproduce boost_core's dense tensors exactly.
    Used only by the equivalence test and by the 27-bus reference path.
    """
    B = ef.shape[0]
    dev = ef.device
    pairs = torch.as_tensor(np.asarray(edge_pairs).astype(np.int64), device=dev)
    fr, to = pairs[:, 0], pairs[:, 1]
    N = n_bus
    dense = torch.zeros(B, N, N, 3, device=dev, dtype=ef.dtype)
    adj = torch.zeros(B, N, N, device=dev, dtype=ef.dtype)
    dense[:, fr, to, :2] = ef
    dense[:, to, fr, :2] = ef
    dense[:, fr, to, 2] = ins
    dense[:, to, fr, 2] = ins
    adj[:, fr, to] = ins
    adj[:, to, fr] = ins
    adj[:, torch.arange(N, device=dev), torch.arange(N, device=dev)] = 1.0
    base = torch.zeros(N, N, device=dev, dtype=ef.dtype)
    base[fr, to] = 1.0
    base[to, fr] = 1.0
    return dense, adj, base


# ============================================================================
# Attention
# ============================================================================
class GATFast(nn.Module):
    """
    Parameter-identical to boost_core.GAT (`W`, `attn`), so state dicts are
    interchangeable, but computed without the [B,N,N,heads,2d] concatenation.
    """

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.15):
        super().__init__()
        self.heads, self.d = heads, out_dim // heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * self.d, 1, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):                      # x:[B,N,D]  adj:[B,N,N]
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.heads, self.d)
        w = self.attn.weight.view(-1)                                   # [2d]
        a_src, a_dst = w[: self.d], w[self.d:]
        es = torch.einsum('bnhd,d->bnh', h, a_src)                      # a_src . h_i
        et = torch.einsum('bnhd,d->bnh', h, a_dst)                      # a_dst . h_j
        e = F.leaky_relu(es.unsqueeze(2) + et.unsqueeze(1), 0.2)        # [B,N,N,heads]
        e = e.masked_fill((adj == 0).unsqueeze(-1), float('-inf'))
        a = self.drop(torch.nan_to_num(F.softmax(e, dim=2), 0))
        return torch.einsum('bijh,bjhd->bihd', a, h).reshape(B, N, -1)


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


# ============================================================================
# Model
# ============================================================================
class CSGNN118(nn.Module):
    """
    Parameter-identical to boost_core.CSGNNv2; only the edge encoding and the
    attention arithmetic differ.  `set_graph()` replaces `set_base_mask()`.
    """

    def __init__(self, node_dim=4, edge_dim=3, hidden=128, heads=4, dropout=0.15,
                 n_gat=3, branched=True, uncert=False, stack=False, vbus=False,
                 vmax=False):
        """
        vmax: add a V_max regression head mirroring the V_min head.

        Every other auxiliary head is undervoltage-only (V_min, 1[V_i<0.95],
        divergence), which matched Eq. (risk) while the labelling code was
        one-sided.  With the two-sided rule enforced, a contingency can be
        Unstable through V_max >= 1.05 while its V_min is perfectly healthy, and
        nothing in the network represents that: on the 118-bus corpus 82 of 83
        false-safe cases were exactly this.  This head supplies the missing
        signal.  It is REGRESSION, so it trains on every converged scenario
        rather than on the handful of overvoltage positives.

        Off by default: the 27-bus corpus never exceeds 1.045 p.u., so the head
        would supervise a constant there, and leaving it off keeps the deployed
        configuration and test_equivalence.py byte-identical.
        """
        super().__init__()
        self.branched, self.uncert, self.stack, self.vbus = branched, uncert, stack, vbus
        self.vmax = vmax
        self.ctx = None
        self.node_enc = nn.Sequential(nn.Linear(node_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.edge_enc = nn.Sequential(nn.Linear(edge_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        n_sh = (n_gat - 1) if branched else n_gat
        self.gats = nn.ModuleList([GATFast(hidden, hidden, heads, dropout) for _ in range(n_sh)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_sh)])
        if branched:
            self.gat_r = GATFast(hidden, hidden, heads, dropout)
            self.norm_r = nn.LayerNorm(hidden)
            self.ga_r = nn.Linear(hidden, 1)
        self.ga = nn.Linear(hidden, 1)
        self.ga_a = nn.Linear(hidden, 1)
        rd = hidden * 3
        n_sf = ((9 if vbus else 6) + (2 if vmax else 0)) if stack else 0
        self.risk_h = nn.Sequential(nn.Linear(rd + n_sf, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 3))
        self.vmin_h = nn.Sequential(nn.Linear(rd, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        if vmax:
            self.vmax_h = nn.Sequential(nn.Linear(rd, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.vuln_h = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1))
        self.div_h = nn.Sequential(nn.Linear(rd, 64), nn.GELU(), nn.Linear(64, 1))
        if vbus:
            self.vbus_h = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1))
        if uncert:
            self.log_s = nn.Parameter(torch.zeros(4))

    def set_graph(self, ctx):
        """Attach the graph context (cell mapping). Not part of the state dict."""
        self.ctx = ctx

    def _pool(self, h, ga):
        a = torch.softmax(ga(h).squeeze(-1), 1).unsqueeze(-1)
        return torch.cat([h.mean(1), h.max(1).values, (h * a).sum(1)], -1)

    def _edge_context(self, cell_feat):
        """edge_enc on real branch cells only, scattered onto both endpoints."""
        B = cell_feat.shape[0]
        dev = cell_feat.device
        ci, cj = self.ctx['cell_i'].to(dev), self.ctx['cell_j'].to(dev)
        em = self.edge_enc(cell_feat)                                   # [B,n_cell,H]
        ec = torch.zeros(B, self.ctx['n_bus'], em.shape[-1], device=dev, dtype=em.dtype)
        ec.index_add_(1, ci, em)
        ec.index_add_(1, cj, em)
        return ec

    def forward(self, nf, cell_feat, adj, alpha=1.0, beta=1.0):
        h = self.fuse(torch.cat([self.node_enc(nf), self._edge_context(cell_feat)], -1))
        for g, n in zip(self.gats, self.norms):
            h = n(h + g(h, adj))
        ha = gs(h, alpha)
        pa = self._pool(ha, self.ga_a)
        vmin = self.vmin_h(pa).squeeze(-1)
        vmx = self.vmax_h(pa).squeeze(-1) if self.vmax else None
        vuln = self.vuln_h(ha).squeeze(-1)
        div = self.div_h(pa).squeeze(-1)
        vb = (self.vbus_h(ha).squeeze(-1) + 1.0) if self.vbus else None
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
            if vmx is not None:
                feats += [vmx, (vmx - 1.05) * 20.]
            sf = torch.stack(feats, -1).detach() * beta
            risk = self.risk_h(torch.cat([pr, sf], -1))
        else:
            risk = self.risk_h(pr)
        # Variable arity is deliberate: with vmax off the return is byte-identical
        # to the deployed configuration, so every existing caller and
        # test_equivalence.py keep working untouched.
        if self.vmax:
            return risk, vmin, vuln, div, vb, vmx
        return risk, vmin, vuln, div, vb
