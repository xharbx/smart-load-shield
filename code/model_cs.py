"""
Contingency-screening GNN (leakage-free reformulation).

Inputs : pre-contingency node state [B,N,4], dense edge features [B,N,N,3]
         (loading, P_from, in-service), and PER-SAMPLE post-contingency
         adjacency [B,N,N] (tripped branches removed + self-loops).
Outputs: post-contingency risk class (3), post V_min (regression),
         per-bus vulnerability (sigmoid N), divergence flag (sigmoid).

The post-contingency voltages are NOT in the input; the model propagates the
pre-contingency state over the NEW topology to predict the outcome.
"""
import torch, torch.nn as nn, torch.nn.functional as F


class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.15):
        super().__init__()
        self.heads, self.d = heads, out_dim // heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * self.d, 1, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):                       # x:[B,N,D]  adj:[B,N,N]
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.heads, self.d)
        hi = h.unsqueeze(2).expand(B, N, N, self.heads, self.d)
        hj = h.unsqueeze(1).expand(B, N, N, self.heads, self.d)
        e = F.leaky_relu(self.attn(torch.cat([hi, hj], -1)).squeeze(-1), 0.2)   # [B,N,N,heads]
        e = e.masked_fill((adj == 0).unsqueeze(-1), float('-inf'))
        a = self.drop(torch.nan_to_num(F.softmax(e, dim=2), 0))
        return (a.unsqueeze(-1) * hj).sum(2).reshape(B, N, -1)


class ContingencyGNN(nn.Module):
    def __init__(self, node_dim=4, edge_dim=3, hidden=128, heads=4, n_gat=3, dropout=0.15, use_edge=True):
        super().__init__()
        self.use_edge = use_edge
        self.node_enc = nn.Sequential(nn.Linear(node_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.edge_enc = nn.Sequential(nn.Linear(edge_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.gats = nn.ModuleList([GAT(hidden, hidden, heads, dropout) for _ in range(n_gat)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_gat)])
        self.ga = nn.Linear(hidden, 1)
        rd = hidden * 3
        self.risk_h = nn.Sequential(nn.Linear(rd, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 3))
        self.vmin_h = nn.Sequential(nn.Linear(rd, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.vuln_h = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())
        self.div_h = nn.Sequential(nn.Linear(rd, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, nf, ed, adj):
        h = self.node_enc(nf)
        ec = self.edge_enc(ed).sum(2) if self.use_edge else torch.zeros_like(h)
        h = self.fuse(torch.cat([h, ec], -1))
        for g, n in zip(self.gats, self.norms):
            h = n(h + g(h, adj))
        hm, hx = h.mean(1), h.max(1).values
        ha = (h * torch.softmax(self.ga(h).squeeze(-1), 1).unsqueeze(-1)).sum(1)
        pooled = torch.cat([hm, hx, ha], -1)
        risk = self.risk_h(pooled)
        vmin = self.vmin_h(pooled).squeeze(-1)
        vuln = self.vuln_h(h).squeeze(-1)
        div = self.div_h(pooled).squeeze(-1)
        return risk, vmin, vuln, div
