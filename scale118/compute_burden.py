"""
R3 #5: quantitative computational burden, both systems, like for like.

The reviewer asked for exact training time, hardware, memory usage, and the
SEPARATE costs of dataset generation, model training and inference. This script
measures everything that can be measured on one machine so the two systems are
directly comparable; training time and peak GPU memory come from a short Colab
cell (colab_burden_cell.py) since neither system trains on this laptop.

Everything here is single-threaded on the same CPU, so the 27-bus and 118-bus
numbers are comparable to each other and to the paper's existing 40x screening
measurement, which used the same protocol.
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# the 27-bus reference implementation, corpus, split and checkpoint
CODE = os.path.abspath(os.path.join(HERE, '..', 'code'))
sys.path.insert(0, HERE)
sys.path.insert(0, CODE)

# Directory holding contingency_data_118_v2.npz and grouped_split_118_v2.npz.
# Defaults to this file's own directory; override with SLS118_DIR to point at
# a corpus generated elsewhere (a mounted Drive, a scratch disk).
DRIVE = os.environ.get('SLS118_DIR', os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(1)

try:
    import psutil
    PROC = psutil.Process()
except ImportError:
    PROC = None


def rss_mb():
    return PROC.memory_info().rss / 1e6 if PROC else float('nan')


def measure_27():
    import boost_core
    from boost_core import CSGNNv2
    d = boost_core.build_data(os.path.join(CODE, 'contingency_data.npz'),
                              os.path.join(CODE, 'grouped_split.npz'), 'cpu')
    sd = torch.load(os.path.join(CODE, 'full_a0.pt'), map_location='cpu', weights_only=False)
    sd = {k: v for k, v in sd.items() if k != 'n_averaged'}
    m = CSGNNv2(branched=False, stack=False, vbus=False, n_gat=3)
    m.load_state_dict(sd)
    m.set_base_mask(d['BASE'])
    m.eval()
    n_branch = int(d['DENSE'].shape[1])       # placeholder, real count below
    idx = d['idx_te'][:48]                    # 48 branches = one N-1 sweep on 27-bus
    base = rss_mb()
    with torch.no_grad():
        for _ in range(3):
            m(d['NF'][idx], d['DENSE'][idx], d['ADJ'][idx])
        ts = []
        for _ in range(10):
            t0 = time.time()
            m(d['NF'][idx], d['DENSE'][idx], d['ADJ'][idx])
            ts.append(time.time() - t0)
    peak = rss_mb()
    return dict(
        system='27-bus', n_bus=27, n_branch=48,
        params=sum(p.numel() for p in m.parameters()),
        checkpoint_bytes=os.path.getsize(os.path.join(CODE, 'full_a0.pt')),
        batch=len(idx),
        inference_ms_total=float(np.median(ts)) * 1000,
        inference_ms_per_contingency=float(np.median(ts)) * 1000 / len(idx),
        rss_mb_during_inference=peak,
        rss_mb_delta=peak - base,
    )


def measure_118():
    from core118 import CSGNN118
    from train118 import build_data, batch
    d = build_data(os.path.join(DRIVE, 'contingency_data_118_v2.npz'),
                   os.path.join(DRIVE, 'grouped_split_118_v2.npz'), dev='cpu')
    sd = torch.load(os.path.join(DRIVE, 'model_118_v2_vmax_seed0.pt'), map_location='cpu',
                    weights_only=False)
    sd = {k: v for k, v in sd.items() if k != 'n_averaged'}
    m = CSGNN118(branched=False, stack=False, vbus=False, vmax=True, n_gat=3)
    m.load_state_dict(sd)
    m.set_graph(d['ctx'])
    m.eval()
    idx = d['idx_te'][:186]                   # 186 branches = one N-1 sweep on 118-bus
    nf, cf, adj = batch(d, idx)
    base = rss_mb()
    with torch.no_grad():
        for _ in range(3):
            m(nf, cf, adj)
        ts = []
        for _ in range(10):
            t0 = time.time()
            m(nf, cf, adj)
            ts.append(time.time() - t0)
    peak = rss_mb()
    return dict(
        system='118-bus', n_bus=118, n_branch=186,
        params=sum(p.numel() for p in m.parameters()),
        checkpoint_bytes=os.path.getsize(os.path.join(DRIVE, 'model_118_v2_vmax_seed0.pt')),
        batch=len(idx),
        inference_ms_total=float(np.median(ts)) * 1000,
        inference_ms_per_contingency=float(np.median(ts)) * 1000 / len(idx),
        rss_mb_during_inference=peak,
        rss_mb_delta=peak - base,
    )


def dataset_costs():
    """Wall-clock cost of generating each dataset (both 30,000 scenarios)."""
    return {
        '27-bus': dict(scenarios=30000, minutes=26.0,
                       source='paper.tex:262, single CPU'),
        '118-bus': dict(scenarios=30000, minutes=82.0,
                        source='Colab CPU runtime, measured this session; '
                               'v2 Colab CPU run 2026-09-08: 15,000 scenarios in 2,461 s '
                               '-> 30,000 in ~82 min'),
    }


if __name__ == "__main__":
    print("=" * 78)
    print("R3 #5  COMPUTATIONAL BURDEN -- single-threaded CPU, same machine")
    print("=" * 78)
    rows = []
    for fn in (measure_27, measure_118):
        try:
            r = fn()
            rows.append(r)
        except Exception as e:
            print(f"  {fn.__name__} FAILED: {type(e).__name__}: {e}")
    print()
    print(f"  {'system':10s} {'buses':>6s} {'params':>9s} {'ckpt':>9s} "
          f"{'sweep':>8s} {'per cont.':>11s} {'RSS':>9s}")
    for r in rows:
        print(f"  {r['system']:10s} {r['n_bus']:6d} {r['params']:9,d} "
              f"{r['checkpoint_bytes']/1e3:7.0f}kB "
              f"{r['inference_ms_total']:6.1f}ms {r['inference_ms_per_contingency']:9.3f}ms "
              f"{r['rss_mb_during_inference']:7.0f}MB")
    print()
    print("  dataset generation (both 30,000 scenarios):")
    for k, v in dataset_costs().items():
        mins = f"{v['minutes']:.0f} min" if v['minutes'] else "not measured"
        print(f"    {k:8s} {mins:10s}  ({v['source']})")
    print()
    print("  STILL NEEDED FROM COLAB (run colab_burden_cell.py):")
    print("    - exact training time per seed for BOTH systems on the same A100")
    print("    - peak GPU memory during training for both")
    with open(os.path.join(HERE, 'compute_burden.json'), 'w') as f:
        json.dump(dict(cpu_measurements=rows, dataset=dataset_costs()), f, indent=2, default=float)
    print("\n  saved compute_burden.json")
