# Smart Load Shield — a switched-topology graph-attention surrogate for post-contingency voltage-security screening

Code, data and an interactive dashboard for **leakage-free post-contingency
voltage-security screening** on a renewable-integrated 27-bus network and on the
standard IEEE 118-bus system.

This repository accompanies:

> S. Harb and A. Harb, *"Post-Contingency Voltage Security Screening in
> Renewable-Rich Power Systems Using a Switched-Topology Graph Attention
> Network,"* **IEEE Canadian Journal of Electrical and Computer Engineering**
> (under review), 2026.

---

## What the task is, and what it is not

Given the **pre-contingency** operating state and a proposed disturbance, the
surrogate predicts the **post-contingency** risk class, minimum bus voltage,
per-bus vulnerability and power-flow divergence, without solving the
post-contingency power flow.

Two boundaries are worth stating plainly, because both were tightened during
review:

- **This is steady-state post-contingency voltage *security*, not voltage
  stability.** Every label is a single AC power-flow solution. There is no
  time-domain simulation, no fault duration, no clearing time and no
  voltage-recovery trajectory, and the code makes no transient, dynamic or
  rotor-angle claim.
- **No physical constraint is embedded in the loss or the network.** No
  power-flow equation, power-balance residual or electrical-parameter
  constraint enters the objective. What grounding exists is that every training
  target comes from an AC power-flow oracle, and that message passing is
  constrained to the switched post-contingency topology. Representing topology
  through a graph is established practice and is adopted here, not proposed.

## The leakage problem this repository exists to avoid

Classifying the grid's **present** state is circular: the risk label is a
deterministic threshold on the very bus voltages the model receives, so a model
can score highly by inverting its own labelling rule rather than by predicting
anything.

An earlier version of this system did exactly that, reaching near-perfect
apparent accuracy. **Those numbers are not in this repository and should not be
cited.** The pipeline here is the leakage-free reformulation: inputs are strictly
pre-contingency, the contingency enters only as a mask over the topology, and the
train/validation/test split is disjoint by *operating point*, not by scenario.

A direct test of leakage-freedom is included: at a representative operating
point, 84 distinct contingencies share byte-identical node features yet span all
three risk classes, so no function of the node features alone can reproduce the
labels.

---

## Layout

| Path | What it is |
|---|---|
| `code/ieee26_bus.py` | the 26-bus benchmark plus the Bus-27 PV plant |
| `code/gen_contingency_data.py` | 27-bus corpus generator (30,000 scenarios) |
| `code/make_grouped_split.py` | operating-point-disjoint 70/15/15 split |
| `code/boost_core.py` | the surrogate: model, training and evaluation |
| `code/model_cs.py` | the dense reference implementation of the model |
| `code/baselines_cs.py`, `code/baselines_gnn.py` | same-data baselines (27-bus) |
| `code/testbench.py`, `code/local_analyses.py` | evaluation and per-case analyses |
| `code/screening_speedup.py` | screening-throughput benchmark against the AC solver |
| `code/contingency_data.npz` | the 27-bus corpus |
| `code/grouped_split.npz` | the split, with train-only normalisation statistics |
| `code/full_a0.pt` | the deployed 27-bus checkpoint |
| `code/metrics/*.json` | the recorded results behind the reported numbers |
| `dashboard/` | Flask + Cytoscape.js control-room dashboard |
| `notebooks/` | Colab notebooks for the 27-bus experiments |
| `scale118/` | the IEEE 118-bus scalability study |

### `scale118/`

| File | What it is |
|---|---|
| `gen118.py` | 118-bus generator: schedule band, operating-point admission test, two-sided risk rule |
| `core118.py` | memory-efficient rewrite (sparse edge encoding, factorised attention) |
| `train118.py` | training, evaluation, screening benchmark, Unstable-class breakdown |
| `test_equivalence.py` | proves `core118` is numerically identical to `boost_core` |
| `verify_twosided_27bus.py` | proves the two-sided rule changes **zero** 27-bus labels |
| `guardband_calibrate.py` | split-conformal calibration of the overvoltage guard band |
| `baselines118.py` | same-data baselines on the 118-bus corpus |
| `compute_burden.py` | computational-burden measurements |
| `smoke_test.py` | whole pipeline end to end on a small fixture, on CPU |
| `verify_notebooks.py` | checks the notebooks execute and match the modules |
| `*_colab.ipynb` | notebooks that reproduce generation, training and burden on Colab |

The 118-bus corpus is not committed here because of its size. It is
**deterministically reproducible**: `python gen118.py` regenerates both the
corpus and its operating-point-disjoint split from a fixed seed.

---

## Running the dashboard

```bash
cd dashboard
pip install -r requirements.txt
python app.py            # then open http://localhost:5003
```

Adjust load and PV output, click any branch to trip it, and the predicted risk
band, per-bus vulnerability heatmap and post-contingency consequences update in
place, driven by the deployed leakage-free checkpoint. The dashboard carries its
own copy of the checkpoint, the split statistics and the topology metadata, so it
runs without regenerating the corpus.

## Reproducing

```bash
pip install -r requirements.txt
```

### The 27-bus results

These run against the released checkpoint and split, so the printed numbers are
the ones in the paper:

```bash
cd code
python make_full_a0_paper.py   # deployed metrics, confusion / Vmin / vulnerability figures
python local_analyses.py       # calibration, cross-head, boundary MAE, subgroups
python screening_speedup.py    # screening throughput against the AC solver
python baselines_cs.py         # non-graph baselines
python testbench.py            # verification harness
```

To rebuild the corpus and split from scratch beforehand:

```bash
python gen_contingency_data.py   # ~26 min on one CPU
python make_grouped_split.py     # operating-point-disjoint 70/15/15
```

### The 118-bus results

```bash
cd scale118
python gen118.py                 # corpus + split, ~82 min on one CPU
```

`train118.py` is a module rather than a script; `train_118_colab.ipynb` drives it
on a GPU, which is how the reported runs were produced. Directly:

```python
from train118 import build_data, train_eval
d = build_data('contingency_data_118_v2.npz', 'grouped_split_118_v2.npz', 'cuda')
res, model = train_eval(d, seed=0, epochs=150, bs=256, vmax=True)
```

Set `SLS118_DIR` if the corpus lives somewhere other than `scale118/`.

Every stage uses fixed seeds, so the partition and the corpus are reproducible
from this code. Note that GPU training is **not** bit-reproducible: `index_add_`
on CUDA uses atomics, so run-to-run spread at a fixed seed can exceed
seed-to-seed spread. Results are therefore reported as mean ± standard deviation
over five seeds, never from a single run.

## Verification

```bash
python scale118/test_equivalence.py        # 25 checks: core118 == boost_core
python scale118/verify_twosided_27bus.py   # 0 labels change under the two-sided rule
python scale118/smoke_test.py              # whole 118-bus pipeline on a small fixture
python scale118/verify_notebooks.py        # notebooks execute and match the modules
```

Run `smoke_test.py` before `verify_notebooks.py`: it builds the small fixture the
notebook check executes against. All four run on CPU in a few minutes and need no
GPU, no Colab and no generated corpus.
