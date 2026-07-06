# Smart Load Shield — Leakage-Free Contingency Screening with a Physics-Guided Graph Neural Network

Code, data, trained model, and interactive dashboard for the paper:

> **A Physics-Guided Graph Neural Network for Proactive Bus Voltage Security Screening of Renewable-Energy-Integrated Power Systems**
> Salah Harb and Ahmad Harb, *IEEE Canadian Journal of Electrical and Computer Engineering* (under review).

## What this is

The standard way to pose learning-based voltage-stability monitoring — classify the grid's **present** state — is *circular*: the risk label is a deterministic function of the very bus voltages the model is given, so a near-perfect score can mean the network reconstructed its own inputs rather than predicting anything. We diagnose this **target-leakage** failure mode and reformulate the task **predictively**: from the *pre-contingency* state and a proposed disturbance, forecast the *post-contingency* risk class, minimum bus voltage, per-bus vulnerability, and power-flow divergence — **without solving the post-contingency power flow**.

A physics-guided graph-attention surrogate (masked, coupled multi-task) reasons over the switched post-contingency topology against an AC power-flow oracle on a renewable-integrated 27-bus network (the 26-bus benchmark plus an embedded PV generator at Bus 27).

### Headline results (deployed model, operating-point-disjoint test split)

| Metric | Value |
|---|---|
| Risk accuracy | **98.1%** |
| False-safe (unstable predicted stable) | **0 of 1,899** (zero on all 5 seeds) |
| Minimum-voltage MAE / R² | 0.008 p.u. / 0.96 |
| Expected calibration error | 0.035 |
| Per-bus vulnerability ROC-AUC / PR-AUC | 0.998 / 0.997 |
| Nonconvergence-head F1 | 0.98 |
| Screening speedup vs Newton–Raphson | ~40× (surrogate-only, single CPU core) |

A held-out-branch study honestly bounds the safety property: it holds across unseen operating conditions but degrades on contingencies whose branch was never seen in training — a regime an inference-time membership check can route to the exact solver.

## Repository layout

```
smart-load-shield/
├── paper/paper.pdf              # the manuscript
├── code/                        # reproduction pipeline (run scripts from here)
│   ├── boost_core.py            #   model definition + training (CSGNNv2, masked coupled multi-task)
│   ├── ieee26_bus.py            #   27-bus network builder (26-bus benchmark + PV Bus 27)
│   ├── gen_contingency_data.py  #   -> contingency_data.npz  (30k leakage-free scenarios)
│   ├── make_grouped_split.py    #   -> grouped_split.npz     (operating-point-disjoint split)
│   ├── make_full_a0_paper.py    #   deployed metrics + confusion / Vmin-parity / vulnerability figures
│   ├── local_analyses.py        #   calibration, cross-head consistency, boundary-band MAE, subgroup recall, reliability figure
│   ├── make_heldout_split.py    #   held-out-branch generalisation split
│   ├── screening_speedup.py     #   oracle-vs-surrogate timing (~40x)
│   ├── baselines_gnn.py         #   graph baselines (GCN, GraphSAGE, node-only GAT)
│   ├── baselines_cs.py          #   non-graph baselines (logistic regression, random forest, MLP)
│   ├── testbench.py             #   50-category, 210-assertion cross-layer verification harness
│   ├── full_a0.pt               #   deployed model weights (masked coupled multi-task)
│   ├── grouped_split.npz        #   the split + train-only normalisation stats
│   ├── contingency_data.npz     #   30,000-scenario dataset (~20 MB)
│   └── metrics/*.json           #   every number reported in the paper, from the deployed model
├── dashboard/                   # self-contained Flask + Cytoscape.js control-room demo
│   ├── app.py                   #   loads full_a0.pt (the deployed coupled model)
│   ├── templates/index.html
│   └── screenshots/             #   the three dashboard figures in the paper
└── notebooks/                   # Colab notebooks (GPU): training, 5-seed baselines, held-out study, ablation, PV retrains
```

## Quick start

### Run the dashboard
```bash
cd dashboard
pip install -r requirements.txt
python app.py            # then open http://localhost:5000
```
Adjust load and PV, click any branch to trip it, and see the predicted risk band, per-bus vulnerability heatmap, and contingency consequences update in place — driven by the deployed model.

### Reproduce the paper numbers and figures
```bash
cd code
pip install -r ../requirements.txt
python make_full_a0_paper.py    # deployed metrics + confusion / Vmin / vulnerability figures + cs_results_full_a0.json
python local_analyses.py        # calibration / cross-head / boundary-MAE / subgroup + reliability_cs.png
python screening_speedup.py     # ~40x screening speedup
python baselines_cs.py          # non-graph baselines (Table II)
python testbench.py             # verification harness (209 pass / 1 warning / 0 fail)
```
`make_full_a0_paper.py` loads the released `full_a0.pt` on the `grouped_split.npz` test split, so the printed metrics match the paper exactly. To regenerate the dataset and split from scratch, run `gen_contingency_data.py` then `make_grouped_split.py` first (dataset generation ≈ 26 min on one CPU).

### Retrain (Colab, GPU)
The notebooks in `notebooks/` reproduce the training and the study tables. They mount Google Drive and read/write a `smart_load_shield_boost` folder containing `contingency_data.npz`, `grouped_split.npz`, and `boost_core.py`. Training completes in a few minutes per seed on an NVIDIA A100.

## Model

- **Architecture:** `CSGNNv2` — node/edge encoders → 3 masked graph-attention layers → pooled multi-task heads (risk, minimum voltage, per-bus vulnerability, divergence). 163,592 parameters; runs on a single CPU core.
- **Deployed weights:** `full_a0.pt` — masked, **coupled** multi-task (auxiliary gradients into the shared trunk), best of five seeds selected on validation at zero false-safe.
- **Split:** grouped by operating point (no operating point shared between train and test), with train-only normalisation — closes the operating-point and preprocessing leakage paths.

## Citation

```bibtex
@article{harb2026smartloadshield,
  author  = {Salah Harb and Ahmad Harb},
  title   = {A Physics-Guided Graph Neural Network for Proactive Bus Voltage Security Screening of Renewable-Energy-Integrated Power Systems},
  journal = {IEEE Canadian Journal of Electrical and Computer Engineering},
  year    = {2026},
  note    = {Code and data: \url{https://github.com/xharbx/smart-load-shield}}
}
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
