"""
Smart Load Shield -- IEEE 26-Bus Proactive Stability Monitoring Dashboard
Flask backend on port 5003.

CASE A: No renewables + load scaling + line disconnection (faults)
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandapower as pp
import os, sys
from copy import deepcopy
from flask import Flask, render_template, jsonify, request

# ieee26_bus.py lives in code/; boost_core.py and the checkpoint sit beside
# this file, so the script's own directory (already on sys.path) covers those.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
from ieee26_bus import build_ieee26, add_bus27_pv

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
# Refuse to emit NaN/Infinity in responses (browsers reject as invalid JSON)
app.config['JSON_AS_ASCII'] = False
try:
    app.json.compact = False
    app.json.sort_keys = False
except AttributeError:
    pass

# ============================================================
# LOAD MODEL + DATA
# ============================================================
print("Loading network topology and normalisation assets...")

# Topology metadata and normalisation constants for the dashboard. These were
# previously read out of the 30,000-scenario corpus file; they are ten small
# arrays, so they are shipped separately rather than requiring a 23 MB download.
data = np.load(os.path.join(SAVE_DIR, 'dashboard_assets.npz'))
NODE_MEAN = data['node_mean']
NODE_STD = data['node_std']
EDGE_MEAN = data['edge_mean']
EDGE_STD = data['edge_std']
ADJ = data['adj_matrix']
EDGE_PAIRS = data['edge_pairs']
BUS_INDICES = data['bus_indices']
GEN_POS = data['gen_positions'].tolist()
N_BUSES = int(data['n_buses'])
N_EDGES = int(data['n_edges'])
del data

adj_t = torch.tensor(ADJ, dtype=torch.float32)
bus_to_pos = {int(b): i for i, b in enumerate(BUS_INDICES)}

# ---- contingency-screening surrogate: the paper's deployed model (full_a0, masked CSGNNv2) ----
from boost_core import CSGNNv2
model = CSGNNv2(node_dim=4, edge_dim=3, hidden=128, heads=4, branched=False, stack=False, vbus=False)
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'full_a0.pt'), map_location='cpu'))
model.eval()

# topology from cs_assets (edge_pairs verified identical to the training data); normalization
# from grouped_split.npz -- the stats full_a0 was actually trained with (cs_assets stats differ)
_cs = np.load(os.path.join(SAVE_DIR, 'cs_assets.npz'))
_gs = np.load(os.path.join(SAVE_DIR, 'grouped_split.npz'))
CS_NMEAN = torch.tensor(_gs['nmean']).float(); CS_NSTD = torch.tensor(_gs['nstd']).float()
CS_EMEAN = torch.tensor(_gs['emean']).float(); CS_ESTD = torch.tensor(_gs['estd']).float()
CS_PAIRS = torch.tensor(_cs['edge_pairs'], dtype=torch.long)   # [n_branch,2] bus-position pairs
CS_BUS_IDS = _cs['bus_ids'].tolist()                          # node ordering
CS_FR, CS_TO = CS_PAIRS[:, 0], CS_PAIRS[:, 1]
CS_NB = int(_cs['n_bus']); CS_NBR = int(_cs['n_branch'])
cs_b2i = {b: i for i, b in enumerate(CS_BUS_IDS)}
# round-6 #1 base-edge mask (same fixed topology every inference)
_BASE = torch.zeros(CS_NB, CS_NB); _BASE[CS_FR, CS_TO] = 1.0; _BASE[CS_TO, CS_FR] = 1.0
model.set_base_mask(_BASE)

# Build TWO base grids: with and without PV
base_net_with_pv = build_ieee26()
add_bus27_pv(base_net_with_pv)
pp.runpp(base_net_with_pv, enforce_q_lims=True, max_iteration=30)

base_net_no_pv = build_ieee26()
# Add Bus 27 but with 0 MW so dimensions match the model (27 buses)
add_bus27_pv(base_net_no_pv)
# Set PV to 0 MW for no-renewables case
for idx in base_net_no_pv.gen.index:
    if 'PV' in base_net_no_pv.gen.at[idx, 'name']:
        base_net_no_pv.gen.at[idx, 'p_mw'] = 0.0
pp.runpp(base_net_no_pv, enforce_q_lims=True, max_iteration=30)

# Rebuild edge bookkeeping from LIVE grid in case topology changed since the
# .npz was saved (e.g. after Bus 27 rewire on 2026-04-25 — Dr. Ahmad spec).
live_n_edges = len(base_net_with_pv.line) + len(base_net_with_pv.trafo)
if live_n_edges != N_EDGES:
    print(f"WARNING: live grid has {live_n_edges} edges but saved npz has {N_EDGES}.")
    print("         Predictions are UNRELIABLE until you regenerate data + retrain on Colab.")
    new_pairs = []
    for i in base_net_with_pv.line.index:
        new_pairs.append((bus_to_pos[int(base_net_with_pv.line.at[i,'from_bus'])],
                          bus_to_pos[int(base_net_with_pv.line.at[i,'to_bus'])]))
    for i in base_net_with_pv.trafo.index:
        new_pairs.append((bus_to_pos[int(base_net_with_pv.trafo.at[i,'hv_bus'])],
                          bus_to_pos[int(base_net_with_pv.trafo.at[i,'lv_bus'])]))
    EDGE_PAIRS = np.array(new_pairs, dtype=np.int64)
    if EDGE_MEAN.shape[0] < live_n_edges:
        pad = live_n_edges - EDGE_MEAN.shape[0]
        EDGE_MEAN = np.vstack([EDGE_MEAN, np.tile(EDGE_MEAN.mean(0, keepdims=True), (pad, 1))])
        EDGE_STD  = np.vstack([EDGE_STD,  np.tile(EDGE_STD.mean(0, keepdims=True),  (pad, 1))])
    N_EDGES = live_n_edges

# Get list of all lines for the frontend, with R/X/B characteristics from Dr. Ahmad's images
_Z_BASE_OHM = 230.0**2 / 100.0  # 529 ohms

LINE_LIST = []
for i in base_net_with_pv.line.index:
    f = int(base_net_with_pv.line.at[i, 'from_bus'])
    t = int(base_net_with_pv.line.at[i, 'to_bus'])
    name = base_net_with_pv.line.at[i, 'name']
    r_pu = float(base_net_with_pv.line.at[i, 'r_ohm_per_km']) / _Z_BASE_OHM
    x_pu = float(base_net_with_pv.line.at[i, 'x_ohm_per_km']) / _Z_BASE_OHM
    c_nf = float(base_net_with_pv.line.at[i, 'c_nf_per_km'])
    half_b_pu = (c_nf * 1e-9 * 2 * 3.14159265 * 60 * _Z_BASE_OHM) / 2.0
    LINE_LIST.append({
        'index': int(i), 'from': f, 'to': t, 'name': name,
        'r_pu': round(r_pu, 4),
        'x_pu': round(x_pu, 4),
        'b_half_pu': round(half_b_pu, 4),
    })

TRAFO_LIST = []
for i in base_net_with_pv.trafo.index:
    hv = int(base_net_with_pv.trafo.at[i, 'hv_bus'])
    lv = int(base_net_with_pv.trafo.at[i, 'lv_bus'])
    name = base_net_with_pv.trafo.at[i, 'name']
    vk = float(base_net_with_pv.trafo.at[i, 'vk_percent']) / 100.0
    vkr = float(base_net_with_pv.trafo.at[i, 'vkr_percent']) / 100.0
    r_pu = vkr
    x_pu = (max(0, vk**2 - vkr**2)) ** 0.5
    tap_step = float(base_net_with_pv.trafo.at[i, 'tap_step_percent'])
    # Tap setting from image: tap = 1 - tap_step/100 (with sign per image data)
    # Most IEEE 26-bus trafos have tap < 1 (step-down), except 3-13 (1.017) and 4-8/4-12 (1.050)
    image_taps = {(2,3): 0.960, (2,13): 0.960, (3,13): 1.017,
                  (4,8): 1.050, (4,12): 1.050, (6,19): 0.950, (7,9): 0.950}
    tap = image_taps.get((hv, lv), 1.0)
    TRAFO_LIST.append({
        'index': int(i), 'hv': hv, 'lv': lv, 'name': name,
        'r_pu': round(r_pu, 4),
        'x_pu': round(x_pu, 4),
        'tap': round(tap, 3),
    })

print(f"Ready! {N_BUSES} buses, {len(LINE_LIST)} lines, {len(TRAFO_LIST)} trafos")

# ============================================================
# HELPERS
# ============================================================
def extract_bus_features(sim):
    feat = np.zeros((N_BUSES, 4), dtype=np.float32)
    for j, bid in enumerate(BUS_INDICES):
        if bid in sim.res_bus.index:
            v = sim.res_bus.at[bid, 'vm_pu']
            a = sim.res_bus.at[bid, 'va_degree']
            p = sim.res_bus.at[bid, 'p_mw']
            q = sim.res_bus.at[bid, 'q_mvar']
            feat[j] = [v if not np.isnan(v) else 0,
                        a if not np.isnan(a) else 0,
                        p if not np.isnan(p) else 0,
                        q if not np.isnan(q) else 0]
    return feat

def extract_edge_features(sim):
    feat = np.zeros((N_EDGES, 3), dtype=np.float32)
    idx = 0
    for i in sim.line.index:
        if sim.line.at[i, 'in_service'] and i in sim.res_line.index:
            ld = sim.res_line.at[i, 'loading_percent']
            pf = sim.res_line.at[i, 'p_from_mw']
            feat[idx] = [ld if not np.isnan(ld) else 0,
                          pf if not np.isnan(pf) else 0, 1.0]
        else:
            feat[idx, 2] = float(sim.line.at[i, 'in_service'])
        idx += 1
    for i in sim.trafo.index:
        if sim.trafo.at[i, 'in_service'] and i in sim.res_trafo.index:
            ld = sim.res_trafo.at[i, 'loading_percent']
            pf = sim.res_trafo.at[i, 'p_hv_mw']
            feat[idx] = [ld if not np.isnan(ld) else 0,
                          pf if not np.isnan(pf) else 0,
                          float(sim.trafo.at[i, 'in_service'])]
        else:
            feat[idx, 2] = float(sim.trafo.at[i, 'in_service'])
        idx += 1
    return feat

def to_dense(edge_feat):
    dense = np.zeros((N_BUSES, N_BUSES, 3), dtype=np.float32)
    for idx, (s, t) in enumerate(EDGE_PAIRS):
        dense[s, t, :] = edge_feat[idx, :]
        dense[t, s, :] = edge_feat[idx, :]
    return dense

def build_dynamic_adj(sim):
    """Rebuild adjacency from current in-service branches so tripped lines
    actually break GNN message passing (and thus change swing trajectory)."""
    a = np.eye(N_BUSES, dtype=np.float32)
    for i in sim.line.index:
        if bool(sim.line.at[i, 'in_service']):
            f = bus_to_pos[int(sim.line.at[i, 'from_bus'])]
            t = bus_to_pos[int(sim.line.at[i, 'to_bus'])]
            a[f, t] = 1.0; a[t, f] = 1.0
    for i in sim.trafo.index:
        if bool(sim.trafo.at[i, 'in_service']):
            f = bus_to_pos[int(sim.trafo.at[i, 'hv_bus'])]
            t = bus_to_pos[int(sim.trafo.at[i, 'lv_bus'])]
            a[f, t] = 1.0; a[t, f] = 1.0
    return torch.tensor(a, dtype=torch.float32)


NL_LINES = len(base_net_with_pv.line)   # CS branch slots: lines [0..NL_LINES-1], trafos after


def cs_node_feats(sim):
    f = np.zeros((CS_NB, 4), np.float32)
    for b in CS_BUS_IDS:
        if b in sim.res_bus.index:
            i = cs_b2i[b]
            f[i, 0] = sim.res_bus.at[b, 'vm_pu']; f[i, 1] = sim.res_bus.at[b, 'va_degree'] / 180.0
            f[i, 2] = sim.res_bus.at[b, 'p_mw'] / 100.0; f[i, 3] = sim.res_bus.at[b, 'q_mvar'] / 100.0
    return np.nan_to_num(f)


def cs_edge_feats(sim):
    f = np.zeros((CS_NBR, 2), np.float32)
    for k, li in enumerate(sim.line.index):
        if li in sim.res_line.index:
            f[k, 0] = (sim.res_line.at[li, 'loading_percent'] or 0) / 100.0
            f[k, 1] = (sim.res_line.at[li, 'p_from_mw'] or 0) / 100.0
    for k, ti in enumerate(sim.trafo.index):
        if ti in sim.res_trafo.index:
            f[NL_LINES + k, 0] = (sim.res_trafo.at[ti, 'loading_percent'] or 0) / 100.0
            f[NL_LINES + k, 1] = (sim.res_trafo.at[ti, 'p_hv_mw'] or 0) / 100.0
    return np.nan_to_num(f)


def predict_cs(base_sim, tripped_slots):
    """Contingency-screening surrogate: predict the POST-contingency outcome from the
    PRE-contingency state (base_sim, no outage) plus a contingency mask, WITHOUT solving
    the post-contingency power flow."""
    nf = (torch.tensor(cs_node_feats(base_sim)) - CS_NMEAN) / CS_NSTD
    ef = (torch.tensor(cs_edge_feats(base_sim)) - CS_EMEAN) / CS_ESTD
    insvc = torch.ones(CS_NBR)
    for s in tripped_slots:
        if 0 <= s < CS_NBR:
            insvc[s] = 0.0
    dense = torch.zeros(1, CS_NB, CS_NB, 3)
    dense[0, CS_FR, CS_TO, 0:2] = ef; dense[0, CS_TO, CS_FR, 0:2] = ef
    dense[0, CS_FR, CS_TO, 2] = insvc; dense[0, CS_TO, CS_FR, 2] = insvc
    adj = torch.zeros(1, CS_NB, CS_NB)
    adj[0, CS_FR, CS_TO] = insvc; adj[0, CS_TO, CS_FR] = insvc
    adj[0, torch.arange(CS_NB), torch.arange(CS_NB)] = 1.0
    with torch.no_grad():
        risk, vmin, vuln, div, _vb = model(nf.unsqueeze(0), dense, adj)
    rp = F.softmax(risk, 1).squeeze(0).numpy(); rc = int(risk.argmax(1).item())
    # vuln_h emits logits (boost_core.py:96); every other consumer in
    # boost_core sigmoids it first. The frontend renders this field as a
    # percentage, and the diverged path pins it to 1.0, so it must be a
    # probability in [0, 1] -- without this the map showed raw logits.
    vuln_cs = torch.sigmoid(vuln).squeeze(0).numpy()
    return {
        'risk_class': rc,
        'risk_label': ['STABLE', 'MARGINAL', 'UNSTABLE'][rc],
        'risk_probs': {'stable': round(float(rp[0]), 4),
                       'marginal': round(float(rp[1]), 4),
                       'unstable': round(float(rp[2]), 4)},
        'predicted_min_voltage': round(float(vmin.item()), 4),
        'divergence_prob': round(float(div.item()), 4),
        # re-order per-bus vulnerability to the frontend's BUS_INDICES ordering
        'bus_vulnerability': [round(float(vuln_cs[cs_b2i[int(b)]]), 4) for b in BUS_INDICES],
    }

# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/lines', methods=['GET'])
def get_lines():
    """Return list of all lines and trafos for the disconnect dropdown."""
    return jsonify({'lines': LINE_LIST, 'trafos': TRAFO_LIST})

@app.route('/api/monitor', methods=['POST'])
def monitor():
    params = request.json or {}
    # Defensive parsing — never let a malformed payload crash the API
    def _to_float(v, default):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default
    def _to_int_list(v):
        if not isinstance(v, list):
            return []
        out = []
        for x in v:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                continue
        return out

    load_s = max(0.01, _to_float(params.get('load_scale'), 1.0))
    pv_mw = max(0.0, min(1000.0, _to_float(params.get('pv_mw'), 100.0)))
    use_renewables = bool(params.get('use_renewables', True))
    disconnect_lines = _to_int_list(params.get('disconnect_lines', []))
    disconnect_trafos = _to_int_list(params.get('disconnect_trafos', []))

    # Choose base grid
    if use_renewables:
        sim = deepcopy(base_net_with_pv)
    else:
        sim = deepcopy(base_net_no_pv)

    # Scale loads
    sim.load['p_mw'] *= load_s
    sim.load['q_mvar'] *= load_s

    # Set PV output (absolute MW)
    if use_renewables:
        for idx in sim.gen.index:
            if 'PV' in sim.gen.at[idx, 'name']:
                sim.gen.at[idx, 'p_mw'] = pv_mw

    # PRE-contingency solve (current operating point, no outage) -> the input state
    # for the screening surrogate. The surrogate predicts the post-contingency outcome
    # from this pre-fault state plus the contingency mask, without solving the post-fault PF.
    base_feat_sim = deepcopy(sim)
    base_converged = True
    try:
        pp.runpp(base_feat_sim, enforce_q_lims=True, max_iteration=30)
    except Exception:
        base_converged = False
    # map tripped components to contingency-surrogate branch slots (lines first, then trafos)
    tripped_slots = [int(li) for li in disconnect_lines] + \
                    [NL_LINES + int(ti) for ti in disconnect_trafos]

    # Disconnect lines (fault simulation)
    disconnected = []
    for li in disconnect_lines:
        li = int(li)
        if li in sim.line.index:
            sim.line.at[li, 'in_service'] = False
            f = int(sim.line.at[li, 'from_bus'])
            t = int(sim.line.at[li, 'to_bus'])
            disconnected.append(f'Line {f}-{t}')

    for ti in disconnect_trafos:
        ti = int(ti)
        if ti in sim.trafo.index:
            sim.trafo.at[ti, 'in_service'] = False
            hv = int(sim.trafo.at[ti, 'hv_bus'])
            lv = int(sim.trafo.at[ti, 'lv_bus'])
            disconnected.append(f'Trafo {hv}-{lv}')

    # Run power flow (post-fault)
    converged = True
    try:
        pp.runpp(sim, enforce_q_lims=True, max_iteration=30)
    except:
        converged = False

    if not converged:
        # Return same shape as success path so frontend & API consumers don't break.
        # All numeric fields = None; vulnerabilities pinned at 1.0 (everything is vulnerable).
        return jsonify({
            'converged': False,
            'message': 'Power flow DIVERGED - system collapsed',
            'prediction': {
                'risk_label': 'UNSTABLE',
                'risk_class': 2,
                'risk_probs': {'stable': 0.0, 'marginal': 0.0, 'unstable': 1.0},
                'predicted_min_voltage': 0.0,
                'divergence_prob': 1.0,
                'bus_vulnerability': [1.0] * len(BUS_INDICES),
            },
            'buses': [{'bus_id': int(b), 'name': f'Bus {b}',
                       'voltage': None, 'angle': None, 'p_mw': None, 'q_mvar': None,
                       'vulnerability': 1.0} for b in BUS_INDICES],
            'generators': [],
            'line_status': [],
            'disconnected': disconnected,
            'metrics': {
                'min_voltage': None, 'max_loading': None,
                'total_gen_mw': None, 'total_load_mw': None,
                'slack_p_mw': None, 'losses_mw': None, 'imbalance': None,
            },
            'operating': {
                'load_scale': load_s, 'pv_mw': pv_mw,
                'use_renewables': use_renewables,
            },
        })

    # The contingency-screening surrogate predicts the post-contingency outcome directly
    # from the pre-fault state and the contingency mask -- no physics override needed
    # (unlike the earlier load/PV-only model, this one is trained on line outages).
    if base_converged:
        pred = predict_cs(base_feat_sim, tripped_slots)
    else:
        pred = {'risk_class': 2, 'risk_label': 'UNSTABLE',
                'risk_probs': {'stable': 0.0, 'marginal': 0.0, 'unstable': 1.0},
                'predicted_min_voltage': 0.0, 'divergence_prob': 1.0,
                'bus_vulnerability': [1.0] * len(BUS_INDICES)}
    if False:
        pass

    # NaN -> None so JSON.parse in browser doesn't choke (islanded buses go NaN)
    def _safe(x, ndigits=None):
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return None
        return round(float(x), ndigits) if ndigits is not None else float(x)

    buses = []
    for j, bid in enumerate(BUS_INDICES):
        v = _safe(sim.res_bus.at[bid, 'vm_pu'], 4)
        a = _safe(sim.res_bus.at[bid, 'va_degree'], 2)
        p = _safe(sim.res_bus.at[bid, 'p_mw'], 2)
        q = _safe(sim.res_bus.at[bid, 'q_mvar'], 2)
        buses.append({
            'bus_id': int(bid), 'name': f'Bus {bid}',
            'voltage': v, 'angle': a, 'p_mw': p, 'q_mvar': q,
            'vulnerability': pred['bus_vulnerability'][j],
        })

    valid_v = [b['voltage'] for b in buses if b['voltage'] is not None]
    min_v = min(valid_v) if valid_v else None
    slack_p = float(sim.res_ext_grid.at[sim.ext_grid.index[0], 'p_mw'])
    total_load = float(sim.load['p_mw'].sum())
    total_gen = sum(float(sim.res_gen.at[i, 'p_mw']) for i in sim.gen.index) + abs(slack_p)

    # Generation data
    gen_info = []
    for i in sim.gen.index:
        gen_info.append({
            'name': sim.gen.at[i, 'name'],
            'bus': int(sim.gen.at[i, 'bus']),
            'p_mw': _safe(sim.res_gen.at[i, 'p_mw'], 2),
            'q_mvar': _safe(sim.res_gen.at[i, 'q_mvar'], 2),
        })

    # Line status
    line_status = []
    for i in sim.line.index:
        f = int(sim.line.at[i, 'from_bus'])
        t = int(sim.line.at[i, 'to_bus'])
        in_service = bool(sim.line.at[i, 'in_service'])
        loading = 0.0
        if in_service and i in sim.res_line.index:
            ld = sim.res_line.at[i, 'loading_percent']
            loading = float(ld) if not np.isnan(ld) else 0.0
        line_status.append({
            'index': int(i), 'from': f, 'to': t,
            'in_service': in_service,
            'loading': round(loading, 1),
        })

    line_losses = sim.res_line['pl_mw'].fillna(0).sum()
    trafo_losses = sim.res_trafo['pl_mw'].fillna(0).sum()
    losses = float(line_losses) + float(trafo_losses)

    return jsonify({
        'converged': True,
        'prediction': pred,
        'buses': buses,
        'generators': gen_info,
        'line_status': line_status,
        'disconnected': disconnected,
        'metrics': {
            'min_voltage': _safe(min_v, 4),
            'max_loading': _safe(sim.res_line['loading_percent'].fillna(0).max(), 1),
            'total_gen_mw': _safe(total_gen, 1),
            'total_load_mw': _safe(total_load, 1),
            'slack_p_mw': _safe(slack_p, 1),
            'losses_mw': _safe(losses, 2),
            'imbalance': _safe(total_gen - total_load - losses, 3),
        },
        'operating': {
            'load_scale': load_s,
            'pv_mw': pv_mw,
            'use_renewables': use_renewables,
        },
    })

if __name__ == '__main__':
    app.run(debug=False, port=5003)
