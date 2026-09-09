"""
Verify the generated notebooks BEFORE they are uploaded to Colab.

A notebook that fails on Colab costs a session and a reconnect, so everything
checkable offline is checked here:

  V1  valid JSON, valid nbformat structure, no empty cells
  V2  every code cell compiles as Python (Colab magics stripped first)
  V3  no files.upload(); Drive is mounted; the Drive folder convention is used
  V4  the inlined module cells EXECUTE in a fresh namespace and define every
      symbol the later cells call
  V5  the later cells' free variables are all actually defined by then
  V6  the training notebook's real path runs on the smoke dataset: build_data,
      one training step, evaluate, screening benchmark -- in a namespace built
      only from the notebook's own cells
"""
import ast
import io
import json
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.path.join(HERE, '_smoke')
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def cell_src(c):
    """
    Join EXACTLY as Jupyter/Colab does: plain concatenation, no repair.

    An earlier version of this helper re-added missing newlines, which silently
    hid a generator bug that stripped them -- every cell rendered as one run-on
    line in Colab while every check here passed. Never normalise the thing you
    are trying to verify.
    """
    s = c['source']
    return s if isinstance(s, str) else ''.join(s)


def newline_violations(c):
    """Every source line but the last must carry its trailing newline."""
    s = c['source']
    if isinstance(s, str):
        return 0
    return sum(1 for x in s[:-1] if not x.endswith('\n'))


def strip_magics(src):
    out = []
    for ln in src.split('\n'):
        st = ln.lstrip()
        if st.startswith('!') or st.startswith('%'):
            out.append(' ' * (len(ln) - len(st)) + 'pass')
        else:
            out.append(ln)
    return '\n'.join(out)


def load(nb_name):
    with open(os.path.join(HERE, nb_name), 'r', encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------- V1-V3
def structural(nb_name):
    print(f"\n--- {nb_name} ---")
    nb = load(nb_name)
    check("valid nbformat 4", nb.get('nbformat') == 4)
    check("has cells", len(nb['cells']) > 0, f"{len(nb['cells'])} cells")
    empty = [i for i, c in enumerate(nb['cells']) if not cell_src(c).strip()]
    check("no empty cells", not empty, f"empty at {empty}" if empty else "")

    # The bug that shipped once: lines stored without trailing newlines render
    # as one run-on line in Colab even though the JSON is perfectly valid.
    bad_nl = {i: newline_violations(c) for i, c in enumerate(nb['cells'])
              if newline_violations(c)}
    check("source lines keep their trailing newlines", not bad_nl,
          f"cells {sorted(bad_nl)} missing newlines" if bad_nl else "")
    multi = [i for i, c in enumerate(nb['cells'])
             if len(c['source']) > 1 and '\n' in cell_src(c)]
    check("multi-line cells actually contain newlines",
          len(multi) == len([c for c in nb['cells'] if len(c['source']) > 1]),
          f"{len(multi)} multi-line cells")

    codes = [cell_src(c) for c in nb['cells'] if c['cell_type'] == 'code']
    bad = []
    for i, src in enumerate(codes):
        try:
            compile(strip_magics(src), f'<cell {i}>', 'exec')
        except SyntaxError as e:
            bad.append((i, str(e)))
    check("every code cell compiles", not bad,
          "; ".join(f"cell {i}: {m}" for i, m in bad) if bad else f"{len(codes)} cells")

    joined = "\n".join(codes)
    check("no files.upload()", 'files.upload' not in joined)
    check("mounts Google Drive", 'drive.mount' in joined)
    check("uses smart_load_shield_boost", 'smart_load_shield_boost' in joined)
    check("writes into a scale118 subfolder", "'scale118'" in joined)
    check("no leftover cross-module import",
          'from core118 import' not in joined and 'import gen118' not in joined)
    return nb, codes


# ---------------------------------------------------------------- V4-V5
def semantic(nb_name, codes, expect_symbols):
    """Execute the module cells for real, then check the remaining cells' names resolve."""
    print(f"  semantic checks for {nb_name}")
    ns = {'__name__': '__notebook__'}
    module_cells = [s for s in codes
                    if ('def generate(' in s or 'class CSGNN118' in s or 'def train_eval(' in s)]
    check("module cells found", len(module_cells) >= 1, f"{len(module_cells)} module cells")

    buf = io.StringIO()
    old = sys.stdout
    try:
        sys.stdout = buf
        for src in module_cells:
            exec(compile(strip_magics(src), '<module cell>', 'exec'), ns)
    except Exception as e:
        sys.stdout = old
        check("module cells execute", False, f"{type(e).__name__}: {e}")
        return ns
    finally:
        sys.stdout = old

    check("module cells execute", True)
    missing = [s for s in expect_symbols if s not in ns]
    check("all expected symbols defined", not missing,
          f"missing {missing}" if missing else f"{len(expect_symbols)} symbols")

    # names used by non-module cells must exist by the time they run
    provided = set(ns) | set(dir(__builtins__)) | {
        'drive', 'FOLDER', 'OUT_DIR', 'NPZ', 'SPLIT', 'glob', 'os', 'sys', 'json', 'time',
        'torch', 'np', 'pandapower', 'numba', 'runs', 'models', 'model', 'res', 's', 'r',
        'best_i', 'd', 'dev', 'sb', 'out', 'path', 'types', 'gen_module', 'summary',
        'EPOCHS', 'BATCH', 'SEEDS', 'VMAX', 'TAG', 'TARGET', 'N_OP', 'NUMBA',
        't0', 'part', 'f', 'c', 'n',
        'D', 'risk', 'div', 'ins', 'vuln', 'net0', 'branches', 'islanding_branch', 'k',
        'typ', 'bi', 'sim', 'unst', 'diverged', 'hit_islander', 'top', 'deepcopy',
        'name', 'row', 'accs', 'x', 'cand', 'i',
        'PANDAPOWER_VERSION', 'importlib', 'NUMBA_PF', 'n_isl', 'n_solo', 'cache',
        'key', 'islands', 'trip_idx', 'b', 'bb', 'ii', 'nb',
    }
    unresolved = set()
    for src in codes:
        if src in module_cells:
            continue
        try:
            tree = ast.parse(strip_magics(src))
        except SyntaxError:
            continue
        assigned = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store,)):
                assigned.add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    assigned.add((a.asname or a.name).split('.')[0])
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assigned.add(node.name)
            elif isinstance(node, ast.comprehension):
                for nn in ast.walk(node.target):
                    if isinstance(nn, ast.Name):
                        assigned.add(nn.id)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in provided and node.id not in assigned:
                    unresolved.add(node.id)
    import builtins
    unresolved -= set(dir(builtins))
    check("no unresolved names in non-module cells", not unresolved,
          f"unresolved: {sorted(unresolved)}" if unresolved else "")
    return ns


# ---------------------------------------------------------------- V6
def functional(ns):
    """Run the training notebook's real path, using ONLY symbols the notebook defines."""
    print("  functional check: training path on the smoke dataset")
    npz = os.path.join(SMOKE, 'smoke_118.npz')
    split = os.path.join(SMOKE, 'smoke_split_118.npz')
    if not (os.path.exists(npz) and os.path.exists(split)):
        check("smoke dataset present (run smoke_test.py first)", False)
        return
    import torch
    import numpy as np

    build_data = ns['build_data']; train_eval = ns['train_eval']
    evaluate = ns['evaluate']; screening_benchmark = ns['screening_benchmark']
    unstable_breakdown = ns['unstable_breakdown']

    d = build_data(npz, split, dev='cpu')
    check("build_data works from notebook namespace",
          d['n_bus'] == 118 and d['ctx']['n_cell'] == 179,
          f"n_bus={d['n_bus']} n_cell={d['ctx']['n_cell']}")

    buf, old = io.StringIO(), sys.stdout
    try:
        sys.stdout = buf
        res, model = train_eval(d, seed=0, epochs=2, bs=64, verbose=False, amp=False)
    finally:
        sys.stdout = old
    check("train_eval runs", np.isfinite(res['acc']), f"acc={res['acc']:.3f}")
    check("params match the paper's model", res['params'] == 163592, str(res['params']))
    check("PR-AUC present for vulnerability head",
          'vuln_pr_auc' in res and np.isfinite(res['vuln_pr_auc']))

    gen_module = types.SimpleNamespace(
        build_net=ns['build_net'], risk_of=ns['risk_of'], PV_MODEL=ns['PV_MODEL'],
        K_CHOICES=ns['K_CHOICES'], V_SET_FLOOR=ns['V_SET_FLOOR'],
        V_SET_CEIL=ns['V_SET_CEIL'], NUMBA_PF=ns['NUMBA_PF'])
    try:
        sys.stdout = buf
        sb = screening_benchmark(model, d, gen_module, n_repeat=1, verbose=False)
        ub = unstable_breakdown(npz, gen_module)
    finally:
        sys.stdout = old
    check("screening benchmark runs", np.isfinite(sb['speedup_end_to_end']),
          f"{sb['speedup_end_to_end']:.1f}x end-to-end, "
          f"{sb['speedup_forward_only']:.1f}x forward-only")
    check("sweep scored against oracle",
          'sweep_false_safe' in sb and np.isfinite(sb['sweep_accuracy']),
          f"sweep acc={sb['sweep_accuracy']*100:.1f}%, "
          f"false-safe={sb['sweep_false_safe']}/{sb['sweep_n_unstable_true']}")
    check("unstable breakdown runs",
          ub['voltage_violation'] + ub['divergence_islanding'] + ub['divergence_connected']
          == ub['n_unstable'])


if __name__ == "__main__":
    print("=" * 72)
    print("NOTEBOOK VERIFICATION")
    print("=" * 72)

    nb1, codes1 = structural('gen_118_colab.ipynb')
    semantic('gen_118_colab.ipynb', codes1,
             ['build_net', 'generate', 'make_grouped_split', 'risk_of',
              'PV_MODEL', 'PV_BUSES', 'PV_TOTAL_MAX', 'LOAD_LO', 'LOAD_HI',
              'DV_LO', 'DV_HI', 'V_SET_FLOOR', 'V_SET_CEIL', 'NUMBA_PF'])

    nb2, codes2 = structural('train_118_colab.ipynb')
    ns2 = semantic('train_118_colab.ipynb', codes2,
                   ['build_net', 'CSGNN118', 'build_graph_ctx', 'cells_and_adj',
                    'build_data', 'train_eval', 'evaluate', 'screening_benchmark',
                    'unstable_breakdown', 'PV_MODEL', 'K_CHOICES'])
    functional(ns2)

    print("\n" + "=" * 72)
    print("NOTEBOOKS " + ("VERIFIED" if ok else "FAILED VERIFICATION"))
    print("=" * 72)
    sys.exit(0 if ok else 1)
