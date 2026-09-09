"""
PROOF that enforcing both sides of Eq. (risk) changes NO published 27-bus label.

The released 27-bus generator labelled on V_min alone -- risk_of(vmin, conv) --
under an equation that states a two-sided criterion.  The gap is now closed in
code/gen_contingency_data.py.  This script re-labels the SHIPPED corpus with
the corrected two-sided rule and compares against the stored labels.

Expected: 0 differences, because the corpus never reaches 1.05 p.u.

    python verify_twosided_27bus.py
"""
import os
import numpy as np

# The shipped 27-bus corpus, one directory up in code/.
NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'code', 'contingency_data.npz')
V_HI = 1.05


def two_sided(vmin, vmax, conv):
    if not conv:
        return 2
    if vmin < 0.90 or vmax >= V_HI:
        return 2
    if vmin < 0.95:
        return 1
    return 0


def main():
    d = np.load(NPZ)
    vbus, div, stored = d['t_vbus'], d['t_div'], d['t_risk']
    conv = div < 0.5
    vmin = np.where(conv, vbus.min(axis=1), 0.0)
    vmax = np.where(conv, vbus.max(axis=1), 0.0)

    relabelled = np.array([two_sided(float(a), float(b), bool(c))
                           for a, b, c in zip(vmin, vmax, conv)], np.int64)
    diff = int((relabelled != stored).sum())

    print('27-bus corpus: %d scenarios (%d converged)' % (len(stored), int(conv.sum())))
    print('  highest bus voltage anywhere in the corpus : %.5f p.u.' % vmax.max())
    print('  scenarios with V_max >= %.2f                : %d' % (V_HI, int((vmax >= V_HI).sum())))
    print('  labels changed by enforcing both sides      : %d' % diff)
    c0 = np.bincount(stored, minlength=3)
    c1 = np.bincount(relabelled, minlength=3)
    print('  stored     S/M/U = %d/%d/%d' % tuple(c0))
    print('  re-labelled S/M/U = %d/%d/%d' % tuple(c1))
    if diff == 0:
        print('\nPASS: byte-identical. No published 27-bus result is affected.')
    else:
        raise SystemExit('FAIL: %d labels changed -- published results WOULD move' % diff)


if __name__ == '__main__':
    main()
