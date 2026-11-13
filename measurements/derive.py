"""Offline derivations behind the numbers on screen, recomputed from the notebook outputs, and checked.

Every number in the film that is not printed directly in a notebook output is computed here from those outputs:
calibration bins (act III, calibration), the probe and baseline AUCs, the "sure" subset, what the model says against
what the probe reads (act III, "It knows better than it says"), the two showcases, and the full next-word distributions.

    python derive.py                      # recompute from results/ and compare with the shipped files
    python derive.py --activations activations_llama8b_layers.npz
                                          # also refit the probes from the activations (release asset, ~2 min on a laptop)

Only numpy is needed. Exit code 0 when everything matches.
"""
import argparse
import io
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FAILS = []


def load(name):
    with io.open(os.path.join(RES, name), encoding='utf-8') as f:
        return json.load(f)


def check(label, got, want, tol=1e-9):
    """Compare numbers, lists and dicts recursively; record and print the outcome."""
    def same(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            return math.isclose(a, b, rel_tol=0, abs_tol=tol)
        return a == b
    ok = same(got, want)
    print(('  ok    ' if ok else '  DIFF  ') + label)
    if not ok:
        FAILS.append(label)
        print('        got :', json.dumps(got)[:300])
        print('        want:', json.dumps(want)[:300])


# ---------------------------------------------------------------- the same AUC as the notebook (rank-based, ties averaged)
def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    order = np.argsort(s, kind='mergesort'); ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    s_sorted = s[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n1 = float(y.sum()); n0 = float(len(y) - n1)
    return float('nan') if n1 == 0 or n0 == 0 else float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--activations', help='activations_llama8b_layers.npz (GitHub release asset) to refit the probes')
    a = ap.parse_args()

    items = [json.loads(l) for l in io.open(os.path.join(RES, 'e3_items.jsonl'), encoding='utf-8')]
    tq = [it for it in items if it['category'] == 'trivia']
    y = np.array([it['correct'] for it in tq])
    ent = np.array([it['first_token_entropy_bits'] for it in tq])
    mxp = np.array([it['first_token_max_prob'] for it in tq])
    cloud = load('h3_cloud.json')
    probe = np.array([p['probe'] for p in cloud['points'] if p['cat'] == 'trivia'], float)
    check('e3_items.jsonl: probe column = h3_cloud.json', [it['probe_oof_block28'] for it in tq], probe.tolist())

    # ------------------------------------------------------------ E2: calibration of the first answer token (act III, calibration)
    print('E2  calibration, e2_reliability_before.json')
    want = load('e2_reliability_before.json')
    bins = []
    k = np.minimum((mxp * 10).astype(int), 9)
    for i in range(10):
        m = k == i
        if m.any():
            bins.append({'lo': i * 0.1, 'hi': (i + 1) * 0.1, 'n': int(m.sum()), 'conf': float(mxp[m].mean()), 'acc': float(y[m].mean())})
    ece = sum(b['n'] / len(y) * abs(b['acc'] - b['conf']) for b in bins)
    check('bins (n, confidence, accuracy per tenth)', bins, want['bins'])
    check('ECE %.3f, mean confidence %.3f, accuracy %.2f' % (ece, mxp.mean(), y.mean()),
          [ece, float(mxp.mean()), float(y.mean())], [want['ece'], want['mean_conf'], want['accuracy']])

    # ------------------------------------------------------------ H3/E3: probe against the output shape (act III, the probe)
    print('E3  probe and baselines, e3_numbers.json / h3_metrics.json')
    want = load('e3_numbers.json')
    sure = ent < 1.0
    got = {'block': 28, 'n': len(y), 'probe_auc': auc(y, probe), 'entropy_auc': auc(y, -ent), 'max_prob_auc': auc(y, mxp),
           'sure_subset': {'definition': 'first-token entropy < 1 bit', 'n': int(sure.sum()),
                           'probe_auc': auc(y[sure], probe[sure]), 'entropy_auc': auc(y[sure], -ent[sure])},
           'ruler_at_zero_accuracy': float(((probe > 0).astype(int) == y).mean()), 'folds': 5}
    # exact: the AUCs below come out identical from the 4-decimal probe scores stored in h3_cloud.json
    check('probe %.3f, entropy %.3f, max-prob %.3f; when the output looks sure (n=%d): probe %.3f, entropy %.3f; sign of the probe right %.1f %%'
          % (got['probe_auc'], got['entropy_auc'], got['max_prob_auc'], got['sure_subset']['n'], got['sure_subset']['probe_auc'],
             got['sure_subset']['entropy_auc'], 100 * got['ruler_at_zero_accuracy']), got, want)
    hm = load('h3_metrics.json')
    check('h3_metrics.json: accuracy, entropy and max-prob AUC', [float(y.mean()), auc(y, -ent), auc(y, mxp)],
          [hm['accuracy_base'], hm['entropy_auc'], hm['max_prob_auc']])

    print('E3  showcase, e3_showcase.json (four right answers with the highest probe scores, four wrong ones with the lowest)')
    order = np.argsort(probe, kind='stable')
    hi = [i for i in order[::-1] if y[i] == 1][:4]
    lo = [i for i in order if y[i] == 0][:4]
    show = [{'question': tq[i]['question'], 'response': tq[i]['response'], 'gold': tq[i]['gold'][0] if tq[i]['gold'] else '',
             'correct': int(y[i]), 'probe_score': round(float(probe[i]), 3), 'entropy_bits': round(float(ent[i]), 3)} for i in hi + lo]
    check('the eight questions, answers and scores', show, load('e3_showcase.json'), tol=1e-3)

    # ------------------------------------------------------------ E5: what it says, against what the probe reads (act III, "It knows better than it says")
    print('E5  stated confidence against the probe, e5_analysis.json')
    e5 = load('e5_items.json')
    e5t = [it for it in e5 if it['category'] == 'trivia']
    assert [it['item_id'] for it in e5t] == [it['item_id'] for it in tq]
    said = np.array([np.nan if it['said'] is None else it['said'] for it in e5t])
    ok5 = np.array([it['correct'] for it in e5t])
    v = ~np.isnan(said)
    sv, yv, pv = said[v], ok5[v], probe[v]
    want = load('e5_analysis.json')
    got = {'n_valid': int(v.sum()), 'accuracy_e5': float(yv.mean()), 'said_auc': auc(yv, sv), 'probe_auc_same_items': auc(yv, pv),
           'corr_said_probe': float(np.corrcoef(sv, pv)[0, 1]), 'said_mean': float(sv.mean()),
           'said_mean_when_right': float(sv[yv == 1].mean()), 'said_mean_when_wrong': float(sv[yv == 0].mean()),
           'probe_mean_when_right': float(pv[yv == 1].mean()), 'probe_mean_when_wrong': float(pv[yv == 0].mean()),
           'probe_range': [round(float(pv.min()), 3), round(float(pv.max()), 3)]}
    s100 = sv == 100
    got['says_100'] = {'n': int(s100.sum()), 'accuracy': float(yv[s100].mean()), 'probe_auc_within': auc(yv[s100], pv[s100])}
    got['says_below_100'] = {'n': int((~s100).sum()), 'accuracy': float(yv[~s100].mean())}
    got['unsure_and_right'] = []
    got['unanswerable_said'] = sorted(float(it['said'] or 0.0) for it in e5 if it['category'] == 'unanswerable')
    check('valid %d; says %.1f on average (%.1f when right, %.1f when wrong); AUC of what it says %.2f, of the probe %.2f; says 100 on %d, right %.0f %%, probe within %.2f'
          % (got['n_valid'], got['said_mean'], got['said_mean_when_right'], got['said_mean_when_wrong'], got['said_auc'],
             got['probe_auc_same_items'], got['says_100']['n'], 100 * got['says_100']['accuracy'], got['says_100']['probe_auc_within']),
          got, {k: want[k] for k in want if k != 'source'})

    print('E5  unanswerable questions, e5_unanswerable_labels.csv (labels read by hand)')
    lab = {}
    for line in io.open(os.path.join(RES, 'e5_unanswerable_labels.csv'), encoding='utf-8').read().splitlines()[1:]:
        iid, said_, label = line.split(',')[:3]
        lab[iid] = (float(said_), label)
    un = [it for it in e5 if it['category'] == 'unanswerable']
    check('labels cover the 30 items, with the same stated confidence', [(it['item_id'], float(it['said'] or 0)) for it in un],
          [(k, lab[k][0]) for k in [it['item_id'] for it in un]])
    inv = [k for k, (s_, l) in lab.items() if l == 'invented']
    check('invented answers: %d of 30, %d of them with a confidence of 80 or more' % (len(inv), sum(lab[k][0] >= 80 for k in inv)),
          [len(inv), sum(lab[k][0] >= 80 for k in inv)], [7, 6])

    print('E3  unanswerable questions, neutral prompt, e3_unanswerable_labels.csv (labels read by hand)')
    lab3 = {}
    for line in io.open(os.path.join(RES, 'e3_unanswerable_labels.csv'), encoding='utf-8').read().splitlines()[1:]:
        iid, label = line.split(',')[:2]
        lab3[iid] = label
    un3 = [it for it in items if it['category'] == 'unanswerable']
    check('labels cover the 30 items', sorted(lab3), sorted(it['item_id'] for it in un3))
    inv3 = [k for k, l in lab3.items() if l == 'invented']
    check('refused or denied %d, invented %d' % (30 - len(inv3), len(inv3)), [30 - len(inv3), len(inv3)], [24, 6])

    print('E5  showcases, e5_showcase_all.json and e5_showcase.json')
    vi = np.where(v)[0]
    cand = [i for i in vi if said[i] == 100]
    low = sorted([i for i in cand if ok5[i] == 0], key=lambda i: probe[i])[:4]
    high = sorted([i for i in cand if ok5[i] == 1], key=lambda i: -probe[i])[:4]
    def row(i, kind):
        return {'question': e5t[i]['question'], 'response': e5t[i]['response'], 'gold': e5t[i]['gold'], 'correct': int(ok5[i]),
                'said': float(said[i]), 'probe_score': round(float(probe[i]), 3), 'kind': kind}
    allshow = [row(i, 'says 100 · probe low · wrong') for i in low] + [row(i, 'says 100 · probe high · right') for i in high]
    check('says 100: the four wrong answers with the lowest probe scores, the four right ones with the highest', allshow, load('e5_showcase_all.json'), tol=1e-3)
    screen = load('e5_showcase.json')
    check('the four on screen are among those eight', all(r in allshow for r in screen) and len(screen) == 4, True)

    # ------------------------------------------------------------ E1b: full next-word distributions (act III, the bars)
    print('E1b full distributions, e1b_*_full.json')
    for name in ('santiago', 'alpha3'):
        p = np.load(os.path.join(RES, 'e1b_%s_next.npy' % name)).astype(np.float64)
        want = load('e1b_%s_full.json' % name)
        top = np.sort(p)[::-1][:len(want['top'])]
        q = p[p > 0]
        got = {'n_vocab': int(p.size), 'top': [round(float(x), 7) for x in top], 'tail_mass': float(1.0 - top.sum()),
               'entropy_bits': float(-(q * np.log2(q)).sum())}
        check('%s: %d tokens in the vocabulary, top %d probabilities, tail %.3f, entropy %.2f bits'
              % (name, got['n_vocab'], len(got['top']), got['tail_mass'], got['entropy_bits']), got, want, tol=1e-3)

    # ------------------------------------------------------------ optional: refit the probes from the activations
    if a.activations:
        print('H3  refit from', os.path.basename(a.activations))
        z = np.load(a.activations, allow_pickle=True)
        tr = z['categories'] == 'trivia'; yt = z['y'][tr]
        scan = load('h3_layer_scan.json'); s28 = None
        for L in (8, 12, 16, 20, 24, 28):
            s = cv_scores(z['X_L%d' % L][tr].astype(np.float64), yt)
            sure_t = z['entropy'][tr] < 1.0
            check('block %2d: probe AUC %.3f, when the output looks sure %.3f' % (L, auc(yt, s), auc(yt[sure_t], s[sure_t])),
                  {'probe_auc': auc(yt, s), 'probe_auc_when_output_looks_sure': auc(yt[sure_t], s[sure_t])}, scan[str(L)], tol=1e-6)
            if L == 28:
                s28 = s
        check('block 28: out-of-sample probe scores = h3_cloud.json (4 decimals)', [round(float(x), 4) for x in s28], probe.tolist(), tol=1e-4)

    print('\n%s' % ('all checks passed' if not FAILS else '%d check(s) failed: %s' % (len(FAILS), '; '.join(FAILS))))
    sys.exit(1 if FAILS else 0)


# ---------------------------------------------------------------- the probe, exactly as in the notebook (§7)
def fit_logistic(X, y, l2=1.0, iters=600, lr=0.5):
    n, d = X.shape; w = np.zeros(d); b = 0.0; vw = np.zeros(d); vb = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30)))
        gw = X.T @ (p - y) / n + l2 * w / n; gb = float((p - y).mean())
        vw = 0.9 * vw + gw; vb = 0.9 * vb + gb; w -= lr * vw; b -= lr * vb
    return w, b


def standardize(Xtr, Xte):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd < 1e-6] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def cv_scores(X, y, k=5, seed=0, l2=1.0):
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); folds = np.empty(len(y), int)
    for cls in (0, 1):
        ci = idx[y == cls]; rng.shuffle(ci); folds[ci] = np.arange(len(ci)) % k
    out = np.empty(len(y))
    for f in range(k):
        trn, te = folds != f, folds == f
        Xtr, Xte = standardize(X[trn], X[te]); w, b = fit_logistic(Xtr, y[trn].astype(float), l2=l2)
        out[te] = Xte @ w + b
    return out


if __name__ == '__main__':
    main()
