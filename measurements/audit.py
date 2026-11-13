"""Audit of the Llama measurements: how much do the film's numbers depend on the automatic labels, on chance, on the
choice of block, on the wording of the questions, and on comparing two prompts? See AUDIT.md for the findings.

    python audit.py                                         # labels, confidence intervals, text-only baseline, two prompts
    python audit.py --activations activations_llama8b_layers.npz   # also: refit on reviewed labels, fold seeds, nested CV

Only numpy is needed. The manual review of the labels is in results/label_review.csv.
"""
import argparse, csv, io, json, os, re, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, 'results')


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    order = np.argsort(s, kind='mergesort'); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    ss = s[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]: j += 1
        if j > i: ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(c, y):
    k = np.minimum((c * 10).astype(int), 9)
    return float(sum((k == i).sum() / len(y) * abs(y[k == i].mean() - c[k == i].mean()) for i in range(10) if (k == i).any()))


def boot(fn, n, B=2000, seed=1):
    rng = np.random.default_rng(seed); v = []
    for _ in range(B):
        b = rng.integers(0, n, n)
        try: v.append(fn(b))
        except ZeroDivisionError: pass
    v = np.array(v); v = v[~np.isnan(v)]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


# the probe, exactly as in the notebook
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


def folds_of(y, k=5, seed=0):
    r = np.random.default_rng(seed); idx = np.arange(len(y)); f = np.empty(len(y), int)
    for cls in (0, 1):
        ci = idx[y == cls]; r.shuffle(ci); f[ci] = np.arange(len(ci)) % k
    return f


def cv(X, y, k=5, seed=0):
    f = folds_of(y, k, seed); out = np.empty(len(y))
    for i in range(k):
        a, b = standardize(X[f != i], X[f == i]); w, c = fit_logistic(a, y[f != i].astype(float)); out[f == i] = b @ w + c
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--activations'); a = ap.parse_args()
    items = [json.loads(l) for l in io.open(os.path.join(RES, 'e3_items.jsonl'), encoding='utf-8')]
    e5 = {it['item_id']: it for it in json.load(io.open(os.path.join(RES, 'e5_items.json'), encoding='utf-8'))}
    tq = [it for it in items if it['category'] == 'trivia']; ids = [it['item_id'] for it in tq]
    y0 = np.array([it['correct'] for it in tq]); ent = np.array([it['first_token_entropy_bits'] for it in tq])
    mxp = np.array([it['first_token_max_prob'] for it in tq]); probe = np.array([it['probe_oof_block28'] for it in tq])
    said = np.array([np.nan if e5[i]['said'] is None else e5[i]['said'] for i in ids]); y5 = np.array([e5[i]['correct'] for i in ids])
    rev = list(csv.DictReader(io.open(os.path.join(RES, 'label_review.csv'), encoding='utf-8')))
    def reviewed(y, prompt, statuses):
        y = y.copy()
        for r in rev:
            if r['prompt'] == prompt and r['status'] in statuses: y[ids.index(r['item_id'])] = int(r['reviewed_label'])
        return y
    LAB3 = {'automatic': y0, 'reviewed': reviewed(y0, 'E3', {'clear'}), 'reviewed, debatable as well': reviewed(y0, 'E3', {'clear', 'debatable'})}
    LAB5 = {'automatic': y5, 'reviewed': reviewed(y5, 'E5', {'clear'}), 'reviewed, debatable as well': reviewed(y5, 'E5', {'clear', 'debatable'})}
    sure = ent < 1.0; v = ~np.isnan(said)

    print('1. Labels: %d reviewed decisions (%s)' % (len(rev), ', '.join('%s %s: %d' % (p, s, sum(r['prompt'] == p and r['status'] == s for r in rev)) for p in ('E3', 'E5') for s in ('clear', 'debatable'))))
    for k, y in LAB3.items():
        print('   E3, %-28s accuracy %.3f  ECE %.3f  probe %.3f  entropy %.3f  max-prob %.3f  | looks sure (n=%d, %d wrong): probe %.3f  entropy %.3f'
              % (k, y.mean(), ece(mxp, y), auc(y, probe), auc(y, -ent), auc(y, mxp), sure.sum(), (y[sure] == 0).sum(), auc(y[sure], probe[sure]), auc(y[sure], -ent[sure])))
    for k, y in LAB5.items():
        yv, sv, pv = y[v], said[v], probe[v]; h = sv == 100
        print('   E5, %-28s accuracy %.3f  stated confidence %.3f  probe %.3f  | says 100 (n=%d): right %.3f  probe %.3f'
              % (k, yv.mean(), auc(yv, sv), auc(yv, pv), h.sum(), yv[h].mean(), auc(yv[h], pv[h])))

    print('2. 95 % bootstrap intervals (2,000 resamples, reviewed labels)')
    y = LAB3['reviewed']; n = len(y); si = np.where(sure)[0]
    vi = np.where(v)[0]; yv = LAB5['reviewed'][vi]; sv = said[vi]; pv = probe[vi]; h = np.where(sv == 100)[0]
    rows = [('E3 probe', lambda b: auc(y[b], probe[b]), n), ('E3 first-token entropy', lambda b: auc(y[b], -ent[b]), n),
            ('E3 probe minus entropy', lambda b: auc(y[b], probe[b]) - auc(y[b], -ent[b]), n),
            ('looks sure: probe', lambda b: auc(y[si][b], probe[si][b]), len(si)), ('looks sure: entropy', lambda b: auc(y[si][b], -ent[si][b]), len(si)),
            ('looks sure: probe minus entropy', lambda b: auc(y[si][b], probe[si][b]) - auc(y[si][b], -ent[si][b]), len(si)),
            ('E5 stated confidence', lambda b: auc(yv[b], sv[b]), len(vi)), ('E5 probe', lambda b: auc(yv[b], pv[b]), len(vi)),
            ('E5 probe minus stated confidence', lambda b: auc(yv[b], pv[b]) - auc(yv[b], sv[b]), len(vi)),
            ('says 100: probe', lambda b: auc(yv[h][b], pv[h][b]), len(h)), ('says 100: accuracy', lambda b: yv[h][b].mean(), len(h)),
            ('E2 expected calibration error', lambda b: ece(mxp[b], y[b]), n)]
    for lab, fn, m in rows:
        lo, hi = boot(fn, m); print('   %-34s %.3f  [%.3f, %.3f]' % (lab, fn(np.arange(m)), lo, hi))

    print('3. Wording alone: a probe on the words of the question (hashed words and word pairs), and the question length')
    def feats(q, D=4096):
        w = re.findall(r"[a-z0-9']+", q.lower()); vec = np.zeros(D)
        for g in w + [x + '_' + z for x, z in zip(w, w[1:])]: vec[zlib.crc32(g.encode()) % D] = 1.0
        return vec
    Q = np.stack([feats(it['question']) for it in tq]); ql = np.array([len(it['question']) for it in tq])
    print('   question words %.3f ; question length %.3f' % (auc(y0, cv(Q, y0)), max(auc(y0, ql), auc(y0, -ql))))

    print('4. Two prompts: the probe reads the neutral prompt (E3); the stated confidence comes from the E5 prompt')
    print('   E3 and E5 correctness agree on %.1f %% of the %d items ; probe against E3 correctness %.3f, against E5 correctness %.3f ; entropy against E5 correctness %.3f'
          % (100 * (y0[v] == y5[v]).mean(), v.sum(), auc(y0[v], probe[v]), auc(y5[v], probe[v]), auc(y5[v], -ent[v])))

    if a.activations:
        z = np.load(a.activations, allow_pickle=True); tr = z['categories'] == 'trivia'; assert list(z['item_ids'][tr]) == ids
        XL = {L: z['X_L%d' % L][tr].astype(np.float64) for L in (8, 12, 16, 20, 24, 28)}
        yr = LAB3['reviewed']; s1 = cv(XL[28], yr)
        print('5. Probe refit on the reviewed labels (block 28): %.3f ; looks sure %.3f ; against E5 (reviewed) %.3f, stated confidence %.3f'
              % (auc(yr, s1), auc(yr[sure], s1[sure]), auc(LAB5['reviewed'][v], s1[v]), auc(LAB5['reviewed'][v], said[v])))
        print('6. Fold seeds 1 to 5 (block 28): ' + ', '.join('%.3f' % auc(y0, cv(XL[28], y0, seed=s)) for s in range(1, 6)))
        fo = folds_of(y0); out = np.empty(len(y0)); picks = []
        for i in range(5):
            trn = np.where(fo != i)[0]; te = np.where(fo == i)[0]
            best = max(XL, key=lambda L: auc(y0[trn], cv(XL[L][trn], y0[trn], k=4, seed=1))); picks.append(best)
            p_, q_ = standardize(XL[best][trn], XL[best][te]); w, c = fit_logistic(p_, y0[trn].astype(float)); out[te] = q_ @ w + c
        print('7. Nested cross-validation, the block chosen inside each training fold: blocks %s, AUC %.3f' % (picks, auc(y0, out)))


if __name__ == '__main__':
    main()
