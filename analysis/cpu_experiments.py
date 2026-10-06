"""
CPU experiments for the IASC revision (no GPU needed).
Uses saved per-sentence logits in res/preds_<model>.npz.

E1  in-domain metrics (recomputed, sanity check vs table4)
E2  full metrics on the human-labelled target set (892) + TF-IDF baseline
E3  per-language in-domain metrics
E4  label cardinality analysis + chance baselines for primary-aspect accuracy
E5  threshold sensitivity on the target set (reviewer ICIC #7)
E6  bootstrap CIs: in-domain vs target, and vs Vietnamese-only in-domain subset
E7  per-aspect F1 + error-type decomposition (missed vs spurious labels)
E8  silver-label ablation with cardinality-matched subgroup

    python analysis/cpu_experiments.py  -> results/analysis_cpu/*.csv
"""
import json, os, time
import numpy as np, pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             average_precision_score, label_ranking_average_precision_score,
                             roc_auc_score)
from sklearn.model_selection import KFold
from statsmodels.stats.contingency_tables import mcnemar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "analysis_cpu"); os.makedirs(OUT, exist_ok=True)
L = ['overall', 'amenities', 'food', 'service', 'location', 'ambiance', 'room', 'price', 'cleanliness']
MODELS = ['mBERT', 'XLM-R', 'mT5']
RNG = np.random.default_rng(42)
B = 10000
summary = {}

sig = lambda z: 1 / (1 + np.exp(-z))
te = pd.read_csv(f"{ROOT}/vn_hotel_res_aspect/test.csv")
tr_df = pd.read_csv(f"{ROOT}/vn_hotel_res_aspect/train.csv")
va_df = pd.read_csv(f"{ROOT}/vn_hotel_res_aspect/validation.csv")
tt = pd.read_csv(f"{ROOT}/data_unilever/splits/transfer_test.csv")
P = {m: np.load(f"{ROOT}/results/seed42_original/preds_{m}.npz") for m in MODELS}
S = {m: np.load(f"{ROOT}/results/seed42_original/preds_silverabl_{m}.npz") for m in MODELS}
Yi, Yt = te[L].values, tt[L].values


def metrics(Y, Yh, prob=None):
    r = dict(microP=precision_score(Y, Yh, average='micro', zero_division=0),
             microR=recall_score(Y, Yh, average='micro', zero_division=0),
             microF1=f1_score(Y, Yh, average='micro', zero_division=0),
             macroF1=f1_score(Y, Yh, average='macro', zero_division=0),
             subset=(Y == Yh).all(1).mean(),
             pred_card=Yh.sum(1).mean())
    if prob is not None:
        r['primary'] = Y[np.arange(len(Y)), prob.argmax(1)].mean()
    return r


def primary_vec(Y, prob):
    return Y[np.arange(len(Y)), prob.argmax(1)].astype(int)


def subset_vec(Y, Yh):
    return (Y == Yh).all(1).astype(int)


def mcn(a, b):
    t = [[((a == 1) & (b == 1)).sum(), ((a == 1) & (b == 0)).sum()],
         [((a == 0) & (b == 1)).sum(), ((a == 0) & (b == 0)).sum()]]
    r = mcnemar(t, exact=(t[0][1] + t[1][0]) < 25, correction=True)
    return float(r.pvalue)


# ---------------- TF-IDF + LinearSVC multi-label baseline (OvR), same features as src/baseline_tfidf.py
def tfidf_baseline():
    """Exactly the notebook baseline: char_wb(2,5)+word(1,2) TF-IDF, OvR LinearSVC(C=0.5),
    per-label decision thresholds tuned on VALIDATION (grid -1.2..0.6)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_union
    from sklearn.svm import LinearSVC
    feats = make_union(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300_000, sublinear_tf=True),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, sublinear_tf=True))
    Xtr = feats.fit_transform(tr_df.sentence.astype(str))
    Xte = feats.transform(te.sentence.astype(str)); Xtt = feats.transform(tt.sentence.astype(str))
    Xva = feats.transform(va_df.sentence.astype(str))
    sc = {k: np.zeros((X.shape[0], 9)) for k, X in [('te', Xte), ('tt', Xtt), ('va', Xva)]}
    for j, a in enumerate(L):
        c = LinearSVC(C=0.5).fit(Xtr, tr_df[a].values); print('svm', a, flush=True)
        sc['te'][:, j] = c.decision_function(Xte); sc['tt'][:, j] = c.decision_function(Xtt)
        sc['va'][:, j] = c.decision_function(Xva)
    Yv = va_df[L].values
    grid = np.arange(-1.2, 0.61, 0.02)
    sc['thr'] = np.array([grid[int(np.argmax([f1_score(Yv[:, j], (sc['va'][:, j] > t).astype(int), zero_division=0)
                                              for t in grid]))] for j in range(9)])
    return sc


def lrap_fast(Y, p):
    """vectorised label ranking average precision (scores continuous, ties negligible)"""
    order = np.argsort(-p, 1)
    Ys = np.take_along_axis(Y, order, 1)
    prec = np.cumsum(Ys, 1) / np.arange(1, Y.shape[1] + 1)
    num = (prec * Ys).sum(1); den = Ys.sum(1); ok = den > 0
    return (num[ok] / den[ok]).mean()


t0 = time.time()
if os.path.exists(f"{OUT}/preds_tfidf.npz"):
    tf = dict(np.load(f"{OUT}/preds_tfidf.npz"))
else:
    tf = tfidf_baseline()
    np.savez_compressed(f"{OUT}/preds_tfidf.npz", **tf)
print(f"tfidf done {time.time()-t0:.0f}s", flush=True)

# probabilities / decisions per system
SYS = {}
for m in MODELS:
    SYS[m] = dict(pi=sig(P[m]['logit']), pt=sig(P[m]['logit_t']), thr=P[m]['thr'])
# TF-IDF: decision scores, threshold 0 (default SVM decision) ; "prob" = scores for ranking
SYS['TF-IDF+SVM'] = dict(pi=tf['te'], pt=tf['tt'], thr=tf['thr'])
ALL = ['TF-IDF+SVM'] + MODELS

# ---------------- E1 / E2
rows = []
for m in ALL:
    s = SYS[m]
    for split, Y, p in [('in-domain (vn_hotel test, n=7075)', Yi, s['pi']), ('target (human-labelled, n=892)', Yt, s['pt'])]:
        rows.append(dict(model=m, split=split, **metrics(Y, (p > s['thr']).astype(int), p)))
e12 = pd.DataFrame(rows); e12.to_csv(f"{OUT}/E1E2_full_metrics.csv", index=False)
print(e12.round(4).to_string())

# McNemar on target (subset + primary), Bonferroni over the 3 transformer pairs
pairs = [('mBERT', 'XLM-R'), ('mBERT', 'mT5'), ('XLM-R', 'mT5')]
mrows = []
for split, Y, key in [('in-domain', Yi, 'pi'), ('target', Yt, 'pt')]:
    for a, b in pairs:
        ya = (SYS[a][key] > SYS[a]['thr']).astype(int); yb = (SYS[b][key] > SYS[b]['thr']).astype(int)
        mrows.append(dict(split=split, pair=f"{a} vs {b}",
                          subset_a=subset_vec(Y, ya).mean(), subset_b=subset_vec(Y, yb).mean(),
                          p_subset=mcn(subset_vec(Y, ya), subset_vec(Y, yb)),
                          primary_a=primary_vec(Y, SYS[a][key]).mean(), primary_b=primary_vec(Y, SYS[b][key]).mean(),
                          p_primary=mcn(primary_vec(Y, SYS[a][key]), primary_vec(Y, SYS[b][key]))))
pd.DataFrame(mrows).to_csv(f"{OUT}/E2_mcnemar.csv", index=False)

# paired bootstrap of micro-F1 differences on target
def paired_boot_f1(Y, ya, yb, n=B):
    N = len(Y); d = np.empty(n)
    for i in range(n):
        ix = RNG.integers(0, N, N)
        d[i] = f1_score(Y[ix], ya[ix], average='micro') - f1_score(Y[ix], yb[ix], average='micro')
    return np.percentile(d, [2.5, 97.5]), (d <= 0).mean()

brows = []
for a, b in pairs:
    ya = (SYS[a]['pt'] > SYS[a]['thr']).astype(int); yb = (SYS[b]['pt'] > SYS[b]['thr']).astype(int)
    ci, pneg = paired_boot_f1(Yt, ya, yb, 2000)
    brows.append(dict(pair=f"{a} - {b}", diff=f1_score(Yt, ya, average='micro') - f1_score(Yt, yb, average='micro'),
                      ci_lo=ci[0], ci_hi=ci[1]))
pd.DataFrame(brows).to_csv(f"{OUT}/E2_paired_bootstrap_microF1_target.csv", index=False)

print(' E3', time.time()-t0, flush=True)
# ---------------- E3 per language
rows = []
for m in ALL:
    s = SYS[m]
    for lg in ['en', 'vi', 'zh', 'ja']:
        ix = (te.language == lg).values
        rows.append(dict(model=m, lang=lg, n=ix.sum(), gold_card=Yi[ix].sum(1).mean(),
                         **metrics(Yi[ix], (s['pi'][ix] > s['thr']).astype(int), s['pi'][ix])))
e3 = pd.DataFrame(rows); e3.to_csv(f"{OUT}/E3_per_language.csv", index=False)

print(' E4', time.time()-t0, flush=True)
# ---------------- E4 cardinality
def card_stats(Y, name, extra=None):
    c = Y.sum(1)
    d = dict(set=name, n=len(Y), mean_card=c.mean(), pct_1=(c == 1).mean(), pct_2=(c == 2).mean(),
             pct_ge3=(c >= 3).mean(), pct_0=(c == 0).mean())
    # chance primary accuracy: random label = mean(card)/9 ; majority-label primary = freq of most common label
    d['chance_primary_random'] = (c / 9).mean()
    d['majority_label'] = L[Y.sum(0).argmax()]; d['chance_primary_majority'] = Y[:, Y.sum(0).argmax()].mean()
    return d

crow = [card_stats(tr_df[L].values, 'vn_hotel train'), card_stats(Yi, 'vn_hotel test'),
        card_stats(Yi[(te.language == 'vi').values], 'vn_hotel test (vi)'), card_stats(Yt, 'target (human)')]
# length-matched comparison (vi only, by token count bucket)
vi = te[te.language == 'vi'].copy(); vi['len'] = vi.sentence.str.split().str.len(); vi['card'] = vi[L].sum(1)
tt2 = tt.copy(); tt2['len'] = tt2.sentence.str.split().str.len(); tt2['card'] = tt2[L].sum(1)
bins = [0, 5, 10, 15, 20, 30, 1000]
vi['b'] = pd.cut(vi.len, bins); tt2['b'] = pd.cut(tt2.len, bins)
lm = pd.DataFrame({'vn_hotel_vi_card': vi.groupby('b', observed=False).card.mean(), 'vn_hotel_vi_n': vi.groupby('b', observed=False).size(),
                   'target_card': tt2.groupby('b', observed=False).card.mean(), 'target_n': tt2.groupby('b', observed=False).size()})
lm.to_csv(f"{OUT}/E4_cardinality_by_length.csv")
pd.DataFrame(crow).to_csv(f"{OUT}/E4_cardinality.csv", index=False)
# primary accuracy by gold cardinality on target
rows = []
for m in ALL:
    pv = primary_vec(Yt, SYS[m]['pt']); c = Yt.sum(1)
    for k, mask in [('1', c == 1), ('2', c == 2), ('3', c == 3), ('>=4', c >= 4)]:
        rows.append(dict(model=m, gold_card=k, n=mask.sum(), primary=pv[mask].mean(), chance=(c[mask] / 9).mean()))
pd.DataFrame(rows).to_csv(f"{OUT}/E4_primary_by_cardinality.csv", index=False)

print(' E5', time.time()-t0, flush=True)
# ---------------- E5 threshold sensitivity on target
def cv_thresholds(Y, p, k=5, grid=np.linspace(0.02, 0.98, 49), seed=42):
    """tune per-label thresholds on k-1 folds of the TARGET set, apply to held-out fold"""
    Yh = np.zeros_like(Y)
    for trn, tst in KFold(k, shuffle=True, random_state=seed).split(Y):
        for j in range(Y.shape[1]):
            best, bt = -1, 0.5
            for t in grid:
                f = f1_score(Y[trn, j], (p[trn, j] >= t).astype(int), zero_division=0)
                if f > best: best, bt = f, t
            Yh[tst, j] = (p[tst, j] >= bt).astype(int)
    return Yh

def cv_global(Y, p, k=5, grid=np.linspace(0.02, 0.98, 49), seed=42):
    Yh = np.zeros_like(Y)
    for trn, tst in KFold(k, shuffle=True, random_state=seed).split(Y):
        bt = max(grid, key=lambda t: f1_score(Y[trn], (p[trn] >= t).astype(int), average='micro', zero_division=0))
        Yh[tst] = (p[tst] >= bt).astype(int)
    return Yh

def topk(p, k):
    Yh = np.zeros_like(p, dtype=int)
    idx = np.argsort(-p, 1)
    for i in range(len(p)):
        Yh[i, idx[i, :k[i]]] = 1
    return Yh

rows = []
for m in ALL:
    s = SYS[m]; p = s['pt']
    is_prob = m != 'TF-IDF+SVM'
    pp = p if is_prob else sig(p)  # map SVM scores to (0,1) for threshold grids
    variants = {
        'source-tuned thresholds (main protocol)': (p > s['thr']).astype(int),
        'fixed 0.5' if is_prob else 'fixed (SVM score 0)': (pp > 0.5).astype(int),
        'target-tuned global threshold (5-fold CV)': cv_global(Yt, pp),
        'target-tuned per-label thresholds (5-fold CV)': cv_thresholds(Yt, pp),
        'top-k, k = gold cardinality (oracle)': topk(pp, Yt.sum(1)),
        'top-k, k = 2 (target mean)': topk(pp, np.full(len(pp), 2)),
    }
    for v, Yh in variants.items():
        rows.append(dict(model=m, decoding=v, **metrics(Yt, Yh, pp)))
    rows.append(dict(model=m, decoding='threshold-free',
                     microPRAUC=average_precision_score(Yt, pp, average='micro'),
                     macroPRAUC=average_precision_score(Yt, pp, average='macro'),
                     LRAP=lrap_fast(Yt, pp),
                     macroROCAUC=roc_auc_score(Yt, pp, average='macro')))
    # same threshold-free on in-domain for reference
    ppi = s['pi'] if is_prob else sig(s['pi'])
    rows.append(dict(model=m, decoding='threshold-free (in-domain ref.)',
                     microPRAUC=average_precision_score(Yi, ppi, average='micro'),
                     macroPRAUC=average_precision_score(Yi, ppi, average='macro'),
                     LRAP=lrap_fast(Yi, ppi),
                     macroROCAUC=roc_auc_score(Yi, ppi, average='macro')))
e5 = pd.DataFrame(rows); e5.to_csv(f"{OUT}/E5_threshold_sensitivity_target.csv", index=False)

# bootstrap CI for LRAP differences between transformers (paired)
def paired_boot_metric(fn, Y, pa, pb, n=1000):
    N = len(Y); d = np.empty(n)
    for i in range(n):
        ix = RNG.integers(0, N, N); d[i] = fn(Y[ix], pa[ix]) - fn(Y[ix], pb[ix])
    return np.percentile(d, [2.5, 97.5])
rows = []
for a, b in pairs:
    for nm, fn in [('LRAP', lrap_fast),
                   ('microPRAUC', lambda Y, p: average_precision_score(Y, p, average='micro'))]:
        ci = paired_boot_metric(fn, Yt, SYS[a]['pt'], SYS[b]['pt'])
        rows.append(dict(pair=f"{a} - {b}", metric=nm, diff=fn(Yt, SYS[a]['pt']) - fn(Yt, SYS[b]['pt']), ci_lo=ci[0], ci_hi=ci[1]))
pd.DataFrame(rows).to_csv(f"{OUT}/E5_paired_bootstrap_threshold_free.csv", index=False)

print(' E6', time.time()-t0, flush=True)
# ---------------- E6 bootstrap (independent samples) in-domain vs target primary acc
rows = []
vi_mask = (te.language == 'vi').values
for m in ALL:
    a_all = primary_vec(Yi, SYS[m]['pi']); a_vi = a_all[vi_mask]; b = primary_vec(Yt, SYS[m]['pt'])
    for ref_name, a in [('all languages', a_all), ('Vietnamese only', a_vi)]:
        d = np.array([RNG.choice(a, len(a)).mean() - RNG.choice(b, len(b)).mean() for _ in range(B)])
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append(dict(model=m, reference=ref_name, in_domain=a.mean(), target=b.mean(), diff=a.mean() - b.mean(),
                         ci_lo=lo, ci_hi=hi, p_two_sided=2 * min((d <= 0).mean(), (d >= 0).mean())))
pd.DataFrame(rows).to_csv(f"{OUT}/E6_bootstrap_primary_indomain_vs_target.csv", index=False)

print(' E7', time.time()-t0, flush=True)
# ---------------- E7 per-aspect F1 and error types
rows = []
for m in ALL:
    s = SYS[m]
    for split, Y, p in [('in-domain', Yi, s['pi']), ('target', Yt, s['pt'])]:
        Yh = (p > s['thr']).astype(int)
        f = f1_score(Y, Yh, average=None, zero_division=0)
        pr = precision_score(Y, Yh, average=None, zero_division=0); rc = recall_score(Y, Yh, average=None, zero_division=0)
        for j, a in enumerate(L):
            rows.append(dict(model=m, split=split, aspect=a, support=int(Y[:, j].sum()), P=pr[j], R=rc[j], F1=f[j]))
pd.DataFrame(rows).to_csv(f"{OUT}/E7_per_aspect.csv", index=False)
rows = []
for m in ALL:
    s = SYS[m]
    for split, Y, p in [('in-domain', Yi, s['pi']), ('target', Yt, s['pt'])]:
        Yh = (p > s['thr']).astype(int)
        miss = ((Y == 1) & (Yh == 0)).sum(); spur = ((Y == 0) & (Yh == 1)).sum()
        wrong = ~(Y == Yh).all(1)
        only_miss = (wrong & ~((Y == 0) & (Yh == 1)).any(1)).sum()
        only_spur = (wrong & ~((Y == 1) & (Yh == 0)).any(1)).sum()
        rows.append(dict(model=m, split=split, missed_labels=int(miss), spurious_labels=int(spur),
                         missed_share=miss / (miss + spur), wrong_sentences=int(wrong.sum()),
                         only_missing=int(only_miss), only_spurious=int(only_spur),
                         both=int(wrong.sum() - only_miss - only_spur)))
pd.DataFrame(rows).to_csv(f"{OUT}/E7_error_types.csv", index=False)

print(' E8', time.time()-t0, flush=True)
# ---------------- E8 silver ablation (+ cardinality-2 subgroup)
rows = []
c2 = Yt.sum(1) == 2
for m in MODELS:
    for src, z, thr in [('vn_hotel (clean, natural cardinality)', P[m]['logit_t'], P[m]['thr']),
                        ('target-domain silver (forced 2 labels)', S[m]['logit'], S[m]['thr'])]:
        p = sig(z); Yh = (p > thr).astype(int)
        rows.append(dict(model=m, train_source=src, eval_subset='all 892', **metrics(Yt, Yh, p)))
        rows.append(dict(model=m, train_source=src, eval_subset='gold cardinality = 2 (n=%d)' % c2.sum(), **metrics(Yt[c2], Yh[c2], p[c2])))
pd.DataFrame(rows).to_csv(f"{OUT}/E8_silver_ablation.csv", index=False)
rows = []
for m in MODELS:
    pa, pb = sig(P[m]['logit_t']), sig(S[m]['logit'])
    ya, yb = (pa > P[m]['thr']).astype(int), (pb > S[m]['thr']).astype(int)
    for sub, mask in [('all', np.ones(len(Yt), bool)), ('card=2', c2)]:
        rows.append(dict(model=m, eval_subset=sub, p_subset=mcn(subset_vec(Yt[mask], ya[mask]), subset_vec(Yt[mask], yb[mask])),
                         p_primary=mcn(primary_vec(Yt[mask], pa[mask]), primary_vec(Yt[mask], pb[mask]))))
pd.DataFrame(rows).to_csv(f"{OUT}/E8_silver_mcnemar.csv", index=False)

print("done", time.time() - t0)
