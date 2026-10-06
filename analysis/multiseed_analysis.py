"""
Re-run every diagnostic of cpu_experiments.py on the 3-seed GPU runs (seeds 13, 42, 2026).
Main configuration = TF-IDF fusion, 10-epoch budget.  Output: results/analysis_3seed/*.csv

    python analysis/multiseed_analysis.py --runs results/gpu_runs
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from sklearn.model_selection import KFold
from statsmodels.stats.contingency_tables import mcnemar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L = ['overall', 'amenities', 'food', 'service', 'location', 'ambiance', 'room', 'price', 'cleanliness']
MODELS = ['mBERT', 'XLM-R', 'mT5']; SEEDS = [13, 42, 2026]
EXT = ['semeval15_en_hotel', 'semeval16_en_rest', 'semeval16_nl_rest']
sig = lambda z: 1 / (1 + np.exp(-z))
RNG = np.random.default_rng(7)
ap = argparse.ArgumentParser(); ap.add_argument('--runs', default=f'{ROOT}/results/gpu_runs')
ap.add_argument('--out', default=f'{ROOT}/results/analysis_3seed'); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
OUT = a.out

te = pd.read_csv(f'{ROOT}/vn_hotel_res_aspect/test.csv'); tt = pd.read_csv(f'{ROOT}/data_unilever/splits/transfer_test.csv')
ov = pd.read_csv(f'{ROOT}/data/test_review_overlap.csv')
Yi, Yt = te[L].values, tt[L].values
tf = np.load(f'{ROOT}/results/analysis_cpu/preds_tfidf.npz')


def load(model, seed, variant='tfidf', ep=10):
    d = np.load(f'{a.runs}/{model}_{variant}_e{ep}_s{seed}.npz')
    if 'test_Y' in d.files:
        assert (d['test_Y'] == Yi).all() and (d['transfer_test_Y'] == Yt).all()
    return dict(pi=sig(d['test_logit']), pt=sig(d['transfer_test_logit']), thr=d['thr'],
                **{e: (sig(d[f'{e}_logit']), d[f'{e}_Y'] if f'{e}_Y' in d.files else pd.read_csv(f'{ROOT}/data_external/{e}.csv')[L].values) for e in EXT})


R = {(m, s): load(m, s) for m in MODELS for s in SEEDS}


def met(Y, Yh, p=None):
    r = dict(microP=precision_score(Y, Yh, average='micro', zero_division=0), microR=recall_score(Y, Yh, average='micro', zero_division=0),
             microF1=f1_score(Y, Yh, average='micro', zero_division=0), macroF1=f1_score(Y, Yh, average='macro', zero_division=0),
             subset=(Y == Yh).all(1).mean(), pred_card=Yh.sum(1).mean())
    if p is not None:
        r['primary'] = Y[np.arange(len(Y)), p.argmax(1)].mean()
    return r


def lrap(Y, p):
    o = np.argsort(-p, 1); Ys = np.take_along_axis(Y, o, 1); prec = np.cumsum(Ys, 1) / np.arange(1, Y.shape[1] + 1)
    den = Ys.sum(1); ok = den > 0
    return ((prec * Ys).sum(1)[ok] / den[ok]).mean()


def agg(rows, keys):
    df = pd.DataFrame(rows); num = [c for c in df.columns if c not in keys + ['seed'] and df[c].dtype != object]
    g = df.groupby(keys, sort=False)[num]
    m, s = g.mean(), g.std(ddof=1)
    out = m.copy()
    for c in num:
        out[c + '_sd'] = s[c]
    return out.reset_index()


prim = lambda Y, p: Y[np.arange(len(Y)), p.argmax(1)].astype(int)
subs = lambda Y, Yh: (Y == Yh).all(1).astype(int)

# ---------------- M1 main metrics
rows = []
for (m, s), r in R.items():
    rows.append(dict(model=m, seed=s, split='in-domain', **met(Yi, (r['pi'] > r['thr']).astype(int), r['pi'])))
    rows.append(dict(model=m, seed=s, split='target', **met(Yt, (r['pt'] > r['thr']).astype(int), r['pt'])))
M1 = agg(rows, ['split', 'model'])
for split, Y, sc in [('in-domain', Yi, tf['te']), ('target', Yt, tf['tt'])]:
    M1 = pd.concat([M1, pd.DataFrame([dict(split=split, model='TF-IDF+SVM', **met(Y, (sc > tf['thr']).astype(int), sc))])])
M1.to_csv(f'{OUT}/M1_main.csv', index=False); pd.DataFrame(rows).to_csv(f'{OUT}/M1_per_seed.csv', index=False)

# McNemar per seed (Bonferroni over 3 pairs)
rows = []
for split, Y, k in [('in-domain', Yi, 'pi'), ('target', Yt, 'pt')]:
    for x, y in [('mBERT', 'XLM-R'), ('mBERT', 'mT5'), ('XLM-R', 'mT5')]:
        for s in SEEDS:
            rx, ry = R[(x, s)], R[(y, s)]
            for nm, fx, fy in [('subset', subs(Y, (rx[k] > rx['thr']).astype(int)), subs(Y, (ry[k] > ry['thr']).astype(int))),
                               ('primary', prim(Y, rx[k]), prim(Y, ry[k]))]:
                t = [[((fx == 1) & (fy == 1)).sum(), ((fx == 1) & (fy == 0)).sum()], [((fx == 0) & (fy == 1)).sum(), ((fx == 0) & (fy == 0)).sum()]]
                p = mcnemar(t, exact=(t[0][1] + t[1][0]) < 25, correction=True).pvalue
                rows.append(dict(split=split, pair=f'{x} vs {y}', metric=nm, seed=s, acc_a=fx.mean(), acc_b=fy.mean(), p=p, sig=p < 0.0167))
mc = pd.DataFrame(rows); mc.to_csv(f'{OUT}/M1_mcnemar_per_seed.csv', index=False)
mc.groupby(['split', 'pair', 'metric']).agg(mean_a=('acc_a', 'mean'), mean_b=('acc_b', 'mean'), seeds_sig=('sig', 'sum'),
                                             p_max=('p', 'max')).reset_index().to_csv(f'{OUT}/M1_mcnemar_summary.csv', index=False)

# paired bootstrap of seed-averaged micro-F1 differences on target
def boot_pair_f1(Y, key, x, y, n=1000):
    N = len(Y); d = np.empty(n)
    Yh = {mm: [(R[(mm, s)][key] > R[(mm, s)]['thr']).astype(int) for s in SEEDS] for mm in (x, y)}
    for i in range(n):
        ix = RNG.integers(0, N, N)
        d[i] = np.mean([f1_score(Y[ix], h[ix], average='micro') for h in Yh[x]]) - np.mean([f1_score(Y[ix], h[ix], average='micro') for h in Yh[y]])
    full = np.mean([f1_score(Y, h, average='micro') for h in Yh[x]]) - np.mean([f1_score(Y, h, average='micro') for h in Yh[y]])
    return full, *np.percentile(d, [2.5, 97.5])
rows = []
for split, Y, k in [('in-domain', Yi, 'pi'), ('target', Yt, 'pt')]:
    for x, y in [('XLM-R', 'mBERT'), ('XLM-R', 'mT5'), ('mBERT', 'mT5')]:
        f, lo, hi = boot_pair_f1(Y, k, x, y, 500 if split == 'in-domain' else 1000)
        rows.append(dict(split=split, pair=f'{x} - {y}', diff=f, ci_lo=lo, ci_hi=hi))
pd.DataFrame(rows).to_csv(f'{OUT}/M1_boot_microF1.csv', index=False)
print('M1 done', flush=True)

# ---------------- M2 per language + review overlap (in-domain micro-F1)
groups = {**{lg: (te.language == lg).values for lg in ['en', 'vi', 'zh', 'ja']},
          'review-disjoint': (ov.review_in_train == False).values, 'untraceable': ov.review_in_train.isna().values}
rows = []
for (m, s), r in R.items():
    for g, mask in groups.items():
        rows.append(dict(model=m, seed=s, grp=g, n=int(mask.sum()), **met(Yi[mask], (r['pi'][mask] > r['thr']).astype(int), r['pi'][mask])))
agg(rows, ['grp', 'model']).to_csv(f'{OUT}/M2_subsets.csv', index=False)

# ---------------- M3 error composition
rows = []
for (m, s), r in R.items():
    for split, Y, p in [('in-domain', Yi, r['pi']), ('target', Yt, r['pt'])]:
        Yh = (p > r['thr']).astype(int); miss = ((Y == 1) & (Yh == 0)).sum(); spur = ((Y == 0) & (Yh == 1)).sum(); wrong = ~(Y == Yh).all(1)
        rows.append(dict(model=m, seed=s, split=split, missed_share=miss / (miss + spur), missed=miss, spurious=spur,
                         wrong=wrong.sum(), only_missing=(wrong & ~((Y == 0) & (Yh == 1)).any(1)).sum(),
                         only_spurious=(wrong & ~((Y == 1) & (Yh == 0)).any(1)).sum()))
agg(rows, ['split', 'model']).to_csv(f'{OUT}/M3_errors.csv', index=False)

# ---------------- M4 threshold sensitivity on target
def cv_thr(Y, p, k=5, grid=np.linspace(0.02, 0.98, 49)):
    Yh = np.zeros_like(Y)
    for tr, ts in KFold(k, shuffle=True, random_state=42).split(Y):
        for j in range(Y.shape[1]):
            t = grid[int(np.argmax([f1_score(Y[tr, j], (p[tr, j] >= g).astype(int), zero_division=0) for g in grid]))]
            Yh[ts, j] = (p[ts, j] >= t).astype(int)
    return Yh
def cv_glob(Y, p, k=5, grid=np.linspace(0.02, 0.98, 49)):
    Yh = np.zeros_like(Y)
    for tr, ts in KFold(k, shuffle=True, random_state=42).split(Y):
        t = grid[int(np.argmax([f1_score(Y[tr], (p[tr] >= g).astype(int), average='micro', zero_division=0) for g in grid]))]
        Yh[ts] = (p[ts] >= t).astype(int)
    return Yh
def topk(p, k):
    Yh = np.zeros_like(p, dtype=int); o = np.argsort(-p, 1)
    for i in range(len(p)):
        Yh[i, o[i, :k[i]]] = 1
    return Yh
rows = []
for (m, s), r in R.items():
    p = r['pt']
    for nm, Yh in [('source-tuned (main)', (p > r['thr']).astype(int)), ('fixed 0.5', (p > 0.5).astype(int)),
                   ('target-CV global', cv_glob(Yt, p)), ('target-CV per-aspect', cv_thr(Yt, p)),
                   ('oracle top-k', topk(p, Yt.sum(1))), ('top-2', topk(p, np.full(len(p), 2)))]:
        rows.append(dict(model=m, seed=s, decoder=nm, **met(Yt, Yh)))
    rows.append(dict(model=m, seed=s, decoder='threshold-free', microPRAUC=average_precision_score(Yt, p, average='micro'), LRAP=lrap(Yt, p)))
    rows.append(dict(model=m, seed=s, decoder='threshold-free (in-domain)', microPRAUC=average_precision_score(Yi, r['pi'], average='micro'), LRAP=lrap(Yi, r['pi'])))
agg(rows, ['decoder', 'model']).to_csv(f'{OUT}/M4_thresholds.csv', index=False)
print('M4 done', flush=True)

# ---------------- M5 primary accuracy: VI in-domain vs target, seed-averaged per-sentence correctness, bootstrap
vi = (te.language == 'vi').values
rows = []
for m in MODELS:
    ai = np.mean([prim(Yi, R[(m, s)]['pi']) for s in SEEDS], 0); at = np.mean([prim(Yt, R[(m, s)]['pt']) for s in SEEDS], 0)
    for ref, A in [('all', ai), ('vi', ai[vi])]:
        d = np.array([RNG.choice(A, len(A)).mean() - RNG.choice(at, len(at)).mean() for _ in range(10000)])
        rows.append(dict(model=m, reference=ref, in_domain=A.mean(), target=at.mean(), diff=A.mean() - at.mean(),
                         ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5), p=2 * min((d <= 0).mean(), (d >= 0).mean())))
pd.DataFrame(rows).to_csv(f'{OUT}/M5_primary_vi_vs_target.csv', index=False)
rows = []
c = Yt.sum(1)
for (m, s), r in R.items():
    pv = prim(Yt, r['pt'])
    for k, mask in [('1', c == 1), ('2', c == 2), ('3', c == 3), ('>=4', c >= 4)]:
        rows.append(dict(model=m, seed=s, card=k, n=mask.sum(), primary=pv[mask].mean(), chance=(c[mask] / 9).mean()))
agg(rows, ['card', 'model']).to_csv(f'{OUT}/M5_primary_by_card.csv', index=False)

# ---------------- M6 per aspect (all models, mean over seeds)
rows = []
for (m, s), r in R.items():
    for split, Y, p in [('in-domain', Yi, r['pi']), ('target', Yt, r['pt'])]:
        Yh = (p > r['thr']).astype(int)
        P = precision_score(Y, Yh, average=None, zero_division=0); Rc = recall_score(Y, Yh, average=None, zero_division=0); F = f1_score(Y, Yh, average=None, zero_division=0)
        for j, asp in enumerate(L):
            rows.append(dict(model=m, seed=s, split=split, aspect=asp, P=P[j], R=Rc[j], F1=F[j]))
agg(rows, ['split', 'model', 'aspect']).to_csv(f'{OUT}/M6_per_aspect.csv', index=False)

# ---------------- M7 TF-IDF ablation (paired by seed)
rows = []
for m in MODELS:
    for s in SEEDS:
        for var in ['tfidf', 'notfidf']:
            r = R[(m, s)] if var == 'tfidf' else load(m, s, 'notfidf')
            row = dict(model=m, seed=s, variant=var)
            row['in_F1'] = f1_score(Yi, (r['pi'] > r['thr']).astype(int), average='micro')
            row['tgt_F1'] = f1_score(Yt, (r['pt'] > r['thr']).astype(int), average='micro')
            row['tgt_cv_F1'] = f1_score(Yt, cv_thr(Yt, r['pt']), average='micro')
            for e in EXT:
                p, Y = r[e]; ext = pd.read_csv(f'{ROOT}/data_external/{e}.csv')
                cols = [i for i, x in enumerate(L) if ext['applicable_' + x].iloc[0] == 1]
                row[e] = f1_score(Y[:, cols], (p[:, cols] > r['thr'][cols]).astype(int), average='micro')
            rows.append(row)
ab = pd.DataFrame(rows); ab.to_csv(f'{OUT}/M7_ablation_per_seed.csv', index=False)
cols = ['in_F1', 'tgt_F1', 'tgt_cv_F1'] + EXT
w = ab.pivot_table(index=['model', 'seed'], columns='variant', values=cols)
rows = []
for m in MODELS:
    x = w.loc[m]
    r = dict(model=m)
    for cc in cols:
        dlt = x[(cc, 'tfidf')] - x[(cc, 'notfidf')]
        r[cc + '_fusion'] = x[(cc, 'tfidf')].mean(); r[cc + '_enc'] = x[(cc, 'notfidf')].mean()
        r[cc + '_delta'] = dlt.mean(); r[cc + '_delta_sd'] = dlt.std(ddof=1); r[cc + '_delta_pos'] = int((dlt > 0).sum())
    rows.append(r)
pd.DataFrame(rows).to_csv(f'{OUT}/M7_ablation_summary.csv', index=False)
print('M7 done', flush=True)

# ---------------- M8 SemEval (applicable labels)
rows = []
for (m, s), r in R.items():
    for e in EXT:
        p, Y = r[e]; ext = pd.read_csv(f'{ROOT}/data_external/{e}.csv')
        cols = [i for i, x in enumerate(L) if ext['applicable_' + x].iloc[0] == 1]
        rows.append(dict(model=m, seed=s, set=e, **met(Y[:, cols], (p[:, cols] > r['thr'][cols]).astype(int), p[:, cols])))
agg(rows, ['set', 'model']).to_csv(f'{OUT}/M8_semeval.csv', index=False)
print('all done')
