"""
E9  Review-level overlap: the vn_hotel split is sentence-level, so most test sentences come
    from reviews that also contributed sentences to training. Re-score every system on the
    review-disjoint test subset (sentences whose source review has no training sentence).
E10 TF-IDF baseline on the external SemEval sets (CPU; transformer numbers need GPU runs).

    python analysis/review_overlap_and_external.py
"""
import os, numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/results/analysis_cpu"
L = ['overall', 'amenities', 'food', 'service', 'location', 'ambiance', 'room', 'price', 'cleanliness']
sig = lambda z: 1 / (1 + np.exp(-z))
te = pd.read_csv(f"{ROOT}/vn_hotel_res_aspect/test.csv")
ov = pd.read_csv(f"{ROOT}/data/test_review_overlap.csv")
assert len(ov) == len(te)
Y = te[L].values


def m(Y, p, thr):
    Yh = (p > thr).astype(int)
    return dict(microP=precision_score(Y, Yh, average='micro', zero_division=0),
                microR=recall_score(Y, Yh, average='micro', zero_division=0),
                microF1=f1_score(Y, Yh, average='micro', zero_division=0),
                macroF1=f1_score(Y, Yh, average='macro', zero_division=0),
                subset=(Y == Yh).all(1).mean(), primary=Y[np.arange(len(Y)), p.argmax(1)].mean())


tf = np.load(f"{OUT}/preds_tfidf.npz")
sys = {'TF-IDF+SVM': (tf['te'], tf['thr'])}
for k in ['mBERT', 'XLM-R', 'mT5']:
    d = np.load(f"{ROOT}/results/seed42_original/preds_{k}.npz"); sys[k] = (sig(d['logit']), d['thr'])
groups = {'all test': np.ones(len(te), bool),
          'review seen in training': (ov.review_in_train == True).values,
          'review-disjoint': (ov.review_in_train == False).values,
          'source review unknown': ov.review_in_train.isna().values}
rows = []
for g, mask in groups.items():
    for k, (p, thr) in sys.items():
        rows.append(dict(test_subset=g, n=int(mask.sum()), model=k, **m(Y[mask], p[mask], thr)))
pd.DataFrame(rows).to_csv(f"{OUT}/E9_review_overlap.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string())

# ---------------- E10: TF-IDF on SemEval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_union
from sklearn.svm import LinearSVC
tr = pd.read_csv(f"{ROOT}/vn_hotel_res_aspect/train.csv")
feats = make_union(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300_000, sublinear_tf=True),
                   TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, sublinear_tf=True))
X = feats.fit_transform(tr.sentence.astype(str))
clfs = [LinearSVC(C=0.5).fit(X, tr[a].values) for a in L]
rows = []
for name in ['semeval15_en_hotel', 'semeval16_en_rest', 'semeval16_nl_rest']:
    e = pd.read_csv(f"{ROOT}/data_external/{name}.csv")
    S = np.column_stack([c.decision_function(feats.transform(e.sentence.astype(str))) for c in clfs])
    cols = [i for i, a in enumerate(L) if e['applicable_' + a].iloc[0] == 1]
    rows.append(dict(set=name, n=len(e), applicable=len(cols), gold_card=e[L].values[:, cols].sum(1).mean(),
                     model='TF-IDF+SVM', **m(e[L].values[:, cols], S[:, cols], tf['thr'][cols])))
pd.DataFrame(rows).to_csv(f"{OUT}/E10_semeval_tfidf.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string())
