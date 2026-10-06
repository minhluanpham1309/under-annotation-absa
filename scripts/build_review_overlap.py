"""
Recover the source review of each test sentence by exact matching (NFC, lower-case, collapsed
whitespace) against the review-level release of the corpus (dataset_luan/B_{train,validation,test}.csv,
columns review_id, source, place_type, sentence), and flag whether that review also contributed
sentences to the training split. Sentences with no match get review_in_train = NaN and are not used
in the review-disjoint analysis.

    python scripts/build_review_overlap.py   -> data/test_review_overlap.csv
"""
import os, re, unicodedata
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
norm = lambda s: re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s)).strip().lower())

d = pd.concat([pd.read_csv(f"{ROOT}/dataset_luan/B_{s}.csv") for s in ["train", "validation", "test"]])
d["k"] = d.sentence.map(norm)
g = d.groupby("k").agg(rids=("review_id", lambda x: "|".join(sorted(set(map(str, x))))),
                       source=("source", lambda x: "|".join(sorted(set(x)))),
                       place=("place_type", lambda x: "|".join(sorted(set(x))))).reset_index()
tr = pd.read_csv(f"{ROOT}/vn_hotel_res_aspect/train.csv"); tr["k"] = tr.sentence.map(norm)
train_reviews = set("|".join(tr.merge(g, on="k", how="left").rids.dropna()).split("|"))
te = pd.read_csv(f"{ROOT}/vn_hotel_res_aspect/test.csv"); te["k"] = te.sentence.map(norm)
te = te.merge(g, on="k", how="left")
te["review_in_train"] = te.rids.map(lambda r: bool(set(r.split("|")) & train_reviews) if isinstance(r, str) else None)
os.makedirs(f"{ROOT}/data", exist_ok=True)
te[["source", "place", "review_in_train"]].to_csv(f"{ROOT}/data/test_review_overlap.csv", index=False)
print(te.review_in_train.value_counts(dropna=False).to_dict())
