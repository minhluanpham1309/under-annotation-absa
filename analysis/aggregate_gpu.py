"""
Aggregate GPU runs (results_gpu/*.npz) into the tables used in the IASC manuscript.

T_seed    mean ± sd over seeds, in-domain and target, per model (TF-IDF fusion, 10 ep)
T_abl     TF-IDF ablation: fusion vs encoder-only (mean ± sd, paired by seed)
T_mt5     mT5-small: 10-epoch budget vs convergence budget (+ epochs actually run)
T_ext     external SemEval sets (EN hotel, EN restaurant, NL restaurant): metrics on the
          applicable labels only, mean ± sd over seeds; TF-IDF baseline added if present
T_rank    how often each model ranks first per (set, metric) across seeds
T_thr     target-set threshold sensitivity averaged over seeds (source vs target-CV thresholds)

    python analysis/aggregate_gpu.py --runs results/gpu_runs --out results/tables_gpu
"""
import argparse, glob, json, os, re
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from sklearn.model_selection import KFold

L = ["overall", "amenities", "food", "service", "location", "ambiance", "room", "price", "cleanliness"]
EXT = ["semeval15_en_hotel", "semeval16_en_rest", "semeval16_nl_rest"]
sig = lambda z: 1 / (1 + np.exp(-z))


_LBL = {}
def labels(d, key, ext_dir="data_external"):
    """Gold labels: stored in the run file if present, otherwise read from the dataset CSV."""
    if f"{key}_Y" in d.files:
        return d[f"{key}_Y"]
    if key not in _LBL:
        path = {"test": "vn_hotel_res_aspect/test.csv", "val": "vn_hotel_res_aspect/validation.csv",
                "transfer_test": "data_unilever/splits/transfer_test.csv"}.get(key, f"{ext_dir}/{key}.csv")
        _LBL[key] = pd.read_csv(path)[L].values
    return _LBL[key]


def lrap(Y, p):
    o = np.argsort(-p, 1); Ys = np.take_along_axis(Y, o, 1)
    prec = np.cumsum(Ys, 1) / np.arange(1, Y.shape[1] + 1)
    den = Ys.sum(1); ok = den > 0
    return ((prec * Ys).sum(1)[ok] / den[ok]).mean()


def metr(Y, p, thr, cols=None):
    if cols is not None:
        Y, p, thr = Y[:, cols], p[:, cols], thr[cols]
    keep = Y.sum(1) > 0
    Y, p = Y[keep], p[keep]
    Yh = (p > thr).astype(int)
    return dict(microP=precision_score(Y, Yh, average="micro", zero_division=0),
                microR=recall_score(Y, Yh, average="micro", zero_division=0),
                microF1=f1_score(Y, Yh, average="micro", zero_division=0),
                macroF1=f1_score(Y, Yh, average="macro", zero_division=0),
                subset=(Y == Yh).all(1).mean(),
                primary=Y[np.arange(len(Y)), p.argmax(1)].mean(),
                LRAP=lrap(Y, p),
                microPRAUC=average_precision_score(Y, p, average="micro"))


def cv_thr(Y, p, k=5, seed=42, grid=np.linspace(0.02, 0.98, 49)):
    Yh = np.zeros_like(Y)
    for a, b in KFold(k, shuffle=True, random_state=seed).split(Y):
        for j in range(Y.shape[1]):
            t = max(grid, key=lambda t: f1_score(Y[a, j], (p[a, j] >= t).astype(int), zero_division=0))
            Yh[b, j] = (p[b, j] >= t).astype(int)
    return Yh


def ms(x):
    x = np.asarray(x, float)
    return f"{x.mean():.4f} ± {x.std(ddof=1) if len(x) > 1 else 0:.4f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="results_gpu")
    ap.add_argument("--out", default="analysis_out_gpu"); ap.add_argument("--ext_dir", default="data_external")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(f"{a.runs}/*.npz")):
        tag = os.path.basename(f)[:-4]; meta = json.load(open(f[:-4] + ".json")) if os.path.exists(f[:-4] + ".json") else {}
        d = np.load(f); thr = d["thr"]
        model = meta.get("model", tag.split("_")[0])
        seed = meta["seed"] if "seed" in meta else int((re.findall(r"s(\d+)$", tag) or ["0"])[0])
        variant = ("encoder-only" if meta.get("no_tfidf", "notfidf" in tag) else "TF-IDF fusion")
        budget = f"{meta.get('epochs', 10)} ep"
        base = dict(tag=tag, model=model, seed=seed, variant=variant, budget=budget,
                    epochs_run=int(d["epochs_run"]), early_stopped=int(d["early_stopped_epoch"]) > 0,
                    train_min=float(d["train_time"]) / 60)
        rows.append(dict(base, set="in-domain", **metr(labels(d, "test"), sig(d["test_logit"]), thr)))
        if "transfer_test_logit" in d.files:
            Yt, pt = labels(d, "transfer_test"), sig(d["transfer_test_logit"])
            rows.append(dict(base, set="target", **metr(Yt, pt, thr)))
            Yh = cv_thr(Yt.astype(int), pt)
            rows.append(dict(base, set="target (target-CV thresholds)",
                             microF1=f1_score(Yt, Yh, average="micro"), macroF1=f1_score(Yt, Yh, average="macro"),
                             subset=(Yt == Yh).all(1).mean()))
        for e in EXT:
            if f"{e}_logit" in d.files:
                ext = pd.read_csv(f"{a.ext_dir}/{e}.csv")
                cols = [i for i, x in enumerate(L) if ext["applicable_" + x].iloc[0] == 1]
                rows.append(dict(base, set=e, **metr(labels(d, e, a.ext_dir), sig(d[f"{e}_logit"]), thr, cols)))
    if not rows:
        print("no runs found in", a.runs); return
    R = pd.DataFrame(rows); R.to_csv(f"{a.out}/all_runs_long.csv", index=False)
    metrics = ["microP", "microR", "microF1", "macroF1", "subset", "primary", "LRAP", "microPRAUC"]

    def table(df, keys):
        g = df.groupby(keys)
        t = g[metrics].agg(lambda x: ms(x.dropna()) if x.notna().any() else "")
        t["n_seeds"] = g.size(); return t.reset_index()

    main_ = R[(R.variant == "TF-IDF fusion") & (R.budget == "10 ep")]
    table(main_, ["set", "model"]).to_csv(f"{a.out}/T_seed.csv", index=False)
    table(R[R.budget == "10 ep"], ["set", "model", "variant"]).to_csv(f"{a.out}/T_abl.csv", index=False)
    mt5 = R[R.model.str.startswith("mT5") & (R.variant == "TF-IDF fusion")]
    t = table(mt5, ["set", "model", "budget"])
    t = t.merge(mt5.groupby(["model", "budget"]).agg(epochs_run=("epochs_run", "mean"),
                                                     early_stopped=("early_stopped", "mean"),
                                                     train_min=("train_min", "mean")).reset_index(), on=["model", "budget"])
    t.to_csv(f"{a.out}/T_mt5.csv", index=False)
    table(main_[main_.set.isin(EXT)], ["set", "model"]).to_csv(f"{a.out}/T_ext.csv", index=False)
    rk = []
    for (s, seed), g in main_.groupby(["set", "seed"]):
        for m in ["microF1", "macroF1", "primary", "LRAP"]:
            if g[m].notna().all():
                rk.append(dict(set=s, seed=seed, metric=m, winner=g.loc[g[m].idxmax(), "model"]))
    if rk:
        rk = pd.DataFrame(rk); rk.groupby(["set", "metric", "winner"]).size().rename("n_seeds_won").reset_index() \
            .to_csv(f"{a.out}/T_rank.csv", index=False)
    print(table(main_, ["set", "model"]).to_string())


if __name__ == "__main__":
    main()
