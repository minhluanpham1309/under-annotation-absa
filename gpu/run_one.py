"""
One training run = one (model, seed, TF-IDF on/off, epoch budget) configuration.
Faithful CLI port of thuc_nghiem_3_mo_hinh.ipynb (same architecture, optimiser,
thresholding and checkpoint rule), plus:
  * --seed                 multi-seed runs
  * --no_tfidf             ablation of the TF-IDF branch
  * --epochs/--patience    longer budget for mT5 (convergence check)
  * --eval                 extra test CSVs (target set, SemEval sets) -> logits saved
Every run writes  <out>/<tag>.npz  with logits for validation, in-domain test and every
extra eval set, the tuned thresholds, the val-loss curve and timing.

Example:
  python gpu/run_one.py --model XLM-R --seed 13
  python gpu/run_one.py --model XLM-R --seed 13 --no_tfidf
  python gpu/run_one.py --model mT5 --seed 13 --epochs 30 --patience 3 --tag mT5_long_s13
"""
import argparse, os, time, json, random
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel, MT5EncoderModel

ASPECTS = ["overall", "amenities", "food", "service", "location", "ambiance", "room", "price", "cleanliness"]
HF = {"mBERT": "bert-base-multilingual-cased", "XLM-R": "xlm-roberta-base",
      "mT5": "google/mt5-small", "mT5-base": "google/mt5-base"}
DEFAULT_EVAL = ["data_unilever/splits/transfer_test.csv",
                "data_external/semeval15_en_hotel.csv",
                "data_external/semeval16_en_rest.csv",
                "data_external/semeval16_nl_rest.csv"]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class AspectDataset(Dataset):
    def __init__(self, df, tok, tf, max_len):
        enc = tok(df.sentence.tolist(), truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
        self.ids, self.am = enc["input_ids"], enc["attention_mask"]
        self.y = torch.from_numpy(df[ASPECTS].values.astype(np.float32))
        self.tf = tf

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        tf = torch.zeros(0) if self.tf is None else torch.from_numpy(np.asarray(self.tf[i].todense()).ravel())
        return {"input_ids": self.ids[i], "attention_mask": self.am[i], "tfidf": tf, "labels": self.y[i]}


class FusionClassifier(nn.Module):
    def __init__(self, path, tf_dim):
        super().__init__()
        self.is_t5 = "t5" in path.lower()
        self.encoder = MT5EncoderModel.from_pretrained(path) if self.is_t5 else AutoModel.from_pretrained(path)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.encoder.config.hidden_size + tf_dim, len(ASPECTS))

    def forward(self, ids, am, tf):
        h = self.encoder(input_ids=ids, attention_mask=am).last_hidden_state
        if self.is_t5:
            m = am.unsqueeze(-1).float(); v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        else:
            v = h[:, 0]
        if tf.shape[1] > 0:
            v = torch.cat([v, tf], 1)
        return self.fc(self.dropout(v))


@torch.no_grad()
def predict(model, loader, dev):
    model.eval(); crit = nn.BCEWithLogitsLoss(); Ls, Ys, tot, n = [], [], 0.0, 0
    for b in loader:
        lo = model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["tfidf"].to(dev)).float()
        y = b["labels"].to(dev)
        tot += crit(lo, y).item() * len(y); n += len(y)
        Ls.append(lo.cpu().numpy()); Ys.append(y.cpu().numpy())
    return np.vstack(Ls), np.vstack(Ys), tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(HF))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_tfidf", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--lr_enc", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--tfidf_dim", type=int, default=4096)
    ap.add_argument("--data_dir", default="vn_hotel_res_aspect")
    ap.add_argument("--train_file", default=None, help="override training CSV (e.g. silver-label ablation)")
    ap.add_argument("--val_file", default=None)
    ap.add_argument("--eval", nargs="*", default=DEFAULT_EVAL)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (faster; default off = fp32 like the original runs)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results_gpu")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--hf_path", default=None, help="override HF id / local dir (smoke tests)")
    ap.add_argument("--limit", type=int, default=None, help="subsample each split (smoke tests)")
    a = ap.parse_args()

    tag = a.tag or f"{a.model}_{'notfidf' if a.no_tfidf else 'tfidf'}_e{a.epochs}_s{a.seed}"
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(f"{a.out}/{tag}.npz"):
        print("exists, skipping", tag); return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(a.seed)

    rd = lambda p: pd.read_csv(p).assign(sentence=lambda d: d.sentence.astype(str)).reset_index(drop=True)
    train = rd(a.train_file or f"{a.data_dir}/train.csv")
    val = rd(a.val_file or f"{a.data_dir}/validation.csv")
    test = rd(f"{a.data_dir}/test.csv")
    evals = {os.path.splitext(os.path.basename(p))[0]: rd(p) for p in a.eval if os.path.exists(p)}
    missing = [p for p in a.eval if not os.path.exists(p)]
    if missing:
        print("[warn] eval files not found, skipped:", missing)

    if a.limit:
        train, val, test = train.head(a.limit), val.head(a.limit // 4), test.head(a.limit // 4)
        evals = {k: v.head(a.limit // 4) for k, v in evals.items()}
    path = a.hf_path or HF[a.model]
    vec = None
    if not a.no_tfidf:
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=a.tfidf_dim)
        vec.fit(train.sentence)
    tfm = lambda d: None if vec is None else vec.transform(d.sentence).astype(np.float32)
    tf_dim = 0 if vec is None else len(vec.vocabulary_)

    tok = AutoTokenizer.from_pretrained(path)
    mk = lambda d, sh: DataLoader(AspectDataset(d, tok, tfm(d), a.max_len), batch_size=a.bs, shuffle=sh,
                                  num_workers=a.workers, pin_memory=(dev == "cuda"))
    tr_dl, va_dl, te_dl = mk(train, True), mk(val, False), mk(test, False)

    model = FusionClassifier(path, tf_dim).to(dev)
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": a.lr_enc},
                             {"params": model.fc.parameters(), "lr": a.lr_head}], weight_decay=0.01)
    crit = nn.BCEWithLogitsLoss()
    best, bad, best_state, curve, stopped = float("inf"), 0, None, [], None
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); run = 0.0
        for b in tr_dl:
            ids, am = b["input_ids"].to(dev), b["attention_mask"].to(dev)
            tf, y = b["tfidf"].to(dev), b["labels"].to(dev)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp and dev == "cuda"):
                lo = model(ids, am, tf)
            loss = crit(lo.float(), y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); run += loss.item() * len(y)
        _, _, vl = predict(model, va_dl, dev)
        curve.append((ep, run / len(train), vl)); print(f"[{tag}] epoch {ep} train {run/len(train):.4f} val {vl:.4f}", flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                stopped = ep; break
    train_time = time.time() - t0
    model.load_state_dict(best_state)

    lv, yv, _ = predict(model, va_dl, dev); pv = 1 / (1 + np.exp(-lv))
    grid = np.arange(0.05, 0.96, 0.01)
    thr = np.array([grid[int(np.argmax([f1_score(yv[:, j], (pv[:, j] > t).astype(int), zero_division=0) for t in grid]))]
                    for j in range(len(ASPECTS))])
    out = dict(val_logit=lv, val_Y=yv, thr=thr, curve=np.array(curve), train_time=train_time,
               best_val_loss=best, early_stopped_epoch=-1 if stopped is None else stopped,
               epochs_run=len(curve))
    lt, yt, tl = predict(model, te_dl, dev); out.update(test_logit=lt, test_Y=yt, test_loss=tl)
    for k, d in evals.items():
        lo, y, _ = predict(model, mk(d, False), dev); out[f"{k}_logit"] = lo; out[f"{k}_Y"] = y
    np.savez_compressed(f"{a.out}/{tag}.npz", **out)
    json.dump(dict(vars(a), tag=tag, train_time_min=round(train_time / 60, 1), epochs_run=len(curve),
                   early_stopped_epoch=stopped, best_val_loss=best, tfidf_dim=tf_dim, eval_sets=list(evals)),
              open(f"{a.out}/{tag}.json", "w"), indent=2)
    print("saved", tag, f"{train_time/60:.1f} min")


if __name__ == "__main__":
    main()
