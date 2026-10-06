# Under-Annotation, Not Domain Shift

Code and experiment outputs for the manuscript

> **Under-Annotation, Not Domain Shift: A Human-Verified Diagnosis of Multilingual Transformer Models for Multi-Label Aspect Classification of Hospitality Reviews**
> Phuoc Tran, Luan Pham, Ha Huynh, Thom Nguyen (manuscript under review)

We fine-tune mBERT, XLM-R and the mT5-small encoder (three seeds each) for nine-aspect multi-label
classification of hotel and restaurant review sentences in English, Vietnamese, Chinese and Japanese.
We then test them on a human-verified Vietnamese hotel-review set from an independent source and on
three SemEval test sets. The repository contains every script used for training, evaluation,
statistical testing and figures, plus the per-sentence logits of all runs reported in the paper.

## Main findings (mean over seeds 13, 42, 2026)

| Test set | mBERT | XLM-R | mT5-small | XLM-R P / R |
|---|---|---|---|---|
| Source test (4 languages) | 0.847 | **0.865** | 0.837 | 0.86 / 0.87 |
| Human-verified target (VI) | 0.728 | 0.739 | 0.738 | 0.88 / **0.63** |
| SemEval-15 hotel (EN) | 0.800 | **0.812** | 0.797 | 0.85 / 0.78 |
| SemEval-16 restaurant (EN) | 0.845 | **0.874** | 0.840 | 0.91 / 0.84 |
| SemEval-16 restaurant (NL, unseen language) | 0.621 | **0.828** | 0.748 | 0.87 / 0.79 |

Micro-F1. On the target set precision is preserved, recall drops, and 81–83% of label errors are
omissions. The target set carries 2.18 gold aspects per sentence against 1.17 in the training corpus.
This points to under-annotation of the training data rather than domain shift. Re-tuning decision
thresholds on target data recovers 4–6 micro-F1 points. The TF-IDF fusion branch adds no measurable
value.

## Repository layout

```
gpu/
  run_one.py              one training run (model, seed, TF-IDF on/off, epoch budget); saves logits
  run_all.sh              all 21 GPU runs of the paper with one command
analysis/
  aggregate_gpu.py        mean ± sd tables from a folder of runs
  multiseed_analysis.py   all diagnostics on the 3-seed runs (McNemar, bootstrap, thresholds, errors, ...)
  cpu_experiments.py      TF-IDF + LinearSVC baseline and diagnostics on the original seed-42 run
  review_overlap_and_external.py   review-disjoint subset; TF-IDF baseline on SemEval
  make_figures.py         figures of the paper
data_prep/
  fix_lang.py             language re-identification with lingua
  merge_gold.py           merge the two annotators' sheets, Cohen's kappa, build the target set
  prepare_data_unilever.py   cleaning of the second collection and silver labels for the ablation
  prepare_semeval.py      map SemEval ENTITY#ATTRIBUTE categories to the nine aspects
  measure_underlabel.py   compare original source labels with a multi-label re-annotation
scripts/
  build_review_overlap.py recover source reviews of test sentences (review-disjoint analysis)
results/
  gpu_runs/               21 runs: *.npz (logits on validation, source test, target set, SemEval sets;
                          thresholds; loss curve) and *.json (configuration, epochs, time)
  tables_gpu/             aggregated tables (T_seed, T_abl, T_mt5, T_ext, T_rank, all_runs_long)
  analysis_3seed/         M1–M8 diagnostic tables used in Section 5 of the paper
  analysis_cpu/           E1–E10 tables (baseline, cardinality, silver-label ablation, ...)
  seed42_original/        original single-seed run and silver-label ablation (logits)
data/README.md            expected data layout and how to obtain the data
```

Gold label arrays have been removed from the released `.npz` files; the scripts read labels from the
dataset CSVs (see `data/README.md`).

## Reproducing

```bash
pip install -r requirements.txt

# 1. GPU experiments (~14 h on one A100, fp32)
bash gpu/run_all.sh                    # or TIERS="1" for the 9 main runs only

# 2. Diagnostics and tables (CPU)
python scripts/build_review_overlap.py
python analysis/cpu_experiments.py
python analysis/review_overlap_and_external.py
python analysis/multiseed_analysis.py --runs results/gpu_runs
python analysis/make_figures.py
```

Training configuration: AdamW (weight decay 0.01), learning rate 2e-5 for the encoder and 1e-4 for the
classification layer, batch size 16, maximum length 128, at most 10 epochs with early stopping
(patience 3 on validation loss), gradient clipping 1.0, dropout 0.1, per-aspect thresholds tuned on
the source validation set. Models: `bert-base-multilingual-cased`, `xlm-roberta-base`,
`google/mt5-small` (encoder, mean pooling).

## License

Code: MIT (see `LICENSE`). The datasets are not covered by this license and are not included.
