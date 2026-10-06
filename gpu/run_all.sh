#!/usr/bin/env bash
# =====================================================================
#  Run all GPU experiments of the paper with one command (from repo root):
#       bash gpu/run_all.sh
#  Output: results/new_run/ and results/new_run_<timestamp>.zip
#
#  Options (environment variables):
#       TIERS="1 2"          1 = 3 models x 3 seeds (TF-IDF fusion, 10 epochs)
#                            2 = encoder-only ablation + mT5-small with a 30-epoch budget
#                            3 = mT5-base (not used in the paper)
#       SEEDS="13 42 2026"
#       AMP=1                bf16 autocast (the paper used fp32, AMP=0)
#  Re-running the same command resumes: finished runs are skipped.
# =====================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
TIERS=${TIERS:-"1 2"}; SEEDS=${SEEDS:-"13 42 2026"}; AMP=${AMP:-0}
OUT=$ROOT/results/new_run; RUNS=$OUT/runs; LOGS=$OUT/logs
mkdir -p "$RUNS" "$LOGS"
exec > >(tee -a "$OUT/run_all.log") 2>&1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
ts() { date '+%Y-%m-%d %H:%M:%S'; }
PY=$(command -v python3 || command -v python)
echo "==== $(ts) START | TIERS=[$TIERS] SEEDS=[$SEEDS] AMP=$AMP"

echo "---- [1/5] Environment"
$PY -m pip install -r requirements.txt || true
$PY - <<'P' || { echo "ERROR: environment not ready"; exit 1; }
import torch, transformers, sklearn
print(f"  torch {torch.__version__} | transformers {transformers.__version__} | sklearn {sklearn.__version__}")
assert torch.cuda.is_available(), "No GPU found"
print("  GPU:", torch.cuda.get_device_name(0))
P

echo "---- [2/5] Data check"
for f in vn_hotel_res_aspect/train.csv vn_hotel_res_aspect/validation.csv vn_hotel_res_aspect/test.csv \
         data_unilever/splits/transfer_test.csv data_external/semeval15_en_hotel.csv \
         data_external/semeval16_en_rest.csv data_external/semeval16_nl_rest.csv; do
  [[ -f $f ]] || { echo "ERROR: missing $f (see data/README.md)"; exit 1; }
  echo "  OK $f"
done

echo "---- [3/5] Smoke test (mBERT, 400 sentences, 1 epoch)"
if $PY gpu/run_one.py --model mBERT --limit 400 --epochs 1 --out "$OUT/_smoke" --tag smoke > "$LOGS/_smoke.log" 2>&1; then
  echo "  smoke test passed"; rm -rf "$OUT/_smoke"
else
  echo "ERROR in smoke test, see $LOGS/_smoke.log"; tail -30 "$LOGS/_smoke.log"; exit 1
fi

EXTRA=""; [[ $AMP == 1 ]] && EXTRA="--amp"
JOBS=()
for t in $TIERS; do
  case $t in
    1) for s in $SEEDS; do for m in XLM-R mBERT mT5; do JOBS+=("--model $m --seed $s"); done; done ;;
    2) for s in $SEEDS; do for m in XLM-R mBERT mT5; do JOBS+=("--model $m --seed $s --no_tfidf"); done; done
       for s in $SEEDS; do JOBS+=("--model mT5 --seed $s --epochs 30"); done ;;
    3) for s in $SEEDS; do JOBS+=("--model mT5-base --seed $s --epochs 30"); done ;;
  esac
done
N=${#JOBS[@]}; FAILED=(); i=0
echo "---- [4/5] Training: $N runs"
for j in "${JOBS[@]}"; do
  i=$((i + 1)); name=$(echo "$j" | sed 's/--model //; s/ --seed /_s/; s/ --no_tfidf/_notfidf/; s/ --epochs /_e/; s/ /_/g')
  echo "  [$i/$N] $(ts) start: $name"; t0=$(date +%s)
  if $PY gpu/run_one.py $j --out "$RUNS" $EXTRA > "$LOGS/$name.log" 2>&1; then
    echo "  [$i/$N] $(ts) done: $name ($(( ($(date +%s) - t0) / 60 )) min)"
  else
    echo "  [$i/$N] $(ts) FAILED: $name (see logs/$name.log)"; FAILED+=("$name")
  fi
  $PY analysis/aggregate_gpu.py --runs "$RUNS" --out "$OUT/tables" > /dev/null 2>&1 || true
done

echo "---- [5/5] Aggregate"
$PY analysis/aggregate_gpu.py --runs "$RUNS" --out "$OUT/tables" | tail -40
{ echo "Finished: $(ts)"; echo "Successful runs: $(ls "$RUNS"/*.npz 2>/dev/null | wc -l) / $N"
  [[ ${#FAILED[@]} -gt 0 ]] && echo "Failed: ${FAILED[*]}"; } | tee "$OUT/SUMMARY.txt"
STAMP=$(date '+%Y%m%d_%H%M')
$PY -m zipfile -c "$ROOT/results/new_run_$STAMP.zip" "$OUT" && echo "==== $(ts) DONE: results/new_run_$STAMP.zip"
