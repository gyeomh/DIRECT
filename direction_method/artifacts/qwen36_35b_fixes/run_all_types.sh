#!/usr/bin/env bash
set -uo pipefail
cd /home/gyeom/coin_challenge

source /home/gyeom/miniconda3/bin/activate coin_env
export VLM_PORT=8020
export VLM_MODEL_ID="Qwen/Qwen3.6-35B-A3B-FP8"
export VLM_DISABLE_THINKING=1
export VLM_USE_CACHE=0
export VLM_CACHE_DIR=/home/gyeom/coin_challenge/direction_method/artifacts/qwen36_35b_fixes/cache

RUNDIR=/home/gyeom/coin_challenge/direction_method/artifacts/qwen36_35b_fixes
LOGDIR=$RUNDIR/eval_logs
TYPES=(category color context color_feature color_context color_context_feature)

for t in "${TYPES[@]}"; do
  echo "=== starting $t at $(date) ==="
  python3 direction_method/scripts/run_official_eval.py 0 167 \
    --description-type "$t" --oracle stub \
    > "$LOGDIR/${t}.log" 2>&1
  echo "=== finished $t at $(date), exit=$? ==="
  # Preserve this type's raw gzip result immediately -- eval_model.py writes to a fixed filename
  # per type, so the next type's run would otherwise overwrite it (this is how the 3.6 baseline's
  # category gzip was lost on 2026-08-11).
  cp "results/DirectionMethodQuestioner_${t}_train_0_167.gzip.json" "$RUNDIR/${t}.gzip.json"
  # Summarize as we go, so partial results are readable without waiting for all six.
  python3 direction_method/scripts/summarize_official.py \
    --results-glob "results/DirectionMethodQuestioner_${t}_train_0_167.gzip.json" \
    --oracle-model-id "Qwen/Qwen3.6-35B-A3B-FP8" \
    --out-dir "$RUNDIR/summary" > "$LOGDIR/${t}.summary.log" 2>&1
done

echo "=== ALL TYPES DONE at $(date) ==="
