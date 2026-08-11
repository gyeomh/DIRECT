#!/usr/bin/env bash
set -uo pipefail
cd /home/gyeom/coin_challenge

source /home/gyeom/miniconda3/bin/activate coin_env
export VLM_PORT=8020
export VLM_MODEL_ID="Qwen/Qwen3.6-35B-A3B-FP8"
export VLM_DISABLE_THINKING=1
export VLM_USE_CACHE=0
export VLM_CACHE_DIR=/home/gyeom/coin_challenge/direction_method/artifacts/qwen36_35b_fixes2/cache

RUNDIR=/home/gyeom/coin_challenge/direction_method/artifacts/qwen36_35b_fixes2
LOGDIR=$RUNDIR/eval_logs
TYPES=(context color_context)

for t in "${TYPES[@]}"; do
  RESULT="results/DirectionMethodQuestioner_${t}_train_0_167.gzip.json"
  BEFORE=$(stat -c %Y "$RESULT" 2>/dev/null || echo 0)

  echo "=== starting $t at $(date) ==="
  python3 direction_method/scripts/run_official_eval.py 0 167 \
    --description-type "$t" --oracle stub \
    > "$LOGDIR/${t}.log" 2>&1
  EXIT=$?
  echo "=== finished $t at $(date), exit=$EXIT ==="

  # Same two guards as the 14:44 incident fix: a failed run must not fall through to copying and
  # summarizing whatever stale gzip happens to be sitting at $RESULT.
  if [ "$EXIT" -ne 0 ]; then
    echo "!!! $t FAILED (exit=$EXIT) -- not copying, not summarizing. See $LOGDIR/${t}.log"
    continue
  fi
  AFTER=$(stat -c %Y "$RESULT" 2>/dev/null || echo 0)
  if [ "$AFTER" -le "$BEFORE" ]; then
    echo "!!! $t: $RESULT not rewritten (mtime unchanged) -- refusing to summarize stale data."
    continue
  fi

  cp "$RESULT" "$RUNDIR/${t}.gzip.json"
  python3 direction_method/scripts/summarize_official.py \
    --results-glob "$RESULT" \
    --oracle-model-id "Qwen/Qwen3.6-35B-A3B-FP8" \
    --out-dir "$RUNDIR/summary" > "$LOGDIR/${t}.summary.log" 2>&1
done

echo "=== BOTH TYPES DONE at $(date) ==="
