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
t=color_context_feature
RESULT="results/DirectionMethodQuestioner_${t}_train_0_167.gzip.json"

# The 14:44 incident: run_official_eval.py died on preflight (missing `ninja` broke the GDN kernel
# JIT), but this script went on to copy and summarize the STALE Aug-6 gzip still sitting at
# $RESULT, producing a summary that looked real and silently overwrote summary/ALL.json. Guard
# against that twice over: refuse to proceed on a non-zero exit, and refuse to trust a $RESULT
# that the run did not actually rewrite.
BEFORE_MTIME=$(stat -c %Y "$RESULT" 2>/dev/null || echo 0)

echo "=== starting $t at $(date) ==="
python3 direction_method/scripts/run_official_eval.py 0 167 \
  --description-type "$t" --oracle stub \
  > "$LOGDIR/${t}.log" 2>&1
EVAL_EXIT=$?
echo "=== finished $t at $(date), exit=$EVAL_EXIT ==="

if [ "$EVAL_EXIT" -ne 0 ]; then
  echo "!!! eval FAILED (exit=$EVAL_EXIT) -- not copying, not summarizing. See $LOGDIR/${t}.log"
  exit "$EVAL_EXIT"
fi

AFTER_MTIME=$(stat -c %Y "$RESULT" 2>/dev/null || echo 0)
if [ "$AFTER_MTIME" -le "$BEFORE_MTIME" ]; then
  echo "!!! $RESULT was NOT rewritten by this run (mtime unchanged) -- refusing to summarize stale data."
  exit 1
fi

cp "$RESULT" "$RUNDIR/${t}.gzip.json"
python3 direction_method/scripts/summarize_official.py \
  --results-glob "$RESULT" \
  --oracle-model-id "Qwen/Qwen3.6-35B-A3B-FP8" \
  --out-dir "$RUNDIR/summary" > "$LOGDIR/${t}.summary.log" 2>&1

echo "=== $t DONE at $(date) ==="
