#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MAIN_RUN="${1:?Usage: bash experiments/queue_candidates.sh MAIN_RUN_DIRECTORY}"
if [[ ! -f "$MAIN_RUN/manifest.json" && -f "results/$MAIN_RUN/manifest.json" ]]; then
    MAIN_RUN="results/$MAIN_RUN"
fi
if [[ ! -f "$MAIN_RUN/manifest.json" ]]; then
    echo "Main run manifest not found: $MAIN_RUN/manifest.json" >&2
    exit 1
fi
PYTHON="$ROOT/.venv-cuda/bin/python"
[[ -x "$PYTHON" ]] || { echo "Missing CUDA environment: $PYTHON" >&2; exit 1; }
mkdir -p runs
exec 9>runs/candidates_queue.lock
flock -n 9 || { echo 'A candidate queue is already running.' >&2; exit 1; }
SMOKE=results/tuning_smoke_candidates_v1
FULL=results/candidates_cuda_v1
smoke_resume=()
full_resume=()
[[ ! -f "$SMOKE/manifest.json" ]] || smoke_resume=(--resume)
[[ ! -f "$FULL/manifest.json" ]] || full_resume=(--resume)
"$PYTHON" -u experiments/screen_candidates.py --output-dir "$SMOKE" --threads 4 --smoke --after-run "$MAIN_RUN" "${smoke_resume[@]}"
"$PYTHON" -u experiments/screen_candidates.py --output-dir "$FULL" --threads 4 "${full_resume[@]}"
