#!/usr/bin/env bash
# Starts a training run in the background, detached from the terminal: python3 -m pokerl.train with the v76 defaults
# plus any flags given, its output appended to rl/runs/<name>/train.out.
#
#   tools/train.sh NAME [pokerl.train flags]      a fresh run rl/runs/NAME (e.g. tools/train.sh v76_repro)
#   tools/train.sh --resume NAME [flags]          continue it from its last checkpoint
#
# Stop a run with `kill -INT <pid>`: the trainer finishes the running update and saves a checkpoint. The worker
# is $FIREREDSIM_WORKER, else build/worker/fireredsim-worker.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ $# -ge 1 ] || { sed -n '2,10p' "$0"; exit 2; }
if [ "$1" = "--resume" ]; then
    [ $# -ge 2 ] || { echo "train.sh: --resume needs a run name" >&2; exit 2; }
    NAME=$2; ARGS=("$@")
else
    NAME=$1; ARGS=(--run-name "$1" "${@:2}")
fi
export FIREREDSIM_WORKER=${FIREREDSIM_WORKER:-$ROOT/build/worker/fireredsim-worker}
[ -x "$FIREREDSIM_WORKER" ] || { echo "train.sh: no worker at $FIREREDSIM_WORKER (run tools/build_worker.sh)" >&2; exit 1; }
mkdir -p "$ROOT/rl/runs/$NAME"
cd "$ROOT"
# Popen with a new session: a job started with `&` from a non-interactive shell ignores SIGINT, so `kill -INT`
# could not stop it cleanly.
PYTHONPATH=rl:python python3 - "$ROOT/rl/runs/$NAME/train.out" "${ARGS[@]}" <<'PY'
import subprocess, sys
with open(sys.argv[1], "ab", buffering=0) as log:
    p = subprocess.Popen([sys.executable, "-u", "-m", "pokerl.train", *sys.argv[2:]], stdin=subprocess.DEVNULL,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
print(f"pid {p.pid}, log {sys.argv[1]} (stop: kill -INT {p.pid})")
PY
