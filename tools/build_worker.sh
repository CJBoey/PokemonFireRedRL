#!/usr/bin/env bash
# Builds the training worker, BUILD_DIR/worker/fireredsim-worker (default BUILD_DIR: build). Fetches the game's
# decompilation into game/ first if needed (tools/fetch_game.sh).
#
#   tools/build_worker.sh [BUILD_DIR]          game code at -O3, no stack guard check (PN_FAST)
#   PGO=1 tools/build_worker.sh [BUILD_DIR]    the same plus profile-guided optimisation: an instrumented build,
#                                              a few minutes of random play to collect profiles, then the final build
#
# Environment: JOBS (parallel compile jobs, default: all CPUs), NICE (default 0), PGO_SECONDS (profiling run,
# default 180), PGO_ENVS (envs in the profiling run, default 64). The first build compiles the whole game (several
# minutes); later builds only recompile what changed. PGO changes the speed, never the behaviour.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
"$ROOT/tools/fetch_game.sh"   # pret/pokefirered into game/, once
B=$(realpath -m "${1:-$ROOT/build}")
JOBS=${JOBS:-$(nproc)}
NICE=${NICE:-0}

configure() {
    cmake -S "$ROOT" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DPN_FAST=ON -DPN_NPROC="$JOBS" "$@" \
        | grep -E "fireredsim game code|Build files"
}

if [ -z "${PGO:-}" ]; then
    configure -DPN_PGO=
    nice -n "$NICE" cmake --build "$B" -j "$JOBS" --target fireredsim-worker
    echo "worker: $B/worker/fireredsim-worker"
    exit 0
fi

echo "== stage 1: instrumented build in $B"
configure -DPN_PGO=generate
nice -n "$NICE" cmake --build "$B" -j "$JOBS" --target fireredsim-worker
find "$B/game-native" -name '*.gcda' -delete
echo "== profiling: random play for ${PGO_SECONDS:-180} s"
FIREREDSIM_WORKER="$B/worker/fireredsim-worker" PYTHONPATH="$ROOT/rl:$ROOT/python" \
    nice -n "$NICE" python3 "$ROOT/tools/pgo_workload.py" --seconds "${PGO_SECONDS:-180}" --envs "${PGO_ENVS:-64}"
n=$(find "$B/game-native" -name '*.gcda' | wc -l)
[ "$n" -gt 0 ] || { echo "build_worker: no profiles (*.gcda) were written" >&2; exit 1; }
echo "== stage 2: optimised build with $n profiles"
configure -DPN_PGO=use
nice -n "$NICE" cmake --build "$B" -j "$JOBS" --target fireredsim-worker
echo "worker: $B/worker/fireredsim-worker"
