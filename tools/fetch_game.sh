#!/usr/bin/env bash
# Fills game/ with the pret/pokefirered decompilation the simulator is built from. The repository tracks only
# its own files in game/; this downloads pret's files at the commit in game/UPSTREAM.txt and applies
# game/native/pokefirered.patch, the simulator's changes to them. Needs git and network access once.
# tools/build_worker.sh runs it; it does nothing when game/ already holds that commit (game/.pret_commit).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
GAME=$ROOT/game
read -r _ URL _ SHA < "$GAME/UPSTREAM.txt"
[ "$(cat "$GAME/.pret_commit" 2>/dev/null)" = "$SHA" ] && exit 0

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
echo "fetching $URL at $SHA"
git init -q "$TMP/repo"
git -C "$TMP/repo" fetch -q --depth 1 "$URL" "$SHA"
# Patch a staging copy (outside any git repository, so git apply works as a plain patch tool), then copy it over:
# an interrupted fetch never leaves a half-patched game/.
mkdir "$TMP/game"
git -C "$TMP/repo" archive "$SHA" | tar -x -m -C "$TMP/game"
git -C "$TMP/game" apply "$GAME/native/pokefirered.patch"
cp -a "$TMP/game/." "$GAME/"
echo "$SHA" > "$GAME/.pret_commit"
echo "game/: pret/pokefirered $SHA with game/native/pokefirered.patch applied"
