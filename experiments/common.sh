#!/usr/bin/env bash
# Shared gate + runner for every formal experiment. Source this, don't run it.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Rule 3: raw results must map to committed code.
if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: uncommitted changes. Commit first so results map to a commit:" >&2
  git status --short >&2
  exit 1
fi
COMMIT="$(git rev-parse HEAD)"
echo "== code commit: $COMMIT"

# Fresh Release build of exactly this commit, in its own build dir.
cmake -S . -B build-exp -DCMAKE_BUILD_TYPE=Release
cmake --build build-exp -j

# Never measure a build that fails its own correctness tests.
ctest --test-dir build-exp --output-on-failure

run_experiment() {
  local name="$1"
  shift
  local out="results/raw/${name}"
  mkdir -p "$out"
  ./build-exp/gemmx_benchmarks --experiment "$name" --git-commit "$COMMIT" \
    --out "$out" "$@"
}
