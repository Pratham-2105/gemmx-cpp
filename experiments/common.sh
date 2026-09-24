#!/usr/bin/env bash
# Shared setup + runner for every formal experiment. Source this, don't run it.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Record which code version produced the results. Never blocks.
# Only CODE paths count: notebook edits and result files don't matter.
COMMIT="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain -- src include benchmarks tests CMakeLists.txt)" ]]; then
  COMMIT="${COMMIT}-dirty"
  echo "NOTE: uncommitted code changes; runs labelled ${COMMIT}" >&2
fi
echo "== code commit: $COMMIT"

# Refuse to measure on battery: laptops throttle hard when unplugged.
if command -v powershell.exe >/dev/null 2>&1; then
  line="$(powershell.exe -NoProfile -Command \
    'Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SystemInformation]::PowerStatus.PowerLineStatus' \
    2>/dev/null | tr -d '\r')"
  if [[ "$line" != "Online" ]]; then
    echo "ERROR: not on AC power (PowerLineStatus=$line). Plug in first." >&2
    exit 1
  fi
fi

# Fresh Release build, in its own build dir.
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
