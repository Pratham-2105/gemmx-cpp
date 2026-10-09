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

# Two FRESH Release builds of the same code. The build dirs are deleted first,
# so a stale binary from older code can never be measured by accident.
#   build-exp/         portable x86-64 (compiler may use SSE2 only)
#   build-exp-native/  -march=native   (compiler may use AVX2/FMA)
rm -rf build-exp build-exp-native

cmake -S . -B build-exp -DCMAKE_BUILD_TYPE=Release -DGEMMX_NATIVE=OFF
cmake --build build-exp -j
ctest --test-dir build-exp --output-on-failure

cmake -S . -B build-exp-native -DCMAKE_BUILD_TYPE=Release -DGEMMX_NATIVE=ON
cmake --build build-exp-native -j
ctest --test-dir build-exp-native --output-on-failure

_run_with() {
  local build_dir="$1" name="$2"
  shift 2
  local out="results/raw/${name}"
  mkdir -p "$out"
  "./${build_dir}/gemmx_benchmarks" --experiment "$name" --git-commit "$COMMIT" \
    --out "$out" "$@"
}

run_experiment() { _run_with build-exp "$@"; }
run_experiment_native() { _run_with build-exp-native "$@"; }
