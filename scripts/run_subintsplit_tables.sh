#!/usr/bin/env bash
# scripts/run_subintsplit_tables.sh
set -euo pipefail

################################################################################
# One command to go from nothing to rendered SubIntSplit result tables:
#   generate large datasets → configure → build the benchmark → run it → render.
#
#   scripts/run_subintsplit_tables.sh                  # the whole pipeline
#   scripts/run_subintsplit_tables.sh --skip-bench     # re-render existing CSV
################################################################################

################################################################################
# locate repo root, then work from there
################################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

################################################################################
# defaults & flag parsing
################################################################################
ROWS=1048576                       # 2^20
BUILD_DIR="${BUILD_DIR:-bench-build}"
OUT_MD="tables/subintsplit.md"
SKIP_GENERATE=0
SKIP_BUILD=0
SKIP_BENCH=0
FORCE_GENERATE=0

RESULT_CSV="benchmark/result/subintsplit/subintsplit.csv"
GENERATOR="data/generated/subintsplit/generate.py"
RENDERER="scripts/render_subintsplit_tables.py"
DATASETS=(snowflake_i64 tpch_partkey_i32 ipv4_i32)

usage() {
  cat <<'EOF'
Usage: scripts/run_subintsplit_tables.sh [options]

Generates the large SubIntSplit benchmark datasets, builds and runs the
bench_subintsplit benchmark, and renders the Markdown result tables.

Options:
  --rows N          Rows per generated dataset (default: 1048576 = 2^20).
  --build-dir DIR   CMake build directory (default: bench-build, or $BUILD_DIR).
  --out FILE        Markdown output path (default: tables/subintsplit.md).
  --skip-generate   Reuse the datasets already in <build-dir>/subintsplit-data.
  --skip-build      Do not configure or build; use the existing binary.
  --skip-bench      Do not run the benchmark; render the existing CSV.
                    (the fast inner loop when iterating on table formatting)
  --force-generate  Regenerate the datasets even if they look complete.
  -h, --help        Show this help and exit.

Environment:
  FASTLANES_DATA_DIR  Must point at an existing FastLanes test-data checkout.
                      If unset, <repo>/build/_deps/data-src is used when it
                      exists; otherwise the script fails rather than letting
                      CMake re-download ~1.1 GB of test data.
  BUILD_DIR           Same as --build-dir.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rows)
      [[ $# -ge 2 ]] || { echo "error: --rows needs a value" >&2; exit 1; }
      ROWS="$2"; shift 2 ;;
    --build-dir)
      [[ $# -ge 2 ]] || { echo "error: --build-dir needs a value" >&2; exit 1; }
      BUILD_DIR="$2"; shift 2 ;;
    --out)
      [[ $# -ge 2 ]] || { echo "error: --out needs a value" >&2; exit 1; }
      OUT_MD="$2"; shift 2 ;;
    --skip-generate)  SKIP_GENERATE=1; shift ;;
    --skip-build)     SKIP_BUILD=1;    shift ;;
    --skip-bench)     SKIP_BENCH=1;    shift ;;
    --force-generate) FORCE_GENERATE=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    *)
      echo "error: unknown option '$1'" >&2
      echo >&2
      usage >&2
      exit 1 ;;
  esac
done

if [[ ! "$ROWS" =~ ^[0-9]+$ ]] || [[ "$ROWS" -eq 0 ]]; then
  echo "error: --rows must be a positive integer, got '$ROWS'" >&2
  exit 1
fi

DATA_DIR="$BUILD_DIR/subintsplit-data"
BENCH_BIN="$BUILD_DIR/benchmark/bench_subintsplit/bench_subintsplit"

# absolute forms (the benchmark is run from $REPO_ROOT, and the dataset root is
# handed to it through the environment, so both must be unambiguous)
abspath() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$REPO_ROOT/$1" ;; esac; }
DATA_DIR_ABS="$(abspath "$DATA_DIR")"
BENCH_BIN_ABS="$(abspath "$BENCH_BIN")"

################################################################################
echo "── Step 1: guard FASTLANES_DATA_DIR ──────────────────────────────────"
################################################################################
# This must happen before any cmake invocation: without the variable CMake
# clones a ~1.1 GB test-data repository, which must never happen implicitly.
if [[ -z "${FASTLANES_DATA_DIR:-}" ]]; then
  fallback="$REPO_ROOT/build/_deps/data-src"
  if [[ -d "$fallback" ]]; then
    export FASTLANES_DATA_DIR="$fallback"
    echo "✔ FASTLANES_DATA_DIR was unset – using existing checkout:"
    echo "    $FASTLANES_DATA_DIR"
  else
    cat >&2 <<EOF
✖ FASTLANES_DATA_DIR is unset and the usual fallback does not exist:
    $fallback

  Without it CMake would download the ~1.1 GB FastLanes test-data repository.
  Point the variable at an existing checkout and re-run, e.g.:

    export FASTLANES_DATA_DIR=\$PWD/build/_deps/data-src
    $0
EOF
    exit 1
  fi
elif [[ ! -d "$FASTLANES_DATA_DIR" ]]; then
  echo "✖ FASTLANES_DATA_DIR points at a non-existent directory:" >&2
  echo "    $FASTLANES_DATA_DIR" >&2
  echo "  Refusing to continue – CMake would re-download ~1.1 GB." >&2
  exit 1
else
  export FASTLANES_DATA_DIR
  echo "✔ FASTLANES_DATA_DIR = $FASTLANES_DATA_DIR"
fi

################################################################################
echo
echo "── Step 2: generate benchmark datasets ($ROWS rows) ──────────────────"
################################################################################
datasets_complete() {
  local d
  for d in "${DATASETS[@]}"; do
    [[ -f "$DATA_DIR/$d/generated.csv" && -f "$DATA_DIR/$d/schema.json" ]] || return 1
  done
  return 0
}

if [[ "$SKIP_GENERATE" -eq 1 ]]; then
  echo "↷ --skip-generate: using whatever is in $DATA_DIR"
elif [[ "$FORCE_GENERATE" -eq 0 ]] && datasets_complete; then
  echo "✔ datasets already present in $DATA_DIR – skipping (use --force-generate)"
else
  mkdir -p "$DATA_DIR"
  # --out-dir keeps the big (~40 MB) datasets out of the committed fixtures.
  python3 "$GENERATOR" --rows "$ROWS" --out-dir "$DATA_DIR"
  echo "✔ datasets written to $DATA_DIR"
fi

# Real-world snowflake IDs are optional: they need pyarrow and the sibling
# EncodingsPlayground checkout, neither of which is guaranteed to be present. Best
# effort, never fatal -- the benchmark itself skips this dataset if it is absent.
REAL_EXTRACTOR="data/generated/subintsplit/extract_real_snowflake.py"
REAL_DIR="$DATA_DIR/snowflake_i64_real"
if [[ "$SKIP_GENERATE" -eq 0 ]]    && { [[ "$FORCE_GENERATE" -eq 1 ]] || [[ ! -f "$REAL_DIR/generated.csv" ]]; }; then
  if python3 -c "import pyarrow" >/dev/null 2>&1; then
    if python3 "$REAL_EXTRACTOR" --out-dir "$DATA_DIR" --rows "$ROWS"; then
      echo "✔ real-world snowflake dataset written to $REAL_DIR"
    else
      echo "↷ real-world snowflake dataset unavailable (source parquet not found) – skipping"
      echo "  snowflake_i64_real is FastLanes' default/headline SubIntSplit dataset;"
      echo "  falling back to the synthetic snowflake_i64 reference dataset instead."
      echo "  See 'Data source' under Reproducing in docs/subintsplit.md to get the real one."
    fi
  else
    echo "↷ pyarrow not installed – skipping the real-world snowflake dataset"
    echo "  snowflake_i64_real is FastLanes' default/headline SubIntSplit dataset;"
    echo "  falling back to the synthetic snowflake_i64 reference dataset instead."
    echo "  See 'Data source' under Reproducing in docs/subintsplit.md to get the real one."
  fi
fi

################################################################################
echo
echo "── Step 3: configure CMake ($BUILD_DIR) ──────────────────────────────"
################################################################################
if [[ "$SKIP_BUILD" -eq 1 ]]; then
  echo "↷ --skip-build: leaving $BUILD_DIR untouched"
else
  cache="$BUILD_DIR/CMakeCache.txt"
  if [[ -f "$cache" ]]; then
    existing_type="$(sed -n 's/^CMAKE_BUILD_TYPE:[^=]*=//p' "$cache" | head -n 1)"
    if [[ -n "$existing_type" && "$existing_type" != "Release" ]]; then
      echo "⚠  WARNING: $cache has CMAKE_BUILD_TYPE=$existing_type (not Release)."
      echo "⚠  Timings from a non-Release build are meaningless."
      echo "⚠  Delete $BUILD_DIR to reconfigure from scratch."
    fi
  fi
  cmake -S . -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DFLS_BUILD_BENCHMARKING=ON
  echo "✔ configured"
fi

################################################################################
echo
echo "── Step 4: build target bench_subintsplit ────────────────────────────"
################################################################################
if [[ "$SKIP_BUILD" -eq 1 ]]; then
  echo "↷ --skip-build: reusing $BENCH_BIN"
else
  # Only this target: a full benchmark build is minutes of unrelated work.
  cmake --build "$BUILD_DIR" --target bench_subintsplit --parallel
  echo "✔ built $BENCH_BIN"
fi

################################################################################
echo
echo "── Step 5: run bench_subintsplit ─────────────────────────────────────"
################################################################################
if [[ "$SKIP_BENCH" -eq 1 ]]; then
  echo "↷ --skip-bench: rendering the existing $RESULT_CSV"
else
  if [[ ! -x "$BENCH_BIN_ABS" ]]; then
    echo "✖ benchmark binary not found (or not executable): $BENCH_BIN" >&2
    echo "  Re-run without --skip-build." >&2
    exit 1
  fi
  echo "   FLS_SUBINTSPLIT_DATA_DIR=$DATA_DIR_ABS"
  FLS_SUBINTSPLIT_DATA_DIR="$DATA_DIR_ABS" "$BENCH_BIN_ABS"
  echo "✔ results written to $RESULT_CSV"
fi

################################################################################
echo
echo "── Step 6: render Markdown tables ────────────────────────────────────"
################################################################################
if [[ ! -f "$RESULT_CSV" ]]; then
  echo "✖ no benchmark results at $RESULT_CSV" >&2
  echo "  Re-run without --skip-bench." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_MD")"
python3 "$RENDERER" --csv "$RESULT_CSV" --out "$OUT_MD"

echo
echo "✅  SubIntSplit tables written to $OUT_MD"
echo "    ($ROWS rows/dataset · datasets $DATA_DIR · raw results $RESULT_CSV)"
