#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/shuffle_bench.cpp"
BIN="$SCRIPT_DIR/shuffle_bench"

# Build
ARCH_FLAGS=""
if [ "$(uname -m)" = "x86_64" ]; then
  ARCH_FLAGS="-msse4.2"
fi
BUILD_CMD=(g++ -std=c++20 -O2 $ARCH_FLAGS -pthread ${EXTRA_CXXFLAGS:-} -o "$BIN" "$SRC")
echo "${BUILD_CMD[*]}"
"${BUILD_CMD[@]}"
echo "Build OK: $BIN"
echo

if [ "${FAST:-0}" == "1" ]; then
  THREADS=(64)
  ROW_SIZES=(64)
  RING_KS=(1)
  DISTS=(flat)
elif [ "${FAST:-0}" == "2" ]; then
  THREADS=(64)
  ROW_SIZES=(32 128)
  RING_KS=(1 4)
  DISTS=(flat normal)
else
  THREADS=(1 2 4 8 16 32 64 72)
  ROW_SIZES=(32 64 128 256)
  RING_KS=(1 2 3 4)
  DISTS=(flat normal)
fi
ROWS=8192
CHUNKS_PER_PRODUCER=1000
REPEATS=5
MAX_DATA_GB=128 # skip batch when total data exceeds this

run() {
  [ "${V:-0}" != "0" ] && echo "$*" >&2
  "$@"
}

for dist in "${DISTS[@]}"; do
  for rs in "${ROW_SIZES[@]}"; do
    for t in "${THREADS[@]}"; do
      CHUNKS=$((CHUNKS_PER_PRODUCER * t))
      TOTAL_GB=$((CHUNKS * ROWS * rs / 1000000000))
      echo "M=$t N=$t rows=$ROWS rs=$rs chunks=$CHUNKS r=$REPEATS dist=$dist"
      if [ "$TOTAL_GB" -lt "$MAX_DATA_GB" ]; then
        run "$BIN" BC "$t" "$t" "$ROWS" "$rs" "$CHUNKS" "$REPEATS" 2 "$dist"
      else
        echo "  (skipping batch: ${TOTAL_GB} GB > ${MAX_DATA_GB} GB limit)"
        run "$BIN" C "$t" "$t" "$ROWS" "$rs" "$CHUNKS" "$REPEATS" 2 "$dist"
      fi
      for k in "${RING_KS[@]}"; do
        run "$BIN" R "$t" "$t" "$ROWS" "$rs" "$CHUNKS" "$REPEATS" "$k" "$dist"
      done
    done
  done
done
