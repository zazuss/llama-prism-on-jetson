#!/usr/bin/env bash
# Run the full context-length sweep and emit a markdown table.
#
# Usage:
#   ./scripts/bench-sweep.sh                     # default sweep
#   CTX="1024 4096 16384" ./scripts/bench-sweep.sh
#   NGL=99 THREADS=6 ./scripts/bench-sweep.sh
#
# Notes:
#   - Each context size is wrapped in its own systemd-run cgroup so a runaway
#     allocation kills the benchmark, not your desktop session.
#   - 64K prefill takes ~4 min on Orin. Total runtime for the default sweep is
#     roughly 10-12 minutes.

set -euo pipefail

MODEL="${MODEL:-$HOME/models/bonsai/Ternary-Bonsai-2-9B-PTQ1_0.gguf}"
BENCH="${BENCH:-$HOME/llama-prism/build/bin/llama-bench}"
CTX="${CTX:-128 2048 8192 16384 32768 65536}"
NGEN="${NGEN:-64}"
NGL="${NGL:-99}"
THREADS="${THREADS:-6}"
REPS="${REPS:-1}"
MEM_HIGH="${MEM_HIGH:-6.5G}"
MEM_MAX="${MEM_MAX:-7G}"

[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }
[ -x "$BENCH" ] || { echo "bench not found: $BENCH" >&2; exit 1; }

export LD_LIBRARY_PATH="$(dirname "$BENCH"):${LD_LIBRARY_PATH:-}"

printf '| Context | Prefill (t/s) | Generation (t/s) | Wall time |\n'
printf '|---|---|---|---|\n'

for P in $CTX; do
  START=$(date +%s)

  OUT=$(systemd-run --user --scope --quiet \
          -p "MemoryHigh=$MEM_HIGH" -p "MemoryMax=$MEM_MAX" \
          "$BENCH" -m "$MODEL" -p "$P" -n "$NGEN" \
          -ngl "$NGL" -t "$THREADS" -r "$REPS" 2>&1 || true)

  # llama-bench rows look like:  | ... | 99 |   pp65536 |   300.56 ± 0.00 |
  PP=$(printf '%s\n' "$OUT" \
        | grep -oE "pp${P}[[:space:]]*\|[[:space:]]*[0-9.]+" \
        | grep -oE '[0-9.]+$' || true)
  TG=$(printf '%s\n' "$OUT" \
        | grep -oE "tg${NGEN}[[:space:]]*\|[[:space:]]*[0-9.]+" \
        | grep -oE '[0-9.]+$' || true)

  [ -n "$PP" ] || PP="failed"
  [ -n "$TG" ] || TG="OOM/err"

  ELAPSED=$(( $(date +%s) - START ))
  printf '| %s | %s | %s | %ss |\n' "$P" "$PP" "$TG" "$ELAPSED"
done

printf '\nSteady-state memory after sweep:\n'
free -m | head -2 | tail -1
