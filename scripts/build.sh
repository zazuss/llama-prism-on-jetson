#!/usr/bin/env bash
# Build PrismML's llama.cpp fork with CUDA for Jetson Orin.
#
# Usage:
#   ./scripts/build.sh              # full build
#   JOBS=5 ./scripts/build.sh       # override parallelism (default 4)
#   SRC_DIR=~/src/llama-prism ./scripts/build.sh
#
# Idempotent: re-running reuses the source tree and incremental cmake state.

set -euo pipefail

TAG="${TAG:-prism-b10809-b3ebc47}"
SRC_DIR="${SRC_DIR:-$HOME/llama-prism}"
JOBS="${JOBS:-4}"
ARCH="${ARCH:-87}"          # sm_87 = Orin. AGX Orin 32/64GB is also 87.
MIRROR="${MIRROR:-https://ghfast.top/}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- prereqs
say "Checking prerequisites"

command -v cmake >/dev/null || die "cmake not found (apt install cmake)"
command -v nvcc  >/dev/null || die "nvcc not found — is the CUDA toolkit installed and on PATH?"

CMAKE_VER=$(cmake --version | head -1 | awk '{print $3}')
NVCC_VER=$(nvcc --version | grep -oP 'release \K[0-9.]+')
printf '  cmake : %s\n  nvcc  : %s\n  arch  : sm_%s\n  jobs  : %s\n' \
  "$CMAKE_VER" "$NVCC_VER" "$ARCH" "$JOBS"

# Jetson reports compute capability via the device tree; warn on obvious mismatch.
if [ -r /proc/device-tree/model ]; then
  MODEL=$(tr -d '\0' < /proc/device-tree/model)
  printf '  device: %s\n' "$MODEL"
  case "$MODEL" in
    *Orin*) [ "$ARCH" = "87" ] || printf '  WARN: Orin detected but ARCH=%s\n' "$ARCH" ;;
  esac
fi

FREE_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt 10 ]; then
  printf '  WARN: only %s GB free on %s — build needs ~10 GB\n' "$FREE_GB" "$HOME"
fi

SWAP_MB=$(free -m | awk '/^Swap:/ {print $2}')
if [ "${SWAP_MB:-0}" -lt 2048 ]; then
  printf '  WARN: swap is %s MB — 2 GB+ recommended on 8 GB boards\n' "${SWAP_MB:-0}"
fi

# ---------------------------------------------------------------- source
if [ -d "$SRC_DIR/.git" ] || [ -f "$SRC_DIR/CMakeLists.txt" ]; then
  say "Reusing existing source tree at $SRC_DIR"
else
  say "Fetching source (tag $TAG)"
  mkdir -p "$(dirname "$SRC_DIR")"
  URL="https://github.com/PrismML-Eng/llama.cpp/archive/refs/tags/${TAG}.tar.gz"

  # Direct clone is unreliable on some networks (TLS truncation mid-pack).
  # Tarball via mirror is smaller and resumable; try it first.
  TMP=$(mktemp -d)
  if ! curl -fL --connect-timeout 20 --max-time 600 \
        -o "$TMP/src.tar.gz" "${MIRROR}${URL}"; then
    printf '  mirror failed, trying direct...\n'
    curl -fL --connect-timeout 20 --max-time 900 \
      -o "$TMP/src.tar.gz" "$URL" \
      || die "download failed. Get the tarball manually and set SRC_DIR to the extracted dir."
  fi

  tar xzf "$TMP/src.tar.gz" -C "$TMP"
  EXTRACTED=$(find "$TMP" -maxdepth 1 -type d -name 'llama.cpp-*' | head -1)
  [ -n "$EXTRACTED" ] || die "unexpected archive layout"
  mv "$EXTRACTED" "$SRC_DIR"
  rm -rf "$TMP"
  say "Source ready at $SRC_DIR"
fi

cd "$SRC_DIR"

# ---------------------------------------------------------------- configure
say "Configuring (CUDA=$ARCH, no curl)"
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release

# ---------------------------------------------------------------- build
say "Building with -j$JOBS (this takes 20-40 min on Orin)"
cmake --build build -j"$JOBS"

# ---------------------------------------------------------------- verify
say "Verifying"

BIN="$SRC_DIR/build/bin"
[ -f "$BIN/llama-cli" ] || die "llama-cli missing — build did not complete"

export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
"$BIN/llama-cli" --version | head -2

printf '\n  CUDA linkage:\n'
if ldd "$BIN/llama-cli" | grep -q 'libcudart'; then
  ldd "$BIN/llama-cli" | grep -E 'cudart|cublas' | sed 's/^/    /'
else
  die "llama-cli is NOT linked against CUDA — CPU-only build. Check that GGML_CUDA=ON took effect."
fi

printf '\n  Ternary kernel tests:\n'
for t in test-ptq1_0-cuda-dot test-pq2-row-shapes; do
  [ -x "$BIN/$t" ] && printf '    OK   %s\n' "$t" || printf '    MISS %s\n' "$t"
done

printf '\n  Devices:\n'
"$BIN/llama-cli" --list-devices 2>&1 | sed 's/^/    /'

cat <<EOF

Build complete.

Add this to your shell:
  export LD_LIBRARY_PATH=$BIN:\$LD_LIBRARY_PATH

Then run:
  $BIN/llama-cli -m ~/models/bonsai/Ternary-Bonsai-2-9B-PTQ1_0.gguf \\
    -p "hello" -n 32 -c 4096 -ngl 99 -t 6 -st --no-display-prompt

Benchmark all context sizes:
  $BIN/llama-bench -m <model> -p 65536 -n 64 -ngl 99 -t 6 -r 1

EOF
