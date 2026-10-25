# llama-prism on Jetson Orin

Build recipe and benchmarks for [PrismML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) with **CUDA** support on Jetson Orin (sm_87), running the Ternary-Bonsai-2 9B ternary model.

**The problem this solves:** PrismML ships prebuilt binaries for x86 and `ubuntu-arm64`, but the arm64 package is **CPU-only**. There is no CUDA build for Jetson anywhere — not in their releases, not in `jetson-containers`, not on jetson-ai-lab. If you want `PTQ1_0` on an Orin GPU, you compile it yourself. This repo is that recipe, plus the measured numbers you can expect.

## Platform

| Item | Value |
|---|---|
| Device | Jetson Orin Nano 8GB (P3767) |
| L4T | R39.2.1 |
| Ubuntu | 24.04.4 LTS |
| CUDA | 13.2 |
| Compute capability | **sm_87** |
| CPU cores | 6 |
| Usable RAM | **7.3 GiB** (7.68 GiB present) |

## Result summary

Ternary-Bonsai-2 **9B** (`PTQ1_0`, 1.85 GiB, 8.95 B params, 1.75 bpw) on a 7.3 GiB Orin:

| Context | Prefill (t/s) | Generation (t/s) |
|---|---|---|
| 128 | 360.75 ± 12.78 | 36.57 ± 0.26 |
| 2 048 | 376.12 ± 0.11 | 36.46 ± 0.25 |
| 8 192 | 367.23 ± 0.04 | 36.59 ± 0.24 |
| 16 384 | 354.64 | 36.24 |
| 32 768 | 334.09 | 36.20 |
| **65 536** | **300.56** | **36.10** |

**Context length is nearly free on this model.** From 128 to 65 536 tokens — a 512x increase — generation speed drops only 1.3% (36.57 → 36.10 t/s). Prefill degrades faster (360 → 300 t/s, −17%), which only means filling a full 64K prompt takes ~218 s. After that, every generated token costs the same as it did at 128 context.

This is not the behaviour you would get from a dense-attention model. The reason is architectural: `qwen35` uses hybrid attention (`full_attention_interval = 4`), so only a fraction of layers build a KV cache. Steady-state RSS stays at **~2.8 GB** even at 64K context — barely above the 32K case (2.7 GB). Naive KV arithmetic (`2 x layers x kv_heads x key_len x 2 bytes` on every layer) overestimates the cost by roughly 4x.

## Why 9B and not 27B

The 27B variant is 5.95 GB on disk. Model + CUDA context + compute buffers + system needs >8.6 GB on this board, which presents only 7.31 GiB managed memory. That 1.3 GB gap is a hardware limit, not a tuning problem — it is closed. 9B is the ceiling for a 7.3 GiB Orin.

## Build

Prerequisites: CUDA toolkit matching your L4T, `cmake` >= 3.20, `g++`, ~10 GB free disk, and ideally 4 GB+ swap.

```bash
# 1. Get the source at the tagged build.
#    Direct git clone from GitHub is unreliable on some CN networks
#    (TLS truncation: GnuTLS recv error -110, partial pack).
#    If clone fails, use the tarball via a mirror:
curl -L -o llama-prism.tar.gz \
  https://ghfast.top/https://github.com/PrismML-Eng/llama.cpp/archive/refs/tags/prism-b10809-b3ebc47.tar.gz
tar xzf llama-prism.tar.gz && cd llama.cpp-prism-b10809-b3ebc47

# 2. Configure for Orin.
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release

# 3. Build. -j4 is the safe ceiling on a 6-core Orin with a desktop session;
#    -j5+ competes for RAM and has OOM-killed other processes.
cmake --build build -j4
```

Verify the CUDA backend actually linked — this is the step people skip:

```bash
export LD_LIBRARY_PATH=$PWD/build/bin:$LD_LIBRARY_PATH
./build/bin/llama-cli --list-devices
# expect: CUDA0: Orin (7485 MiB, xxxx MiB free)

ldd ./build/bin/llama-cli | grep -E 'cudart|cublas'
# expect: libcudart.so.13, libcublas.so.13
```

Also confirm the ternary kernels were built — these binaries only exist if `PTQ1_0` support was compiled in:

```bash
ls build/bin/test-ptq1_0-cuda-dot build/bin/test-pq2-row-shapes
```

## Model

```bash
# 9B, PTQ1_0 (smallest, 1.85 GiB)
huggingface-cli download prism-ml/Ternary-Bonsai-2-9B-gguf \
  Ternary-Bonsai-2-9B-PTQ1_0.gguf --local-dir ~/models/bonsai
```

Format choice matters:

| Format | Size | Runs on stock llama.cpp? | Notes |
|---|---|---|---|
| `PTQ1_0` (id 143) | 1.85 GiB | **No** | True ternary, tightest memory — use this on Orin |
| `PQ2_0` (id 142) | ~2.4 GB | No | Model card's default, faster dequant, more RAM |
| `Q2_0` | ~2.6 GB | Yes | Compatibility layer, not true ternary |

`PTQ1_0` and `PQ2_0` require this fork. Stock upstream llama.cpp — including every `jetson-containers` and jetson-ai-lab build — cannot read them.

## Run

```bash
export LD_LIBRARY_PATH=$PWD/build/bin:$LD_LIBRARY_PATH

# Single-turn inference. -st is REQUIRED: this fork drops -no-cnv,
# and without -st llama-cli sits in interactive mode printing prompts.
./build/bin/llama-cli \
  -m ~/models/bonsai/Ternary-Bonsai-2-9B-PTQ1_0.gguf \
  -p "Explain ternary quantization in one sentence." \
  -n 64 -c 8192 -ngl 99 -t 6 -st --no-display-prompt --no-warmup
```

Reproduce the table:

```bash
for P in 128 2048 8192 16384 32768 65536; do
  ./build/bin/llama-bench \
    -m ~/models/bonsai/Ternary-Bonsai-2-9B-PTQ1_0.gguf \
    -p $P -n 64 -ngl 99 -t 6 -r 1
done
```

## Memory safety

On a 7.3 GiB board shared with a desktop session, put inference in its own cgroup so a runaway allocation kills the benchmark instead of your session:

```bash
systemd-run --user --scope -p MemoryHigh=6.5G -p MemoryMax=7G \
  ./build/bin/llama-bench -m <model> -p 65536 -n 64 -ngl 99 -t 6 -r 1
```

The 64K prefill peaks at ~6.9 GB momentarily, then settles back to 2.8 GB. Without the cgroup, an OOM during that peak can take down unrelated processes (it killed a Hermes gateway on this board before the cage was added).

Never compile and run inference at the same time on 8 GB. `cc1plus` + `nvcc` + a loaded model exceeds 7.3 GiB; the kernel OOM killer does not care which process is important.

## Pitfalls

1. **No CUDA prebuilt exists.** Do not waste time searching releases, `jetson-containers`, or PyPI. The `bin-ubuntu-arm64` asset is CPU-only.
2. **`-no-cnv` does not exist in this fork.** Use `-st` / `--single-turn`.
3. **Omitting `-c` allocates a huge default context** and can consume ~5 GB of RSS before your prompt even runs. Always pass `-c`.
4. **`CMA` limits are irrelevant here** but commonly blamed: single `cudaMalloc` on Orin caps at **4.85 GB** (measured), while the 1 GB ceiling people remember is the CMA/NvMap path (`CmaTotal` 256 MB) used by V4L2/ISP/display. GGML does not touch CMA.
5. **Do not mix `.so` files between stock llama.cpp and this fork.** `ggml` ABI differs; you get silent wrong output or a crash.
6. **`LD_LIBRARY_PATH` must point at `build/bin`** or the binary loads system `libggml` and fails with a version skew.

## License

Scripts and documentation here: MIT. The fork and model weights carry their own licenses — see [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) and the model card.
