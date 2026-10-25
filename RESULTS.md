# Measured results

All numbers from a Jetson Orin Nano 8GB (L4T R39.2.1, Ubuntu 24.04.4, CUDA 13.2, sm_87, 6 cores, 7.3 GiB usable).

Model: `Ternary-Bonsai-2-9B-PTQ1_0.gguf` — 1.85 GiB on disk, 8.95 B parameters, 1.75 bits/weight, ternary groups of 128.

Command:

```bash
llama-bench -m <model> -p <CTX> -n 64 -ngl 99 -t 6 -r 1
```

## Context-length sweep

| Context | Prefill (t/s) | Generation (t/s) | Generation vs. 128 |
|---|---|---|---|
| 128 | 360.75 ± 12.78 | 36.57 ± 0.26 | 100.0% |
| 2 048 | 376.12 ± 0.11 | 36.46 ± 0.25 | 99.7% |
| 8 192 | 367.23 ± 0.04 | 36.59 ± 0.24 | 100.1% |
| 16 384 | 354.64 | 36.24 | 99.1% |
| 32 768 | 334.09 | 36.20 | 99.0% |
| 65 536 | 300.56 | 36.10 | 98.7% |

`±` where shown is the standard deviation across repeated runs. Single-rep rows (16K and above) are shown without deviation because each takes minutes; the 8K row's ±0.04 on prefill and ±0.24 on generation indicate the run-to-run spread is small enough that single measurements remain meaningful.

### Reading the table

**Generation is flat.** 512x more context costs 1.3% of generation speed. For an 8 GB board this is the single most important property — you can give the model a genuinely long document and it keeps talking at the same rate.

**Prefill is where context costs you.** 65536 tokens at 300 t/s means ~218 s to ingest a full 64K prompt. That is a one-time cost per conversation, not a per-token tax. For a 4K prompt it is 11 s; for 8K, 22 s.

**Why generation does not degrade.** Naive KV arithmetic assumes every layer caches K and V:

```
2 (K,V) x 32 layers x 4 kv_heads x 256 key_len x 2 bytes = 128 KB per token
```

That predicts 8 GiB of KV at 64K — impossible on this board. The model is `qwen35` with **hybrid attention** (`full_attention_interval = 4`), so only roughly a quarter of layers maintain a full KV cache. Measured steady-state RSS confirms it:

| Context | RSS after run |
|---|---|
| 32 768 | 2 716 MB |
| 65 536 | 2 775 MB |

A 2x context increase cost **59 MB**. The dense-attention formula overestimates KV cost by about 4x on this architecture.

**Transient memory is the real constraint.** During 64K prefill, total used memory briefly hit **6 914 / 7 485 MB**, leaving 83 MB free. It returns to 2.8 GB afterwards. Any long-context run on this board needs a cgroup cage (see README) or that peak can trigger a system-wide OOM that kills unrelated processes.

## Comparison points

| Setup | Params | Quant | Context | Generation | Notes |
|---|---|---|---|---|---|
| This repo — Ternary-Bonsai-2 9B | 8.95 B | PTQ1_0 1.75 bpw | 64K | **36.10 t/s** | Fork required |
| llama.cpp + Qwen3-1.7B | 1.7 B | Q4_K_M | 4K | 40.8 t/s | Stock upstream |
| MLC-LLM + Qwen3-1.7B | 1.7 B | q4f16_1 | 4K | 53.7 t/s | TVM build, different repo |

The 9B ternary model is ~31% slower than a 1.7B 4-bit model in raw tok/s while carrying **5.3x the parameters**. That trade is the entire point of ternary weights on a memory-constrained board: you cannot fit a 4-bit 9B model here at all (it would need ~4.5 GB of weights plus buffers, and prefill peaks would OOM), but a 1.75 bpw 9B model fits with 4.5 GB to spare.

## What does not fit

| Model | Format | Disk | Verdict |
|---|---|---|---|
| Ternary-Bonsai-2 9B | PTQ1_0 | 1.85 GiB | Fits, runs at 36 t/s |
| Ternary-Bonsai-2 27B | PTQ1_0 | 5.95 GB | **Does not fit** |

The 27B budget: 5.95 GB weights + ~300 MB CUDA context + ~400 MB compute buffers + KV + ~1.5 GB system = **8.6 GB+** against 7.31 GiB managed. The 1.3 GB gap is the board's hardware limit, not something quantization or flags can close.

Related hardware fact, since it gets misdiagnosed: a single `cudaMalloc` on this Orin caps at **4.85 GB** (measured 1/2/4 GB OK, 5 GB and 6.5 GB OOM). The frequently-cited "1 GB limit on Jetson" belongs to the CMA/NvMap path (`CmaTotal` 256 MB, `CmaFree` 3.3 MB), which V4L2/ISP/display use. GGML allocates through CUDA, not CMA, so CMA is never the reason a model fails to load.

## Reproducing

```bash
./scripts/bench-sweep.sh
```

Emits the same table. Default sweep takes 10-12 minutes, dominated by the 64K prefill.
