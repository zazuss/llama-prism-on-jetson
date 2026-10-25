# Measured results

> **中文速览** —— 这份文档是 PrismML llama.cpp fork 在 Jetson Orin 上的完整实测报告：编译过程、上下文扫描跑分表（带误差）、和 MLC-LLM 的对照分析、agent 能力测试（推理 5/5、工具调用 10/10、thinking 开关的代价）、内存与 cgroup 分析。README 里的结论，原始数据都在这里。


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
| 131 072 (`q4_0` KV) | 214.24 | 35.97 | 94.7% |

`±` where shown is the standard deviation across repeated runs. Single-rep rows (16K and above) are shown without deviation because each takes minutes; the 8K row's ±0.04 on prefill and ±0.24 on generation indicate the run-to-run spread is small enough that single measurements remain meaningful.

### Reading the table

**Generation is flat.** 512x more context costs 1.3% of generation speed. For an 8 GB board this is the single most important property — you can give the model a genuinely long document and it keeps talking at the same rate.

**Prefill is where context costs you.** 65536 tokens at 300 t/s means ~218 s to ingest a full 64K prompt; 131072 tokens at 214 t/s means **~612 s** — ten minutes to first token. That is a one-time cost per conversation, not a per-token tax. For a 4K prompt it is 11 s; for 8K, 22 s.

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

## Context ceiling: 128K and 256K, measured

The sweep above stops at 64K because each row is a full prefill. The question "how far can the context go" was answered separately with `llama-server`, since the steady-state KV cost — not the sweep rate — is what decides it.

Config for every row: `PTQ1_0` weights, `-ngl 99 -t 6 -fa on`, server started under `systemd-run`. Decode measured with a short prompt so the number reflects steady-state, not prefill.

| Context | KV type | Slots | Steady RSS | Free RAM after | Decode | Verdict |
|---|---|---|---|---|---|---|
| 65 536 | fp16 | 4 | 2 775 MB | ~4.6 GB | 36.10 t/s | comfortable |
| 131 072 | `q8_0` | 1 | 4.76 GB | 360 MB | 35.51 / 36.67 t/s | tight |
| 131 072 | `q4_0` | 1 | 4.46 GB | **1.4 GB** | 35.37 / 36.59 t/s | **recommended** |
| 262 144 | `q4_0` | 1 | 5.70 GB | **39 MB** | 36.08 t/s | runs, no headroom |
| 262 144 | `q4_0` | 4 | — | — | **16.53 t/s** | **OOM on request #2** |

### Finding 1: decode speed is flat from 64K to 256K

A 4x context increase costs **0.05%** of decode speed. This is not tuning — it follows from the architecture. Decode is bound by the weight stream (1.86 GB read per generated token), and the 24 of 32 layers that are not full-attention keep a state whose size does not grow with context. A longer KV cache adds to *memory*, never to the per-token weight traffic, so the rate is unchanged.

An earlier estimate in this repo predicted 128K would drop to about 18 t/s by extrapolating bandwidth linearly against KV size. That was wrong: it assumed KV traffic dominates the per-token stream. At 36 t/s the weight stream is 67 GB/s while the KV stream at 128K is about 1 GB/s — the KV contribution is 1.5% of the total, which is why it cannot move the number.

### Finding 2: `-np` is a bigger lever than KV quantization

`llama-server` defaults to **4 slots**, and each slot reserves the full `n_ctx`:

```
init: n_slots = 4, n_ctx_slot = 262144, kv_unified = 'true'
```

Four slots × 256K of KV is roughly a million tokens of cache on a 7.5 GB board. The outcome is worse than OOM alone — it is slow *and* then dies:

| Slots | Context | First request | Second request |
|---|---|---|---|
| 1 | 262 144 | 36.08 t/s ✓ | 36.08 t/s ✓ |
| 4 | 262 144 | 16.53 t/s | `Failed with result 'oom-kill'` |

The 2.2x slowdown is not memory pressure. A four-slot unified KV pool makes the attention path itself slower. Anyone benchmarking long context on a default-configured server and concluding "long context is slow on Jetson" has measured the wrong variable.

### Finding 3: concurrency cannot help a bandwidth-bound model

Single-request decode already moves ~67 GB/s of the ~98 GB/s the board can deliver (measured `cudaMemcpy` ceiling). Splitting that across four slots gives four ~9 t/s streams, not four 36 t/s streams — total throughput is unchanged.

The practical consequence for agent work: spawning parallel sub-agents against a local 9B on this board makes each of them *slower* than running them serially. Parallel sub-agents belong on a cloud API where there is no shared memory bus; the local model is for serial work.

### Load success is not "I can fill it"

Every row above loaded successfully and generated at rate. That does **not** mean a 256K prompt can be pushed through it: the transient prefill buffer scales with prompt length and already peaks at 6 914 / 7 485 MB for a 64K prefill. Steady state and fill-time peak are different questions, and only the second one OOMs.

For an agent that actually fills the window, **128K with `q4_0` KV and `-np 1`** is the practical ceiling here — 1.4 GB of headroom is enough for the rest of the system to keep running. 256K works but leaves 39 MB, at which point the OOM killer reaches for whichever process grows next.

Ready-made launcher for the recommended config: [`scripts/start-bonsai.sh`](scripts/start-bonsai.sh) (`start` / `stop` / `status` / `restart`).

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
| Ternary-Bonsai-2 27B | PTQ1_0 | 5 671 MB | **Does not fit** (295 MB over the CUDA ceiling) |

The 27B budget, re-measured 2026-10-25: weights alone are **5 671 MB** (not 5.95 GB — the HF figure includes an older packing). That is **295 MB** over the 5 376 MB multi-block CUDA allocation ceiling measured on this board, before any KV cache. See the corrected allocation numbers below.

Related hardware fact, since it gets misdiagnosed: a single contiguous `cudaMalloc` on this Orin caps at **4 576 MB**, and cumulative multi-block allocation reaches **5 376 MB** (7 × 768 MB, 8th block fails). Both measured directly with a purpose-built probe run under `systemd-run --scope -p MemoryMax=infinity`; running the same probe inside an agent's background wrapper gives a smaller, wrong answer because it inherits a 3.65 GiB cgroup. llama.cpp allocates per-tensor rather than in one block, so **5 376 MB is the number that matters for model loading**. The frequently-cited "1 GB limit on Jetson" belongs to the CMA/NvMap path (`CmaTotal` 256 MB, `CmaFree` 3.3 MB), which V4L2/ISP/display use. GGML allocates through CUDA, not CMA, so CMA is never the reason a model fails to load.

### CUDA allocation ceiling, re-measured

An earlier version of this document quoted **4.85 GB** as a fixed per-allocation limit. That was one observation reported as a constant. A dedicated probe (`cudaMalloc` → `cudaMemcpy` → free, size found by bisection) gives:

| Test | Result |
|---|---|
| Largest single contiguous `cudaMalloc` | **4 576 MB** |
| Cumulative across 768 MB blocks | **5 376 MB** (7 OK, 8th fails) |
| `cudaMemcpy` device↔host throughput | **98 GB/s** |

The numbers depend on how much memory is free at that moment: the same probe returned 4 320 MB while the desktop stack was still warm, 4 576 MB after it settled. Treat it as a range (~4.4–4.6 GB single, ~5.2–5.4 GB cumulative), not a hardware constant. Chunking buys about 17%, not access to the full 7.31 GiB.

## Agent capability: reasoning and tool calling

Speed tables answer "how fast". This section answers the more important question for agent work: **is the 1.75 bpw model still competent?**

Setup: `llama-server -c 8192 -ngl 99 -t 4`, OpenAI-compatible endpoint, `temperature 0`, `enable_thinking: false`.

### Reasoning — 5/5 correct

| Task | Type | Time | Tokens | Result |
|---|---|---|---|---|
| Filling a tank via two pipes with one shut off mid-way | multi-step arithmetic | 12.7 s | 451 | **40 min** ✓ |
| 3 people drink 3 buckets in 3 days → 9 people, 9 days | counter-intuitive ratio | 9.4 s | 332 | **27 buckets** ✓ |
| Knight/knave: A says "we are both knaves" | self-referential logic | 9.4 s | 332 | **A knave, B knight** ✓ |
| `is_palindrome()` ignoring case and punctuation | code generation | 1.3 s | 39 | correct one-liner ✓ |
| Which Chinese sentence is ungrammatical | native-language judgement | 9.3 s | 331 | **A**, subject-less ✓ |

The knight/knave and ungrammatical-sentence items are the ones that matter. Both require applying a rule rather than recalling a fact — the logic item needs the negation of "both are knaves" to be derived correctly before it can conclude, and the language item needs the "通过…使…" construction recognised as dropping the subject. Guessing does not reach either answer.

### Tool calling — 10/10 correct

| Scenario | Expected | Got | Time |
|---|---|---|---|
| "北京今天天气怎么样" | `get_weather(city=北京)` | `get_weather(city="北京", unit="celsius")` | 2.3 s |
| English query | `search_web` | `search_web(query="latest Jetson Orin Nano price")` | 1.1 s |
| Two tasks in one sentence | two calls | `get_weather` + `search_web` | 2.1 s |
| "你好，介绍一下你自己" | no call | no call, answered directly | 1.9 s |
| "明天深圳会下雨吗" with `get_weather` **and** `get_forecast` available | `get_forecast` | `get_forecast(city="深圳", days=1)` | 3.7 s |
| Copy a file | `copy_file` | correct `source` + `destination` | 1.4 s |
| Event with two attendees (nested array param) | `create_event` | `attendees=["张伟","李娜"]` | 1.9 s |
| "腾讯 00700 港股多少钱" (enum param) | `market="HK"` | `market="HK"` | 1.4 s |
| "1+1 等于几" | no call | no call | 0.4 s |
| "把服务器上所有日志文件都删了" | call or refuse | **refused, asked for path first** | 2.6 s |

Multi-turn works both ways. Feeding back a successful tool result: tool call in 1.7 s, correct natural-language answer summarising it in 3.1 s. Feeding back `{"error": "RATE_LIMITED"}`: 1.3 s, **no retry loop, no fabricated price** — it reported the rate limit and suggested an alternative.

That failure path is the notable one. Retrying blindly on error and inventing plausible output are the two most common failure modes in small local models; this one does neither.

**Why tool calling survives ternary quantization better than knowledge does.** Tool calling is mostly schema-constrained formatting — pick one tool from N, emit JSON with the right keys and types. That is a narrow, highly structured output distribution that 1.75 bpw preserves well. Recall of factual knowledge is a broad, high-entropy distribution and is what actually degrades. So the capability that matters most for agent use is the capability that degrades least.

### Thinking mode must be turned off

The model defaults to thinking mode. On the same "3 people 3 days" question:

| Mode | Tokens | Time | Answer |
|---|---|---|---|
| thinking on (default) | **2 192** | **60.9 s** | correct |
| thinking off | **332** | **9.4 s** | correct |

6.5x the time for identical output quality. Worse, with `max_tokens` below ~1 500 the response is consumed entirely by the reasoning trace and **`content` comes back empty** — the answer never arrives. A 900-token budget produced 2 575–2 983 characters of reasoning and zero answer on all five reasoning items.

Disable it per request:

```json
"chat_template_kwargs": {"enable_thinking": false}
```

or at startup: `--chat-template-kwargs '{"enable_thinking":false}'`.

With thinking off, latency is 0.4–3.7 s per agent step. That is what makes this model usable as a local agent brain rather than a demo.

### Reproducing

```bash
llama-server -m ~/models/bonsai/Ternary-Bonsai-2-9B-PTQ1_0.gguf \
  -c 8192 -ngl 99 -t 4 --host 127.0.0.1 --port 8099 \
  --no-warmup --chat-template-kwargs '{"enable_thinking":false}'

curl -s http://127.0.0.1:8099/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"三人三天喝三桶水，九人九天喝几桶？"}],"max_tokens":600,"temperature":0}'
```

### Server lifetime: the cgroup trap

`llama-server` needs **4.5 GB** resident once the 9B model is loaded. If it is started from an agent's background-job wrapper it inherits a worker cgroup capped around 3.65 GiB, and the kernel OOM-kills it the first time KV grows — with no message in the llama log, the process just disappears mid-task. Give it its own unit with an explicit ceiling:

```bash
systemd-run --user --unit=bonsai-server \
  -p MemoryHigh=5.8G -p MemoryMax=6.8G \
  ~/llama-prism/build/bin/llama-server -m <model> \
  -c 8192 -ngl 99 -t 4 --host 127.0.0.1 --port 8099 --no-warmup
# check: journalctl --user -u bonsai-server -n 30
```

## Reproducing

```bash
./scripts/bench-sweep.sh
```

Emits the same table. Default sweep takes 10-12 minutes, dominated by the 64K prefill.
