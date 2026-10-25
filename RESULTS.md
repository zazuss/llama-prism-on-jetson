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
