# llama-prism on Jetson Orin

在 Jetson Orin（sm_87）上为 [PrismML 的 llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) 编译 **CUDA** 支持，用来跑 Ternary-Bonsai-2 9B 三值量化模型。完整的构建配方 + 实测数据。

> **English**: Build recipe and benchmarks for PrismML's llama.cpp fork with CUDA on Jetson Orin (sm_87), running Ternary-Bonsai-2 9B. PrismML ships prebuilt binaries for x86 and `ubuntu-arm64`, but the arm64 package is **CPU-only** — there is no CUDA build for Jetson anywhere. If you want `PTQ1_0` on an Orin GPU, you compile it yourself. This repo is that recipe, plus the measured numbers you can expect.

**这个仓库要解决的问题：** PrismML 提供了 x86 和 `ubuntu-arm64` 的预编译二进制，但那个 arm64 包**只有 CPU 版本**。Jetson 上能用的 CUDA 构建在任何地方都找不到——他们的 Release 里没有，`jetson-containers` 里没有，jetson-ai-lab 上也没有。想让 `PTQ1_0` 跑在 Orin 的 GPU 上，只能自己编。这个仓库就是那份配方，附带你可以预期的实测数字。

## 平台

| 项 | 值 |
|---|---|
| 设备 | Jetson Orin Nano 8GB（P3767） |
| L4T | R39.2.1 |
| Ubuntu | 24.04.4 LTS |
| CUDA | 13.2 |
| 计算能力 | **sm_87** |
| CPU 核数 | 6 |
| 可用内存 | **7.3 GiB**（present 7.68 GiB） |

## 结果摘要

Ternary-Bonsai-2 **9B**（`PTQ1_0`，1.85 GiB，8.95 B 参数，1.75 bpw）在 7.3 GiB 的 Orin 上：

| 上下文 | Prefill (t/s) | 生成 (t/s) |
|---|---|---|
| 128 | 360.75 ± 12.78 | 36.57 ± 0.26 |
| 2 048 | 376.12 ± 0.11 | 36.46 ± 0.25 |
| 8 192 | 367.23 ± 0.04 | 36.59 ± 0.24 |
| 16 384 | 354.64 | 36.24 |
| 32 768 | 334.09 | 36.20 |
| **65 536** | **300.56** | **36.10** |

**这个模型的上下文长度几乎是免费的。** 从 128 token 到 65 536 token——翻了 512 倍——生成速度只掉 1.3%（36.57 → 36.10 t/s）。prefill 掉得快一些（360 → 300 t/s，−17%），这只意味着塞满一整个 64K prompt 要花约 218 秒；之后每生成一个 token 的成本和 128 上下文时一样。

这不是稠密注意力模型会有的行为。原因在架构上：`qwen35` 用的是混合注意力（`full_attention_interval = 4`），只有一部分层建 KV cache。即使跑到 64K 上下文，稳态 RSS 也保持在**约 2.8 GB**——比 32K 时的 2.7 GB 只多一点。而朴素的 KV 算法（每一层都算 `2 × 层数 × kv_heads × key_len × 2 字节`）会把成本高估大约 4 倍。

## 它真能当 agent 用吗？

原始 tok/s 并不是关键问题。这个模型每权重只有 1.75 bit——真正的问题是推理能力和工具调用有没有被这个位宽毁掉。实测（关闭 thinking）：

- **推理：5/5。** 多步应用题、反直觉比例题、自指骑士/无赖谜题、代码生成、中文语病判断——全部正确。
- **工具调用：10/10。** 单个调用与并行调用、在两个相似工具间选对（`get_weather` 对 `get_forecast`）、嵌套数组参数、enum 取值、普通聊天和简单算术不乱调工具、以及面对没有目标路径的危险删除命令时正确拒绝。多轮续答和错误恢复（`RATE_LIMITED`）都表现正常——不重试、不编造。
- **延迟：每个 agent 步骤 0.4–3.7 秒。** 这才是它能本地实跑的底气。

**thinking 模式必须关掉。** 它默认开着，一道题的代价是 2 192 token / 60.9 秒；关掉后是 332 token / 9.4 秒，答案完全一样。`max_tokens` 低于约 1 500 时，推理过程会把预算吃光，`content` 返回空字符串。

完整表格、复现方法，以及那个会静默 OOM 杀掉 server 的 cgroup 陷阱：[RESULTS.md](RESULTS.md#agent-capability-reasoning-and-tool-calling)。

## 为什么是 9B 而不是 27B

27B 的 `PTQ1_0` 权重是 **5.54 GB（5 671 MB）**，而这块板的分块累积分配上限实测为 **5 376 MB**——光权重就差 **295 MB**。这还没算 KV cache、CUDA context 和计算 buffer。把桌面栈（gdm / gnome-shell / Xorg 及其附属，合计约 250 MB）全杀光也补不上这个缺口，而且一点余量不留必然被 OOM 杀掉。

顺带一个常被问的数字：即使能装下，27B 的速度按带宽外推也只有**约 12 t/s**（9B 是 36.5 t/s，慢 3 倍），因为它每生成一个 token 都要把全部权重从内存读一遍。对 agent 场景来说这个速度已经偏慢了。

结论：**9B 就是 7.3 GiB Orin 的天花板。** 要跑 27B 需要 Orin AGX 32/64 GB。

## 预编译二进制

不想自己编？Release 页面上有编好的 CUDA 二进制：

**→ [下载 `llama-prism-bin-jetson-orin-cuda13.2-sm87.tar.gz`](https://github.com/zazuss/llama-prism-on-jetson/releases/latest/download/llama-prism-bin-jetson-orin-cuda13.2-sm87.tar.gz)**（56.6 MB，135 个文件）

```bash
mkdir -p ~/llama-prism-bin
tar xzf llama-prism-bin-jetson-orin-cuda13.2-sm87.tar.gz -C ~/llama-prism-bin
export LD_LIBRARY_PATH=$HOME/llama-prism-bin:$LD_LIBRARY_PATH
~/llama-prism-bin/llama-cli --list-devices   # 预期 CUDA0: Orin
```

**环境必须严格对得上**，否则你会看到 `undefined symbol` 或 `GLIBCXX not found`：

| 组件 | 要求 |
|---|---|
| L4T / JetPack | **R39.2.1**（JetPack 7.x） |
| CUDA | **13.2**（链接 `libcudart.so.13`、`libcublas.so.13`） |
| 计算能力 | **sm_87** |

Orin Nano / NX / AGX 都能用（都是 sm_87）。L4T 或 CUDA 版本不同的话，请用下面的源码编译——三值 kernel 对 CUDA 版本敏感。

## 编译

前置条件：与你的 L4T 匹配的 CUDA toolkit、`cmake` ≥ 3.20、`g++`、约 10 GB 空闲磁盘，最好再准备 4 GB+ 的 swap。

```bash
# 1. 取打上 build tag 的源码。
#    在某些国内网络上直接从 GitHub clone 不可靠
#    （TLS 截断：GnuTLS recv error -110，pack 不完整）。
#    clone 失败就走镜像下 tarball：
curl -L -o llama-prism.tar.gz \
  https://ghfast.top/https://github.com/PrismML-Eng/llama.cpp/archive/refs/tags/prism-b10809-b3ebc47.tar.gz
tar xzf llama-prism.tar.gz && cd llama.cpp-prism-b10809-b3ebc47

# 2. 针对 Orin 配置。
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release

# 3. 编译。-j4 是带桌面会话的 6 核 Orin 上的安全上限；
#    -j5 以上会抢内存，曾经把别的进程 OOM 掉。
cmake --build build -j4
```

验证 CUDA 后端真的链上了——这一步是大家最爱跳过的：

```bash
export LD_LIBRARY_PATH=$PWD/build/bin:$LD_LIBRARY_PATH
./build/bin/llama-cli --list-devices
# 预期：CUDA0: Orin (7485 MiB, xxxx MiB free)

ldd ./build/bin/llama-cli | grep -E 'cudart|cublas'
# 预期：libcudart.so.13, libcublas.so.13
```

再确认三值 kernel 编出来了——只有把 `PTQ1_0` 支持真正编进去，这些测试二进制才会存在：

```bash
ls build/bin/test-ptq1_0-cuda-dot build/bin/test-pq2-row-shapes
```

## 模型

```bash
# 9B，PTQ1_0（最小，1.85 GiB）
huggingface-cli download prism-ml/Ternary-Bonsai-2-9B-gguf \
  Ternary-Bonsai-2-9B-PTQ1_0.gguf --local-dir ~/models/bonsai
```

格式的选择是有代价的：

| 格式 | 体积 | 原版 llama.cpp 能跑吗 | 说明 |
|---|---|---|---|
| `PTQ1_0`（id 143） | 1.85 GiB | **不能** | 真正的三值，内存最省——Orin 上用它 |
| `PQ2_0`（id 142） | 约 2.4 GB | 不能 | 模型卡里的默认档，反量化更快，更吃内存 |
| `Q2_0` | 约 2.6 GB | 能 | 兼容层，不是真正的三值 |

`PTQ1_0` 和 `PQ2_0` 必须用这个 fork。原版上游 llama.cpp——包括所有 `jetson-containers` 和 jetson-ai-lab 的构建——都读不了这两个格式。

**选档建议：** `PTQ1_0` 的 1.75 bpw 是整个家族里最小、精度最差的一档，当初选它只是为了省内存。实跑建议换 **`PQ2_0`**（约 2.4 GB），速度更快、质量更好，在 7.3 GiB 的板子上仍然放得下。家族里还有 2B / 4B / 27B，9B 和 27B 都另带视觉支持（`mmproj-BF16.gguf` / `mmproj-Q8_0.gguf`）。

## 运行

```bash
export LD_LIBRARY_PATH=$PWD/build/bin:$LD_LIBRARY_PATH

# 单轮推理。-st 是必需的：这个 fork 砍掉了 -no-cnv，
# 不加 -st 的话 llama-cli 会停在交互模式里干等 stdin，看起来像卡死。
./build/bin/llama-cli \
  -m ~/models/bonsai/Ternary-Bonsai-2-9B-PTQ1_0.gguf \
  -p "Explain ternary quantization in one sentence." \
  -n 64 -c 8192 -ngl 99 -t 6 -st --no-display-prompt --no-warmup
```

复现上面那张表：

```bash
for P in 128 2048 8192 16384 32768 65536; do
  ./build/bin/llama-bench \
    -m ~/models/bonsai/Ternary-Bonsai-2-9B-PTQ1_0.gguf \
    -p $P -n 64 -ngl 99 -t 6 -r 1
done
```

## 内存安全

在开了桌面会话、总共 7.3 GiB 的板子上，把推理放进它自己的 cgroup，这样失控的分配会杀掉那个 benchmark，而不是你的会话：

```bash
systemd-run --user --scope -p MemoryHigh=6.5G -p MemoryMax=7G \
  ./build/bin/llama-bench -m <model> -p 65536 -n 64 -ngl 99 -t 6 -r 1
```

64K prefill 会瞬间冲到约 6.9 GB，之后回落到 2.8 GB。没有 cgroup 的话，这个峰值期间的 OOM 会带走不相关的进程（在这块板上关笼子之前，它杀过一个 Hermes 网关）。

在 8 GB 上永远不要同时编译和跑推理。`cc1plus` + `nvcc` + 一个已加载的模型会超过 7.3 GiB，而内核的 OOM killer 并不管哪个进程重要。

## 坑

1. **不存在 CUDA 预编译包。** 别浪费时间翻 Release、`jetson-containers` 或 PyPI。那个 `bin-ubuntu-arm64` 资产只有 CPU 版本。
2. **这个 fork 里没有 `-no-cnv`。** 用 `-st` / `--single-turn`。
3. **不传 `-c` 会分配一个巨大的默认上下文**，prompt 还没开始跑就可能吃掉约 5 GB RSS。永远显式传 `-c`。
4. **这里 `CMA` 限制完全无关**，但老被拿来背锅：Orin 上单次 `cudaMalloc` 上限实测 **4 320–4 576 MB**（随当时空闲内存浮动），分块累积可达 **5 376 MB**（7 × 768 MB）。而大家记忆里那个 1 GB 天花板属于 CMA/NvMap 那条路（`CmaTotal` 256 MB），是 V4L2/ISP/显示用的。GGML 根本不碰 CMA。
5. **别在原版 llama.cpp 和这个 fork 之间混用 `.so`。** `ggml` 的 ABI 不一样，你会得到静默的错误输出或者直接崩。
6. **`LD_LIBRARY_PATH` 必须指向 `build/bin`**，否则二进制会加载系统的 `libggml` 然后因版本错配而失败。
7. **一个加载好的 9B server 需要常驻 4.5 GB。** 如果从 agent/shell 的后台任务包装器里启动，它可能继承一个上限约 3.65 GiB 的 cgroup，KV 一涨内核就把它 OOM 掉——`llama-server` 什么都不打印，进程在请求中途直接消失。要用它自己的 unit 加显式上限启动（`systemd-run --user --unit=... -p MemoryMax=6.8G`），而且它静默死掉时要去看 `journalctl --user -u <unit>`，不是看 llama 的日志。

## License

本仓库的脚本和文档：MIT。fork 和模型权重各自带自己的许可证——见 [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) 和模型卡。
