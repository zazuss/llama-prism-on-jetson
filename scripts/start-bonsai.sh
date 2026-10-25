#!/usr/bin/env bash
# start-bonsai.sh — 启动 Bonsai-2 9B 本地推理服务（单路 / 128K 上下文 / 实测 36 t/s）
#
# 实测配置（Jetson Orin 8GB，2026-10-25）：
#   128K 上下文 + q4_0 KV + 单 slot → 35.4-36.6 t/s，常驻 4.46 GB，余量 1.4 GB
#   256K 上下文 + q4_0 KV + 单 slot → 36.1 t/s，但常驻 5.7 GB，余量仅 39 MB（会被 OOM）
#   prefill 速度：64K 300.56 t/s（218s）→ 128K 214.24 t/s（612s，约 10 分钟出首 token）
#
# 三个必须遵守的坑：
#   1. 必须 -np 1。默认 -np 4 会让每个 slot 都想吃到满上下文，
#      速度从 36 t/s 掉到 16.5 t/s，且第 2 个请求就 OOM。
#   2. 必须用 systemd-run 托管，不能用后台 shell 启动。
#      后台启动会继承 Hermes worker 的 cgroup（memory.max ≈ 3.65 GiB），
#      模型加载到 4.5 GB 时内核会静默 OOM kill。
#   3. thinking 默认关闭。开启后同一道题要 2192 token / 60.9 s，
#      关闭只需 332 token / 9.4 s，答案质量无差别。
#
# 用法：
#   ./start-bonsai.sh          启动（已在跑则提示）
#   ./start-bonsai.sh stop     停止
#   ./start-bonsai.sh status   查看状态与内存
#   ./start-bonsai.sh restart  重启

set -euo pipefail

# ---- 可调配置 ----
_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${BONSAI_MODEL:-/home/orin/models/bonsai/Ternary-Bonsai-2-9B-PTQ1_0.gguf}"
if [ -n "${BONSAI_BIN:-}" ]; then
    BIN_DIR="$BONSAI_BIN"
elif [ -x "$_REPO/bin/llama-server" ]; then
    BIN_DIR="$_REPO/bin"                       # Release 包解压到仓库 bin/
elif [ -x "/home/orin/llama-prism/build/bin/llama-server" ]; then
    BIN_DIR="/home/orin/llama-prism/build/bin" # 本机源码编译产物
else
    BIN_DIR="$_REPO/bin"                       # 报错时会提示这个路径
fi
# 两种写法都认：BONSAI_CTX（推荐）或 CTX
CTX="${BONSAI_CTX:-${CTX:-131072}}"          # 128K。要 256K 改成 262144（余量只剩 39MB）
NP="${BONSAI_NP:-${NP:-1}}"                  # 单路。不要改大
KV="${BONSAI_KV:-${KV:-q4_0}}"               # q4_0 最省；q8_0 质量更好但多占 300MB
PORT="${BONSAI_PORT:-${PORT:-8099}}"
THREADS="${BONSAI_THREADS:-${THREADS:-6}}"
UNIT="bonsai-server"
MEM_HIGH="6.2G"
MEM_MAX="6.8G"
# ------------------

log() { printf '%s\n' "$*"; }
die() { printf '错误: %s\n' "$*" >&2; exit 1; }

is_running() { systemctl --user is-active --quiet "$UNIT" 2>/dev/null; }

cmd_stop() {
    if is_running; then
        log "停止 $UNIT ..."
        systemctl --user stop "$UNIT" 2>/dev/null || true
        sleep 2
        # 兜底：unit 已退但进程残留
        pgrep -x llama-server | while read -r p; do kill "$p" 2>/dev/null || true; done
        sleep 1
    fi
    systemctl --user reset-failed "$UNIT" 2>/dev/null || true
    log "已停止"
    free -m | awk '/^Mem:/{printf "  内存 available: %s MB\n", $7}'
}

cmd_status() {
    if is_running; then
        log "状态: 运行中 (端口 $PORT)"
        systemctl --user show "$UNIT" -p MemoryCurrent 2>/dev/null \
            | sed 's/^/  /' | awk -F= '{if($2!="") printf "  常驻: %.2f GB\n", $2/1073741824}'
        curl -s --max-time 3 "http://127.0.0.1:$PORT/props" 2>/dev/null | python3 -c '
import json,sys
try:
    d=json.load(sys.stdin)
    print("  n_ctx:", d.get("default_generation_settings",{}).get("n_ctx"))
    print("  slots:", d.get("total_slots"))
except Exception:
    print("  服务无响应")
' 2>/dev/null || log "  服务无响应"
    P=$(pgrep -x llama-server | head -1 || true)
    [ -n "${P:-}" ] && grep -E "VmRSS" "/proc/$P/status" 2>/dev/null | sed 's/^/  /'
    free -m | awk '/^Mem:/{printf "  内存 available: %s MB\n", $7}'
    log "  thinking: 已关闭"
    else
        log "状态: 未运行"
    fi
}

cmd_start() {
    [ -f "$MODEL" ] || die "找不到模型: $MODEL"
    [ -x "$BIN_DIR/llama-server" ] || die "找不到 llama-server: $BIN_DIR"

    if is_running; then
        log "已在运行，不重复启动（要重启用 restart）"
        cmd_status
        return 0
    fi

    # 启动前预留：实测 128K + q4_0 常驻约 4.5 GB，但其中含可回收缓存，
    # available 有 4000 MB 就能启动，启动后剩约 1.4 GB 给系统
    local avail
    avail=$(free -m | awk '/^Mem:/{print $7}')
    if [ "$avail" -lt 4000 ]; then
        die "可用内存只有 ${avail} MB，128K 配置至少需要 4000 MB。先关掉占内存的进程。"
    fi

    systemctl --user reset-failed "$UNIT" 2>/dev/null || true

    log "启动 Bonsai-2 9B"
    log "  模型:   $(basename "$MODEL")"
    log "  上下文: $CTX ($([ "$CTX" -ge 131072 ] && echo '128K 或以上' || echo '低于 128K'))"
    log "  KV:     $KV      slot: $NP 路     端口: $PORT"

    systemd-run --user --unit="$UNIT" \
        -p MemoryHigh="$MEM_HIGH" -p MemoryMax="$MEM_MAX" \
        -p Environment="LD_LIBRARY_PATH=$BIN_DIR" \
        "$BIN_DIR/llama-server" \
        -m "$MODEL" \
        -c "$CTX" -np "$NP" -ngl 99 -t "$THREADS" \
        -ctk "$KV" -ctv "$KV" -fa on \
        --host 127.0.0.1 --port "$PORT" --no-warmup \
        --chat-template-kwargs '{"enable_thinking":false}' >/dev/null 2>&1 \
        || die "systemd-run 启动失败"

    printf '  等待就绪'
    local i
    for i in $(seq 1 45); do
        sleep 2
        if curl -s --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then
            printf ' 就绪（%s 秒）\n' "$((i*2))"
            log ""
            cmd_status
            return 0
        fi
        printf '.'
    done
    printf '\n'
    die "90 秒内未就绪，查日志: journalctl --user -u $UNIT -n 40"
}

case "${1:-start}" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    restart) cmd_stop; sleep 1; cmd_start ;;
    status)  cmd_status ;;
    *)       die "用法: $0 {start|stop|restart|status}" ;;
esac
