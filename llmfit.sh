#!/usr/bin/env bash
# llmfit.sh - find the best-fitting llama-server parameters for a given model
#             quant on a Vulkan or ROCm device (integrated or discrete).
#
# Phases:
#   0 probe    device + GGUF facts, memory bandwidth, decode/prefill ceilings,
#              and a predicted spec-decoding curve
#   1 raw      llama-bench pp512/tg128 -> measured prefill, decode, and the
#              achieved effective memory bandwidth
#   2 spec     spec-draft-n-max sweep on llama-server with a no-speculation
#              control, repeated in opposite order to cancel power/thermal drift
#   3 prefill  batch/ubatch/flash-attn variants (prefill is usually flat here)
#   4 ngram    n-gram speculation on a varied prompt vs an echo-heavy prompt,
#              to tell whether n-gram drafting pays off for this workload
#   5 report   ranked results, refitted prediction, ready-to-run command
#
# Every server run uses a FIXED SEED: acceptance length depends on the text
# being generated, so unfixed seeds make configurations incomparable.
#
# Usage: ./llmfit.sh --hf <repo>[:quant] [options]      (see --help)

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$(cd "$HERE/.." && pwd)"
LIB="$HERE/llmfit_lib.py"

# ------------------------------- defaults ---------------------------------
BINARY="${BINARY:-$LLAMA_DIR/llama-server}"
BENCH="${BENCH:-$LLAMA_DIR/llama-bench}"
HF=""
MODEL=""
DEV=""
# 128K by default; the probe clamps this to the model's trained maximum and
# reports the effective value, which the runs then use so the preset, the
# measurements and the memory budget all agree.
CTX="${CTX:-131072}"
CACHE_TYPE="${CACHE_TYPE:-q4_0}"
PEAK_TFLOPS="${PEAK_TFLOPS:-0}"
MEM_BUS_BITS="${MEM_BUS_BITS:-0}"
MEM_MT_S="${MEM_MT_S:-0}"
ROUNDS="${ROUNDS:-2}"
NMAX_LIST="${NMAX_LIST:-2 4 6 8}"
# KV cache type is a memory-vs-speed tradeoff, not a free win: a 4-bit cache is
# smallest but dequantises inside the attention loop, which can cost throughput
# once the sequence is long. It has to be measured, not assumed.
KV_TYPES="${KV_TYPES:-q4_0 q8_0 f16}"
# prefill is chunked by --ubatch-size, not --batch-size: sweeping -b alone
# changes nothing about prompt processing
UB_LIST="${UB_LIST:-512 2048}"
PMIN="${PMIN:-0.0}"          # spec-draft-p-min: keep drafting even at low confidence
MAX_TOKENS="${MAX_TOKENS:-128}"
BENCH_REPS="${BENCH_REPS:-5}"
# Depths for the raw benchmark. Depth 0 alone overstates real serving: decode and
# prefill both fall as the context grows, so a baseline should show the curve.
# e.g. BENCH_DEPTHS=0,8192
BENCH_DEPTHS="${BENCH_DEPTHS:-0}"
QUICK="${QUICK:-0}"
NO_NGRAM="${NO_NGRAM:-0}"
DOWNLOAD=0
OUT="${OUT:-}"
DRY_RUN=0
PORT="${PORT:-8420}"
SEED="${SEED:-42}"
EXTRA_ENV="${EXTRA_ENV:-}"
DRAFT_MODEL="${DRAFT_MODEL:-}"   # draft/assistant GGUF, or 'none'; auto-detected
SUMMARY_DIR=""
REPLAY_SESSION=""
REPLAY_NMAX="${REPLAY_NMAX:-4}"
REPLAY_TURNS="${REPLAY_TURNS:-0}"
REPLAY_MAX_CHARS="${REPLAY_MAX_CHARS:-400000}"
REPLAY_GEN="${REPLAY_GEN:-128}"
INI="${INI:-}"
WORKLOAD="${WORKLOAD:-varied}"
PROMPT_VARIED="${PROMPT_VARIED:-$HERE/prompts/varied.txt}"
PROMPT_REPET="${PROMPT_REPET:-$HERE/prompts/repetitive.txt}"
PROMPT_OVERRIDE=""

usage() {
  cat <<'EOF'
llmfit.sh - tune llama-server for a model quant on Vulkan/ROCm.

  --hf REPO[:QUANT]     HuggingFace repo, e.g. bartowski/Model-GGUF:Q4_K_M
                        (default quant Q4_K_M). Must be in the HF cache unless
                        --download is given.
  --model PATH          use a local GGUF instead of --hf
  --dev DEV             device id, e.g. Vulkan0 or ROCm0 (default: first
                        available). Errors out if the build lacks that backend.
  --ctx N               context size for probing and all runs (default 32768)
  --cache-type T        KV cache type (default q4_0; a quantized V cache
                        requires flash-attn)
  --peak-tflops F       device compute peak, for the prefill efficiency figure
  --mem-bus-bits N      discrete GPU memory bus width (bits). Not exposed by
                        sysfs, so this is needed for a theoretical bandwidth.
  --mem-mt-s N          memory data rate (MT/s), used with --mem-bus-bits
  --rounds N            drift-control rounds (default 2; opposite order)
  --nmax "2 4 6 8"      spec-draft-n-max candidates
  --max-tokens N        tokens generated per server run (default 128)
  --bench-reps N        llama-bench repetitions (default 5)
  --quick               phases 1+2 only (skip prefill and n-gram phases)
  --no-ngram            skip phase 4
  --download            fetch the model into the HF cache if missing
  --binary PATH         llama-server (default ../llama-server)
  --bench-binary PATH   llama-bench  (default ../llama-bench)
  --env "K=V ..."       extra environment for the server, e.g. RADV_PERFTEST=sam
  --replay FILE         replay a recorded OMP session (JSONL under
                        ~/.omp/agent/sessions/) against the tuned server instead
                        of sweeping synthetic prompts. One server serves every
                        turn on a single slot, so context grows as it did live and
                        prefill/decode/acceptance are measured at the depths the
                        real workload used. Skips the sweep. The model is read
                        from the recording, so --hf/--model is optional.
  --replay-turns N      replay only the first N turns (default: all)
  --replay-nmax N       spec-draft-n-max to use while replaying (default 4)
  --replay-gen N        tokens to generate per turn (default 128)
  --summary DIR         summarise an existing run directory and (re)generate its
                        preset INI. Runs nothing, needs no --hf/--model.
  --ini PATH            where to write the model preset
                        (default: <run dir>/llama-models-options.ini)
  --workload NAME       prompt marker treated as the primary workload
                        (default: varied)
  --out DIR             output directory (default ./llmfit_<timestamp>)
  --port N              server port (default 8420)
  --seed N              sampling seed for every request (default 42)
  --prompt FILE         override the varied prompt
  --dry-run             print the commands without running them
  -h, --help            this text
EOF
}

while (( $# )); do
  case "$1" in
    --hf)            HF="$2"; shift 2 ;;
    --model)         MODEL="$2"; shift 2 ;;
    --draft-model)   DRAFT_MODEL="$2"; shift 2 ;;
    --dev)           DEV="$2"; shift 2 ;;
    --ctx)           CTX="$2"; shift 2 ;;
    --cache-type)    CACHE_TYPE="$2"; shift 2 ;;
    --peak-tflops)   PEAK_TFLOPS="$2"; shift 2 ;;
    --mem-bus-bits)  MEM_BUS_BITS="$2"; shift 2 ;;
    --mem-mt-s)      MEM_MT_S="$2"; shift 2 ;;
    --rounds)        ROUNDS="$2"; shift 2 ;;
    --nmax)          NMAX_LIST="$2"; shift 2 ;;
    --max-tokens)    MAX_TOKENS="$2"; shift 2 ;;
    --bench-reps)    BENCH_REPS="$2"; shift 2 ;;
    --quick)         QUICK=1; shift ;;
    --no-ngram)      NO_NGRAM=1; shift ;;
    --download)      DOWNLOAD=1; shift ;;
    --binary)        BINARY="$2"; shift 2 ;;
    --bench-binary)  BENCH="$2"; shift 2 ;;
    --env)           EXTRA_ENV="$2"; shift 2 ;;
    --summary)       SUMMARY_DIR="$2"; shift 2 ;;
    --replay)        REPLAY_SESSION="$2"; shift 2 ;;
    --replay-turns)  REPLAY_TURNS="$2"; shift 2 ;;
    --replay-nmax)   REPLAY_NMAX="$2"; shift 2 ;;
    --replay-gen)    REPLAY_GEN="$2"; shift 2 ;;
    --ini)           INI="$2"; shift 2 ;;
    --workload)      WORKLOAD="$2"; shift 2 ;;
    --out)           OUT="$2"; shift 2 ;;
    --port)          PORT="$2"; shift 2 ;;
    --seed)          SEED="$2"; shift 2 ;;
    --prompt)        PROMPT_OVERRIDE="$2"; shift 2 ;;
    --dry-run)       DRY_RUN=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "unknown arg: $1 (try --help)" >&2; exit 2 ;;
  esac
done

command -v python3 >/dev/null || { echo "ERROR: python3 not found" >&2; exit 2; }

# --summary only re-reads an existing run, so it needs none of the run inputs.
if [[ -n "$REPLAY_SESSION" ]]; then
  [[ -f "$REPLAY_SESSION" ]] || { echo "ERROR: no such session file: $REPLAY_SESSION" >&2; exit 2; }
  # A recording names the model that served it, so --replay alone is enough:
  # read it back rather than making the caller repeat --hf/--model.
  if [[ -z "$MODEL" && -z "$HF" ]]; then
    _rtmp="$(mktemp -d)"
    python3 "$LIB" replay-extract --session "$REPLAY_SESSION" --out "$_rtmp" >/dev/null 2>&1
    # pick the first recorded model that actually resolves locally
    while IFS= read -r _rhint; do
      [[ -z "$_rhint" ]] && continue
      case "$_rhint" in
        hf:*)
          _r="${_rhint#hf:}"
          _slug="models--${_r%%:*}"
          _slug="${_slug//\//--}"
          if [[ -d "$HOME/.cache/huggingface/hub/$_slug" ]]; then HF="$_r"; break; fi
          ;;
        *)
          if [[ -f "$_rhint" ]]; then MODEL="$_rhint"; break; fi
          ;;
      esac
    done < "$_rtmp/model.txt" 2>/dev/null
    rm -rf "$_rtmp"
    [[ -n "$MODEL" || -n "$HF" ]] && echo "replay: using model from the recording: ${HF:-$MODEL}"
  fi
  [[ -n "$MODEL" || -n "$HF" ]] || {
    echo "ERROR: could not infer the model from the session; pass --hf or --model" >&2
    exit 2; }
fi
if [[ -z "$SUMMARY_DIR" && -z "$REPLAY_SESSION" ]]; then
  [[ -n "$MODEL" || -n "$HF" ]] || { echo "ERROR: pass --hf REPO[:QUANT] or --model PATH" >&2; exit 2; }
  [[ -x "$BINARY" ]] || { echo "ERROR: not executable: $BINARY" >&2; exit 2; }
  [[ -x "$BENCH" ]] || { echo "ERROR: not executable: $BENCH" >&2; exit 2; }
  [[ -f "$PROMPT_VARIED" ]] || { echo "ERROR: missing prompt: $PROMPT_VARIED" >&2; exit 2; }
  [[ -f "$PROMPT_REPET" ]] || { echo "ERROR: missing prompt: $PROMPT_REPET" >&2; exit 2; }
  [[ -n "$PROMPT_OVERRIDE" ]] && PROMPT_VARIED="$PROMPT_OVERRIDE"
  command -v curl >/dev/null || { echo "ERROR: curl not found" >&2; exit 2; }
fi

# ---------------------------- summary-only mode ---------------------------
# Summarise an existing run and (re)generate its model preset, without touching
# the GPU. Useful for re-reporting after changing the candidate set or for
# sharing results from another machine.
if [[ -n "$SUMMARY_DIR" ]]; then
  [[ -d "$SUMMARY_DIR" ]] || { echo "ERROR: no such run directory: $SUMMARY_DIR" >&2; exit 2; }
  [[ -f "$SUMMARY_DIR/summary.tsv" ]] || {
    echo "ERROR: $SUMMARY_DIR/summary.tsv not found (is that a llmfit run dir?)" >&2; exit 2; }
  [[ -n "$INI" ]] || INI="$SUMMARY_DIR/llama-models-options.ini"
  REPORT_ARGS=(--out "$SUMMARY_DIR" --ctx "$CTX" --workload "$WORKLOAD" --ini "$INI")
  [[ -n "$HF" ]] && REPORT_ARGS+=(--hf "$HF")
  python3 "$LIB" report "${REPORT_ARGS[@]}" || exit $?
  exit 0
fi

[[ -n "$OUT" ]] || OUT="$HERE/llmfit_$(date +%Y%m%d_%H%M%S)"
SUMMARY="$OUT/summary.tsv"
# A dry run must not leave a results directory behind, but the probe still needs
# somewhere to write probe.json for the later phases to read.
PROBE_OUT="$OUT"
if (( DRY_RUN )); then
  PROBE_OUT="$(mktemp -d)"
else
  mkdir -p "$OUT"
  : > "$SUMMARY"
fi

log() { echo "[$(date +%H:%M:%S)] $*"; }
banner() { echo; echo "=== $* ==="; }

CURRENT_PID=""
cleanup() {
  if [[ -n "${CURRENT_PID:-}" ]] && kill -0 "$CURRENT_PID" 2>/dev/null; then
    kill "$CURRENT_PID" 2>/dev/null; sleep 2; kill -9 "$CURRENT_PID" 2>/dev/null
  fi
  pkill -f "llama-server .*--port $PORT" 2>/dev/null
  echo; echo "interrupted; partial results in $OUT" >&2
  exit 130
}
trap cleanup INT TERM

# ------------------------------- helpers ----------------------------------
# GPU power/temperature/usage sampling.  On a UMA part the meaningful memory
# figure is the GPU-addressable window (GTT), not the small dedicated VRAM
# carveout that rocm-smi reports, so prefer sysfs and degrade gracefully.
gpu_sample() {
  local pw tz used total pct
  pw=$(cat /sys/class/drm/card*/device/hwmon/hwmon*/power1_average 2>/dev/null | head -1)
  tz=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)
  used=$(cat /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | head -1)
  total=$(cat /sys/class/drm/card*/device/mem_info_gtt_total 2>/dev/null | head -1)
  if [[ -z "$used" || -z "$total" || "$total" == "0" ]]; then
    used=$(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -1)
    total=$(cat /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null | head -1)
  fi
  pct=$(awk -v u="${used:-0}" -v t="${total:-0}" 'BEGIN{print (t>0)?100*u/t:0}')
  awk -v p="${pw:-0}" -v t="${tz:-0}" -v c="$pct" \
    'BEGIN{printf "%.1f %.1f %.1f\n", p/1e6, t/1000, c}'
}

write_request() {  # $1=prompt file, $2=out json
  python3 - "$1" "$2" "$MAX_TOKENS" "$SEED" <<'PY'
import json, sys
src, dst, maxtok, seed = sys.argv[1:5]
body = {
    "prompt": open(src, encoding="utf-8").read(),
    "max_tokens": int(maxtok),
    "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
    "presence_penalty": 0.0, "repeat_penalty": 1.0,
    "seed": int(seed),
    "stream": False, "stop": [], "ignore_eos": True,
}
open(dst, "w", encoding="utf-8").write(json.dumps(body, ensure_ascii=False))
PY
}

wait_ready() {  # -f makes a 503 "Loading model" count as NOT ready
  for _ in $(seq 1 900); do
    curl -fsS -o /dev/null "http://127.0.0.1:$PORT/health" 2>/dev/null && return 0
    [[ -n "${CURRENT_PID:-}" ]] && ! kill -0 "$CURRENT_PID" 2>/dev/null && return 1
    sleep 1
  done
  return 1
}

stop_server() {
  [[ -z "${CURRENT_PID:-}" ]] && return 0
  kill "$CURRENT_PID" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$CURRENT_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$CURRENT_PID" 2>/dev/null
  wait "$CURRENT_PID" 2>/dev/null
  CURRENT_PID=""
  sleep 2
}

# measure NAME CONFIG PROMPT_FILE -- <server args...>
measure() {
  local name="$1" config="$2" prompt="$3"; shift 3
  [[ "${1:-}" == "--" ]] && shift
  local log="$OUT/$name.log" resp="$OUT/$name.resp.json" pow="$OUT/$name.power"
  log "run $name :: $config"
  if (( DRY_RUN )); then
    echo "    $BINARY $*"
    echo "    prompt=$prompt max_tokens=$MAX_TOKENS seed=$SEED"
    return 0
  fi
  # Two servers on one GPU thrash the shared memory window and silently halve
  # throughput, so never start a run while another server holds the port.
  if pkill -f "llama-server .*--port $PORT" 2>/dev/null; then sleep 3; fi
  if [[ -n "$EXTRA_ENV" ]]; then
    # shellcheck disable=SC2086
    env $EXTRA_ENV "$BINARY" "$@" >"$log" 2>&1 &
  else
    "$BINARY" "$@" >"$log" 2>&1 &
  fi
  CURRENT_PID=$!
  if ! wait_ready; then
    log "$name: server failed to start (see $log); skipping"
    stop_server
    return 1
  fi
  : > "$pow"
  ( while :; do gpu_sample >> "$pow"; sleep 2; done ) & local powpid=$!
  local req="$OUT/$name.req.json"
  write_request "$prompt" "$req"
  curl -s --max-time 21600 -H 'Content-Type: application/json' \
    --data "@$req" "http://127.0.0.1:$PORT/completions" > "$resp"
  local rc=$?
  kill "$powpid" 2>/dev/null; wait "$powpid" 2>/dev/null
  stop_server
  if (( rc != 0 )); then
    log "$name: curl failed (rc=$rc)"
    return 1
  fi
  python3 "$LIB" parse --log "$log" --power "$pow" --name "$name" \
      --config "$config" --summary "$SUMMARY" || log "$name: parse failed"
}

# spec_args N_MAX [B] [UB] [FA] [KV] [SPEC_TYPE]
# Builds SERVER_ARGS in place; an empty N_MAX omits all speculation options.
spec_args() {
  local n="$1" b="${2:-1024}" ub="${3:-512}" fa="${4:-on}" kv="${5:-$CACHE_TYPE}" st="${6:-draft-mtp}"
  SERVER_ARGS=()
  if [[ -n "$MODEL" ]]; then SERVER_ARGS+=(-m "$MODEL"); else SERVER_ARGS+=(-hf "$HF"); fi
  SERVER_ARGS+=(-dev "$DEV" -lv 3 --metrics --port "$PORT" --host 127.0.0.1
                -b "$b" -ub "$ub" -c "$CTX" -fa "$fa" -ctk "$kv" -ctv "$kv"
                --parallel 1)
  if [[ -n "$n" ]]; then
    SERVER_ARGS+=(--spec-type "$st" --spec-draft-n-max "$n" --spec-draft-p-min "$PMIN"
                  --spec-draft-type-k "$kv" --spec-draft-type-v "$kv"
                  --spec-draft-device "$DEV")
    # draft-mtp drives either an inline MTP head or, when given here, a separate
    # companion draft model (draft-simple fails on such a pair in this build)
    [[ -n "$DRAFT_MODEL" ]] && SERVER_ARGS+=(--spec-draft-model "$DRAFT_MODEL")
  fi
}

# ------------------------------- phase 0 ----------------------------------
banner "phase 0: probe"
PROBE_ARGS=(--binary "$BINARY" --bench "$BENCH" --ctx "$CTX"
            --cache-type "$CACHE_TYPE" --out "$PROBE_OUT")
[[ -n "$MODEL" ]] && PROBE_ARGS+=(--model "$MODEL")
[[ -n "$DRAFT_MODEL" ]] && PROBE_ARGS+=(--draft-model "$DRAFT_MODEL")
[[ -n "$HF" ]] && PROBE_ARGS+=(--hf "$HF")
[[ -n "$DEV" ]] && PROBE_ARGS+=(--dev "$DEV")
# string compares, not (( )): the compute peak is a float
[[ "$MEM_BUS_BITS" != "0" ]] && PROBE_ARGS+=(--mem-bus-bits "$MEM_BUS_BITS")
[[ "$MEM_MT_S" != "0" ]] && PROBE_ARGS+=(--mem-mt-s "$MEM_MT_S")
[[ "$PEAK_TFLOPS" != "0" ]] && PROBE_ARGS+=(--peak-tflops "$PEAK_TFLOPS")
(( DOWNLOAD )) && PROBE_ARGS+=(--download)
python3 "$LIB" probe "${PROBE_ARGS[@]}" || exit $?

MODEL_PATH=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['model_path'])" "$PROBE_OUT/probe.json")
DEV=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['device']['id'])" "$PROBE_OUT/probe.json")
UMA=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('uma'))" "$PROBE_OUT/probe.json")
DRAFT_MODEL=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('draft_model',''))" "$PROBE_OUT/probe.json")
CTX_EFF=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('ctx_effective',0) or 0)" "$PROBE_OUT/probe.json")
if [[ -n "$CTX_EFF" && "$CTX_EFF" != "0" && "$CTX_EFF" != "$CTX" ]]; then
  log "context $CTX clamped to $CTX_EFF (model maximum)"
  CTX="$CTX_EFF"
fi

# An integrated part is power-limited by the platform profile, which is a
# bigger lever than any llama.cpp flag.
if [[ "$UMA" == "1" && -r /sys/firmware/acpi/platform_profile ]]; then
  prof=$(cat /sys/firmware/acpi/platform_profile)
  echo "  platform_profile : $prof"
  echo "  NOTE: on an APU the platform profile sets a power cap shared with the"
  echo "        CPU. Measured here with opposite-order repeats (balanced,"
  echo "        performance, performance, balanced): performance gave +48% prefill"
  echo "        and +43% decode, and balanced was also far noisier run to run."
  echo "        A sequential before/after test suggested the OPPOSITE, so use"
  echo "        interleaved repeats, not a single pair:"
  echo "          echo performance | sudo tee /sys/firmware/acpi/platform_profile"
fi
echo "  model path       : $MODEL_PATH"
echo "  device           : $DEV"

# ---------------------------- replay mode ---------------------------------
# Replay a recorded OMP session against the tuned server instead of sweeping
# synthetic prompts. One server serves every turn on a single slot, so the
# shared prefix is reused and the context grows exactly as it did live; prefill,
# decode and acceptance are then measured at the depths actually used.
if [[ -n "$REPLAY_SESSION" ]]; then
  banner "replay: $(basename "$REPLAY_SESSION")"
  if (( DRY_RUN )); then
    # extract into the throwaway dir so the dry run still reports the turn count
    # without leaving a results directory behind
    python3 "$LIB" replay-extract --session "$REPLAY_SESSION" \
        --out "$PROBE_OUT" --max-chars "$REPLAY_MAX_CHARS" >/dev/null 2>&1 || true
    spec_args "$REPLAY_NMAX"
    echo "    ${BINARY} ${SERVER_ARGS[*]}"
    n=$(wc -l < "$PROBE_OUT/turns.jsonl" 2>/dev/null || echo '?')
    echo "    would replay $([[ "$REPLAY_TURNS" == 0 ]] && echo "$n" || echo "$REPLAY_TURNS") turns from the session"
    rm -rf "$PROBE_OUT"
    exit 0
  fi
  python3 "$LIB" replay-extract --session "$REPLAY_SESSION" \
      --out "$OUT" --max-chars "$REPLAY_MAX_CHARS" --show-usage || exit 1
  [[ -f "$OUT/turns.jsonl" ]] || { echo "ERROR: no turns extracted" >&2; exit 1; }

  RLOG="$OUT/replay.log"
  spec_args "$REPLAY_NMAX"
  log "server: ${SERVER_ARGS[*]}"
  if pkill -f "llama-server .*--port $PORT" 2>/dev/null; then sleep 3; fi
  "${BINARY}" "${SERVER_ARGS[@]}" > "$RLOG" 2>&1 &
  CURRENT_PID=$!
  if ! wait_ready; then log "server failed to start (see $RLOG)"; stop_server; exit 1; fi

  python3 "$LIB" replay-run --turns "$OUT/turns.jsonl" --port "$PORT" \
      --gen "$REPLAY_GEN" --limit "$REPLAY_TURNS" --seed "$SEED" \
    || log "WARNING: some turns failed"
  stop_server
  python3 "$LIB" replay-report --log "$RLOG" --max-turns "$REPLAY_TURNS"
  echo
  echo "artifacts: $OUT"
  echo "  turns.jsonl  replay.log  prompt.txt"
  exit 0
fi

# ------------------------------- phase 1 ----------------------------------
banner "phase 1: raw benchmark (llama-bench)"
RAW_CMD=("$BENCH" -m "$MODEL_PATH" -dev "$DEV" -ngl 99 -fa on
         -ctk "$CACHE_TYPE" -ctv "$CACHE_TYPE" -b 2048 -ub 512
         -p 512 -n 128 -r "$BENCH_REPS" -d "$BENCH_DEPTHS")
if (( DRY_RUN )); then
  echo "    ${RAW_CMD[*]}"
else
  log "running llama-bench (${BENCH_REPS} reps)"
  "${RAW_CMD[@]}" > "$OUT/raw_bench.log" 2>&1
  python3 "$LIB" raw --bench-log "$OUT/raw_bench.log" --summary "$SUMMARY" \
      --prefix raw_ --config "dev=$DEV fa=on kv=$CACHE_TYPE" \
    || log "WARNING: could not parse llama-bench output"
fi

# ------------------------------- phase 2 ----------------------------------
banner "phase 2: parameter sweep ($ROUNDS rounds, opposite order)"

# One candidate list across every axis, so a KV-cache or ubatch choice competes
# for "best" under the same drift-control protocol as the n-max sweep.
#   nospec        - control, no speculation
#   nmax:N        - spec-draft-n-max N
#   kv:T          - KV cache type T
#   ub:N          - ubatch size N
declare -a CONFIGS=("nospec:")
for n in $NMAX_LIST;  do CONFIGS+=("nmax:$n"); done
# a KV type or ubatch equal to the base is already covered by the n-max sweep,
# which runs at those base values
for kv in $KV_TYPES;  do [[ "$kv" == "$CACHE_TYPE" ]] || CONFIGS+=("kv:$kv"); done
for u in $UB_LIST;    do [[ "$u" == "512" ]] || CONFIGS+=("ub:$u"); done

run_candidate() {  # $1=type $2=value $3=round -> measure one candidate
  local type="$1" value="$2" rnd="$3" name cfg
  case "$type" in
    nospec)
      spec_args ""
      name="c_nospec_r$rnd"; cfg="nospec prompt=varied" ;;
    nmax)
      spec_args "$value"
      name="c_n${value}_r$rnd"; cfg="n-max=$value p-min=$PMIN prompt=varied" ;;
    kv)
      spec_args "$BEST_GUESS_N" 1024 512 on "$value"
      name="c_kv${value}_r$rnd"; cfg="n-max=$BEST_GUESS_N p-min=$PMIN kv=$value prompt=varied" ;;
    ub)
      spec_args "$BEST_GUESS_N" 1024 "$value"
      name="c_ub${value}_r$rnd"; cfg="n-max=$BEST_GUESS_N p-min=$PMIN ub=$value prompt=varied" ;;
    *) return 0 ;;
  esac
  measure "$name" "$cfg" "$PROMPT_VARIED" -- "${SERVER_ARGS[@]}"
}

# secondary axes need a plausible n-max before the sweep has picked one; use the
# middle of the candidate range, and re-run the winner later anyway
BEST_GUESS_N="$(for n in $NMAX_LIST; do echo "$n"; done | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')"

for (( r=1; r<=ROUNDS; r++ )); do
  log "round $r/$ROUNDS"
  if (( r % 2 == 1 )); then order=$(seq 0 $(( ${#CONFIGS[@]} - 1 )))
  else order=$(seq $(( ${#CONFIGS[@]} - 1 )) -1 0); fi
  for i in $order; do
    IFS=':' read -r ctype cval <<<"${CONFIGS[$i]}"
    run_candidate "$ctype" "$cval" "$r"
  done
done

best_nmax() {
  python3 - "$SUMMARY" <<'PY'
import re, sys
acc = {}
for ln in open(sys.argv[1]).read().splitlines()[1:]:
    p = ln.split("\t")
    if len(p) < 6:
        continue
    m = re.match(r"c_n(\d+)_r\d+$", p[0])
    if not m:
        continue
    try:
        acc.setdefault(int(m.group(1)), []).append(float(p[5]))
    except ValueError:
        pass
print(max(acc, key=lambda k: sum(acc[k]) / len(acc[k])) if acc else 6)
PY
}
BEST_N="$(best_nmax 2>/dev/null || echo 6)"
log "best n_max from phase 2: $BEST_N"

# ------------------------------- phase 3 ----------------------------------
# KV type and ubatch are already candidates in phase 2; what is left here is
# flash attention, which cannot be swept there because a quantized V cache
# requires it. Compare it only against an f16 cache, which works either way.
if (( QUICK )); then
  banner "phase 3: skipped (--quick)"
else
  banner "phase 3: flash attention (f16 KV, $ROUNDS rounds, opposite order)"
  declare -a FA_CFGS=("off" "on")
  for (( fr=1; fr<=ROUNDS; fr++ )); do
    if (( fr % 2 == 1 )); then fa_order=$(seq 0 $(( ${#FA_CFGS[@]} - 1 )));     else fa_order=$(seq $(( ${#FA_CFGS[@]} - 1 )) -1 0); fi
    for fi in $fa_order; do
      fa="${FA_CFGS[$fi]}"
      spec_args "$BEST_N" 1024 512 "$fa" f16
      measure "fa_${fa}_r${fr}" "n-max=$BEST_N fa=$fa kv=f16 prompt=varied" \
        "$PROMPT_VARIED" -- "${SERVER_ARGS[@]}"
    done
  done
fi

# ------------------------------- phase 4 ----------------------------------
if (( QUICK || NO_NGRAM )); then
  banner "phase 4: skipped"
else
  banner "phase 4: n-gram speculation (varied vs echo-heavy prompt)"
  spec_args "$BEST_N" 1024 512 on "$CACHE_TYPE" "draft-mtp,ngram-mod"
  measure "ng_varied_ngram" "n-max=$BEST_N +ngram-mod prompt=varied" \
    "$PROMPT_VARIED" -- "${SERVER_ARGS[@]}"
  spec_args "$BEST_N"
  measure "ng_repet_base" "n-max=$BEST_N prompt=repetitive" \
    "$PROMPT_REPET" -- "${SERVER_ARGS[@]}"
  spec_args "$BEST_N" 1024 512 on "$CACHE_TYPE" "draft-mtp,ngram-mod"
  measure "ng_repet_ngram" "n-max=$BEST_N +ngram-mod prompt=repetitive" \
    "$PROMPT_REPET" -- "${SERVER_ARGS[@]}"
fi

# ------------------------------- phase 5 ----------------------------------
banner "phase 5: report"
if (( DRY_RUN )); then
  echo "    (dry run: nothing to report)"
else
  [[ -n "$INI" ]] || INI="$OUT/llama-models-options.ini"
  REPORT_ARGS=(--out "$OUT" --ctx "$CTX" --workload "$WORKLOAD" --ini "$INI")
  [[ -n "$HF" ]] && REPORT_ARGS+=(--hf "$HF")
  python3 "$LIB" report "${REPORT_ARGS[@]}"
fi

if (( DRY_RUN )); then
  rm -rf "$PROBE_OUT"
else
  echo "artifacts: $OUT"
  echo "  probe.json  summary.tsv  llama-models-options.ini"
  echo "  *.log  *.resp.json  *.power"
  echo
  echo "re-run the summary at any time with:"
  echo "  $0 --summary $OUT"
fi
