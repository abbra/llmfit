# llmfit — parameter fitting for llama-server on Vulkan / ROCm

Finds the best-fitting `llama-server` parameters for a given model quant, on an
integrated or discrete GPU, and explains *why* those are the best available by
computing the memory-bandwidth and compute ceilings first.

It answers three questions in one run:

1. **What is this model/hardware combination capable of?** — bandwidth and
   compute ceilings derived from the GGUF and the device.
2. **What are the best parameters?** — a drift-controlled sweep of
   speculative-decoding and batching parameters.
3. **Is the workload even able to benefit?** — e.g. n-gram speculation pays off
   only when output repeats the context, which is measured, not assumed.

## Requirements

- `llama-server` and `llama-bench` from the same build (defaults to the parent
  directory of this one)
- `python3`, `curl`
- Optional, improves the report: passwordless `sudo dmidecode` (integrated
  memory bandwidth), `rocm-smi` (discrete clocks)

Both shell and Python are stdlib-only. Nothing outside this directory is used.

## Quick start

```sh
# cached HuggingFace repo
./llmfit.sh --hf bartowski/Altworld_Hemmingway-1-GGUF:Q4_K_M

# local file, explicit device and compute peak
./llmfit.sh --model /path/to/model.gguf --dev Vulkan0 --peak-tflops 11.9

# fetch into the HF cache first if missing
./llmfit.sh --hf org/Repo-GGUF:Q5_K_M --download

# fast pass: raw benchmark + spec sweep only
./llmfit.sh --hf org/Repo-GGUF:Q4_K_M --quick

# see every command without running anything
./llmfit.sh --hf org/Repo-GGUF:Q4_K_M --dry-run
```

Results land in `./llmfit_<timestamp>/` unless `--out` is given. The final
report prints ranked configurations and a ready-to-run command. Re-print it at
any time with:

```sh
python3 llmfit_lib.py report --out llmfit_20260101_120000 --hf org/Repo-GGUF:Q4_K_M
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `--hf REPO[:QUANT]` | — | HuggingFace repo; quant defaults to `Q4_K_M` |
| `--model PATH` | — | local GGUF instead of `--hf` |
| `--dev DEV` | first available | `Vulkan0`, `ROCm0`, …; errors if the build lacks that backend |
| `--ctx N` | `131072` (128K) | context for probing and every run; clamped to the model's trained maximum, and the effective value is what the runs use |
| `--cache-type T` | `q4_0` | KV cache type; a quantized **V** cache requires flash-attn |
| `--peak-tflops F` | `0` | device compute peak, for the prefill efficiency figure |
| `--mem-bus-bits N` | `0` | discrete GPU memory bus width; not in sysfs, needed for a theoretical bandwidth |
| `--mem-mt-s N` | `0` | memory data rate (MT/s), used with `--mem-bus-bits` |
| `--rounds N` | `2` | drift-control rounds, run in opposite order |
| `--nmax "2 4 6 8"` | `2 4 6 8` | `--spec-draft-n-max` candidates |
| `KV_TYPES` | `q4_0 q8_0 f16` | KV cache types to compare (env only) |
| `UB_LIST` | `512 2048` | ubatch sizes to compare (env only) |
| `PMIN` | `0.0` | `--spec-draft-p-min` (env only) |
| `--max-tokens N` | `128` | tokens generated per server run |
| `--bench-reps N` | `5` | `llama-bench` repetitions |
| `--quick` | off | phases 0–2 only |
| `--no-ngram` | off | skip the n-gram phase |
| `--download` | off | fetch the model into the HF cache if missing |
| `--env "K=V …"` | — | extra environment for the server, e.g. `RADV_PERFTEST=sam` |
| `--binary PATH` | `../llama-server` | server binary |
| `--bench-binary PATH` | `../llama-bench` | benchmark binary |
| `--summary DIR` | — | summarise an existing run and (re)generate its preset; runs nothing |
| `--ini PATH` | `<run dir>/llama-models-options.ini` | where to write the model preset |
| `--workload NAME` | `varied` | prompt marker ranked as the primary workload |
| `--out DIR` | auto | output directory |
| `--port N` | `8420` | server port |
| `--seed N` | `42` | sampling seed for every request |
| `--prompt FILE` | — | override the varied workload prompt |
| `--dry-run` | off | print commands, run nothing, leave no directory |

Every option also works as an environment variable (`CTX=8192` etc.).

## Phases

| Phase | What it does |
|---|---|
| 0 probe | device + GGUF introspection, bandwidth detection, ceilings, predicted spec curve |
| 1 raw | `llama-bench` pp512 / tg128 — measured prefill, decode, achieved bandwidth |
| 2 sweep | one candidate list over `--spec-draft-n-max`, KV cache type and ubatch, plus a no-speculation control |
| 3 flash-attn | `-fa on` vs `off`, compared on an f16 cache (a quantized V cache requires FA) |
| 4 ngram | n-gram speculation on a varied prompt vs an echo-heavy prompt |
| 5 report | ranked configurations, refitted prediction, ready-to-run command |

A full run is about 15 server starts (10 in phase 2 with the default four
candidates and two rounds, plus 2 prefill variants and 3 n-gram runs); expect
10–20 minutes depending on model size and load time. `--quick` drops phases 3
and 4.

## How the numbers are derived

**Memory bandwidth.** Determined from hardware where the OS exposes it, then
cross-checked against measurement:

- *Integrated / UMA*: `dmidecode` DIMM count x 64-bit x data rate, e.g.
  `2 x DDR5 @ 5600 MT/s -> 128-bit = 89.6 GB/s`.
- *Discrete AMD*: `pp_dpm_mclk` x `--mem-bus-bits` (Linux does not expose the
  memory bus width, so this must be supplied for a theoretical figure).
- *Always*: **achieved** bandwidth = `model bytes x decode t/s`, taken from the
  raw benchmark. This is the number to trust; the report prints both and the
  percentage of peak.

**Decode ceiling** (batch-1 decode is bandwidth-bound):

```
bytes/token = active_weight_bytes + kv_bytes_per_token x ctx
decode_ceiling = bandwidth / bytes/token
```

`active_weight_bytes` accounts for MoE, reading only the routed experts.
`kv_bytes_per_token` counts only the KV-bearing layers (see *Structure* below),
not the block count — on a hybrid SSM stack those differ severalfold.

**Prefill ceiling** (prefill is compute-bound):

```
flops/token  = 2 x active_params
prefill_ceiling = peak_tflops / flops_per_token      (needs --peak-tflops)
```

The ratio `flops/token / bytes/token` is reported as *model intensity*; comparing
it against the device's FLOP-per-byte tells you which side you are on.

**Speculative prediction.** Per pass the target reads its weights once and
drafts `n` tokens; each drafted token additionally reads the output head plus
the MTP block. With per-position acceptance `a`, expected accepted length is
`sum(a^i, i=0..n)`, so

```
tps(n) = sum(a^i) / (t_base + n x t_draft)
```

which is maximised at some finite `n` — the optimum this harness is looking for.

The per-position rate `a` is not the "draft acceptance" ratio an engine logs
(that is accepted/generated). It is recovered by fitting `sum(a^i) = mean len`
from a measured configuration, then the whole curve is predicted and printed
next to the measurements.

> **Caveat the report states itself:** the model assumes draft cost grows
> linearly with `n`. On some backends it grows faster past a threshold (the
> 890M iGPU shows a cliff at `n=8`). Where prediction and measurement disagree,
> trust the measured column.

## Context default

The context defaults to **128K (131072)** and is clamped to the model's trained
maximum, read from the GGUF rather than assumed: gemma-4-E4B is exactly 128K, so
nothing is clamped, while a 32K model would be reduced with a note. The
*effective* value is what the probe, every run, the memory budget and the
generated preset all use, so a preset can never disagree with the numbers beside
it. The preset also records the context its measurements used.

If weights plus KV exceed the free device memory, the probe says so explicitly
rather than letting the engine fail or silently reduce later.

## The swept axes, and two common mistakes

Phase 2 sweeps every axis as *one* candidate list, so a KV-cache or ubatch
choice competes for "best" under the same drift-control protocol as the
speculative settings rather than being assumed:

| Axis | Candidates | Why it is not a free win |
|---|---|---|
| `--spec-draft-n-max` | 2 4 6 8 | acceptance saturates while draft cost keeps growing; the optimum is finite and hardware-specific |
| KV cache type | `q4_0` `q8_0` `f16` | smallest cache is not fastest: a 4-bit cache dequantises inside the attention loop |
| `--ubatch-size` | 512 2048 | prefill chunking; large values can also hurt, as measured here |
| `--flash-attn` | on / off (phase 3) | required by a quantized V cache, so it is compared on an f16 cache |

Two claims that circulate widely and are worth stating precisely:

- **`-b` does not control prompt-processing speed — `-ub` does.** In llama.cpp
  the logical batch (`-b`) is subdivided into physical ubatches (`-ub`) and it
  is the ubatch that determines the prefill GEMM size. Raising `-b` alone
  changes prefill throughput by nothing. This harness therefore sweeps `-ub`.
- **KV cache quantization needs flash attention**, and `-ctv` (the V cache) is
  the side that requires it. A quantized K with an f16 V is fine.

### What the KV sweep actually found here

Widely repeated advice is that `q8_0` KV is a free win and `q4_0` is
catastrophically slow at long context. Measured on the 890M iGPU at depth 0 and
8192, with the *same* f16 configuration run both first and last as a drift
bracket:

| depth | run order | kv | pp64 | tg128 |
|---|---|---|---|---|
| 0 | 1st | f16 | 88.7 | **3.88** |
| 0 | 2nd | q8_0 | 82.8 | 2.74 |
| 0 | 3rd | q4_0 | 72.9 | 2.63 |
| 0 | 4th | **f16** | 82.9 | **2.76** |
| 8192 | 1st | f16 | 77.2 | 3.52 |
| 8192 | 2nd | q8_0 | 77.6 | 3.62 |
| 8192 | 3rd | q4_0 | 72.2 | 3.89 |
| 8192 | 4th | **f16** | 73.0 | **3.66** |

Run order alone moved the identical configuration by **-28.9%** (d0). Against
the bracketed f16 reference the cache types differ by -0.7% / -4.7% (d0) and
-1.1% / +6.3% (d8192) -- inside the +/-0.7 absolute noise. A naive
f16 -> q8_0 -> q4_0 sequence would have "confirmed" the folklore from drift
alone.

**The takeaway: choose KV type for memory, not for speed.** The memory effect
is large and exact -- for this model at 32K context, 576 MiB (q4_0),
1.13 GiB (q8_0), 2.00 GiB (f16). Any speed effect is below what this machine
can resolve at these depths; deeper contexts or other architectures may differ,
which is why it is a swept axis rather than a hardcoded default.

One caveat when reading KV rows in a sweep: a quantized cache changes the
numerics, so the sampled tokens differ and the *acceptance length* changes with
them. Part of any throughput difference is therefore different work, not a
faster kernel. The report prints `acc_len` next to `tps` so the two can be told
apart.

## Structure, context cost and offload guidance

Phase 0 reads the GGUF tensor table itself, so the report can say what the
stack actually is rather than assuming a uniform transformer:

```
structure
---------
  kind            : hybrid  (16 full-attention, 48 recurrent/SSM)
  blocks          : 65
  KV-bearing      : 16 of 65 layers   -> 18.0 KiB/token
  layer pattern   : RRRARRRARRRARRR...RRRAM   (A=KV layer, R=recurrent)
  quant mix       : F32x456, Q4_Kx257, Q6_Kx79, Q8_0x64, Q4_0x8, Q5_Kx2
  bytes by component:
    ffn (dense)        9.594 GiB   59.1%
    attention          3.639 GiB   22.4%
    ssm (recurrent)    1.334 GiB    8.2%
    output head        0.971 GiB    6.0%   <- read only when speculation is on
    embeddings         0.666 GiB    4.1%
    mtp head           0.028 GiB    0.2%
    norms              0.003 GiB    0.0%
```

Layers are grouped by the *set of tensor roles* they contain, which is what
makes a hybrid stack visible: the model above is 48 Mamba/SSM layers (`ssm_a`,
`ssm_conv1d`, `ssm_dt`, `ssm_out`) interleaved with 16 full-attention layers
every 4th block, plus a trailing MTP block.

### Context has two different costs, and only one of them grows

`KV-bearing` counts the layers that hold a *growing* KV cache, using
`attention.recurrent_layers`, or `attention.shared_kv_layers` on models that
share KV across layers, falling back to the block count. The MTP block is
excluded: it is driven by its own draft context, so it takes no slice of the
target KV cache.

This matters more than it looks. Counting all 65 layers gives 73 KiB/token;
the correct answer is 18 KiB/token — a **4× overestimate** that would corrupt
any `-c` sizing decision. The figure is verified against llama.cpp itself:

```
$ llama-server ... -lv 4
llama_kv_cache: size = 576.00 MiB ( 32768 cells, 16 layers, 4/1 seqs)
```

and llmfit computes `16 layers x 4 heads x 512 x 0.5625 B x 32768 = 576.00 MiB`
— exact.

The recurrent layers instead keep a fixed-size state, so they contribute a
context-*independent* buffer (llama.cpp reports 598.50 MiB for this model at
any context). Raising `--ctx` is nearly free on such a stack; the report says
so and points at `-lv 4` for the exact figure rather than inventing one.

### Would keeping some layers off the GPU help?

The `bytes by component` table is the evidence, and the answer depends on the
model type:

- **Dense and hybrid: no.** Every weight is read on every decode step, so
  moving a layer to the CPU removes no GPU traffic — it only routes that
  layer's share through a slower path. Partial offload is a *fitting* tool, not
  a speed tool. This is the common misconception the section exists to answer.
- **Recurrent layers are not "unused layers."** They are cheaper in *context*
  (no KV), not in *weights* — those are still read every token.
- **MoE: yes, partially.** Experts dominate the file but only the routed
  fraction is read per token, so `--cpu-moe` / `-ncmoe N` removes GPU-resident
  weight at a cost proportional to the routed fraction, not the total. The
  report quantifies it, e.g. for a 2-of-8 expert model:

  ```
  MoE: expert weights are 22.00 GiB of 26.00 GiB (85%), but only
  2/8 experts are routed per token, so per-token expert traffic is 5.50 GiB.
  Keeping experts in system RAM (--cpu-moe, or -ncmoe N ...) removes weight
  from the GPU at a cost proportional to the ROUTED fraction (0.25).
  ```

- **Genuinely skippable bytes** are narrow: the MTP head when speculation is
  off, and unrouted experts in an MoE. Everything else is read every token.

> The MoE arithmetic was exercised against a synthetic tensor table matching a
> 2-of-8 expert layout, because no MoE model is present in the local cache. The
> dense, hybrid and shared-KV paths are all verified against real GGUFs and
> against llama.cpp's own buffer accounting.

### Two structural traps found on gemma-4-E4B

Both were caught by checking the harness's arithmetic against llama.cpp's own
buffer report rather than trusting the model.

**Interleaved sliding-window attention (iSWA).** gemma-4 has 42 blocks, 24 of
which hold KV -- but only **4 are full-attention**; the other 20 are *windowed*
(window 512, and different head dims: 512 vs 256). Treating all 24 as
full-context overstates KV **5x** (0.84 GiB vs the real 0.15 GiB). The split is
read from tensor shapes: a layer's `attn_k` output width is
`n_head_kv x key_length` for full attention and `n_head_kv x key_length_swa`
for a windowed layer, and the absence of `attn_k` means the layer holds no KV
at all. Shapes are preferred over metadata heuristics because they are exact --
llama.cpp reports "4 layers" and "20 layers" for the two caches, which is
precisely what the tensor widths say.

Allocation is also not the same as traffic: the windowed cache is sized at
`window x (parallel + 1)` cells, larger than one window. With that correction
the estimate matches llama.cpp exactly:

| model | computed | llama.cpp reports |
|---|---|---|
| gemma-4-E4B | 155.25 MiB | 144.00 + 11.25 = 155.25 MiB |
| Hemmingway-1 | 576.00 MiB | 576.00 MiB |

**A tied output head.** gemma-4 has no `output.weight`; the LM head *is* the
embedding matrix (680 MiB). Counting it as zero would badly understate the cost
of speculative decoding, which reads the head once per drafted token.

**A companion draft model.** Some models embed an MTP head (nextn tensors);
others ship it as a separate small GGUF. gemma-4 is the second kind
(`gemma4-assistant`, 4 blocks) and needs `--spec-draft-model`, which the harness
auto-detects. Which `--spec-type` works is not guessable: on this pair
`draft-simple` fails outright with `failed to decode draft batch`, while
`draft-mtp` on the same file gives 1.91x decode. Found by testing, not by
reading the docs.

## Flash attention: cheaper passes, lower acceptance

Measured on gemma-4 with `n-max=2`, f16 KV, harness sampling (temp 1.0, top_p
0.95, top_k 20, seed 42), interleaved off/on/on/off:

| | decode t/s | mean accepted len | ms/pass |
|---|---|---|---|
| fa=off | **33.66** (33.94, 33.37) | **2.07** | 61.5 |
| fa=on | 32.23 (32.09, 32.37) | **1.69** | 52.4 |

FA makes a pass **cheaper** (52.4 vs 61.5 ms) but changes the numerics enough
that acceptance falls further (1.69 vs 2.07), so speculation nets out slightly
behind: 2.07/61.5 = 33.7 t/s versus 1.69/52.4 = 32.2 t/s. Reproduced identically
in three separate sessions.

**Do not switch to `-fa off` on the strength of this.** A fixed seed makes each
configuration emit one specific text, and acceptance depends on how predictable
that text is; a 22% acceptance gap on one prompt and one seed can be a property
of the sample rather than of the flag. FA also gates V-cache quantization, so
turning it off costs the memory win in the section below.

### Acceptance cannot be compared across different sampling settings

While checking this, an ad-hoc test using `temperature 0.0` reported the
*opposite* sign for both throughput and acceptance (fa=on 1.98 vs fa=off 1.76).
Nothing was wrong with either measurement: greedy and sampled decoding emit
different text, and acceptance is a property of the text. Any acceptance figure
is only comparable against another taken with identical sampling parameters,
prompt and seed. The harness pins all of those for exactly this reason.

## KV cache: unified vs not, and why draft recipes look like they prefer it

The harness runs `--parallel 1`, and its logs show `kv_unified = 'false'`. That
is not a choice about speculation: `kv_unified` only defaults to true when
`--parallel` is left at *auto*, and the server then also picks `n_parallel = 4`
(`server.cpp`: "n_parallel is set to auto, using n_parallel = 4 and
kv_unified = true"). Pinning `--parallel 1` leaves it false. Most recipes that
appear to recommend a unified cache simply never set `--parallel`, so unified is
what they get by default rather than something chosen for drafting.

With a single slot it makes no difference, measured three ways:

| | non-unified | unified |
|---|---|---|
| mean accepted length | 1.98 | 1.98 |
| decode t/s (2 interleaved runs) | 33.38, 33.76 | 26.81, 33.25 |
| full-attention KV buffer | 512.00 MiB (32768 cells, 4 layers) | 512.00 MiB (identical) |
| windowed KV buffer | 40.00 MiB (1024 cells, 20 layers) | 40.00 MiB (identical) |

Acceptance is byte-identical and the allocation is identical, so speculation
neither benefits nor suffers. The decode spread is one slow run out of four
(26.81) with the other three agreeing within 1.5%.

It *does* matter once there are several slots, and there the recipes are right
for a different reason. `llama.h` documents it directly:

```
// NOTE: setting to false when n_seq_max > 1 can cause bad performance in some cases
// try to disable when n_seq_max > 1 for improved performance when the
// sequences do not share a large prefix
```

and the server notes that without a unified cache, clearing a slot frees no
reusable room, so idle slots keep their KV in VRAM. So for multi-slot serving,
prefer `-kvu` (or just leave `--parallel` unset) unless your slots genuinely do
not share prefixes. For the single-stream measurements here it is a no-op.

## Predicting speculative decoding, and where the numbers mislead

Two mistakes made the prediction unusable on gemma-4, both found by comparing
it against a real server run.

**Draft cost depends on which kind of draft you have.** An inline MTP head
reuses the target's output head, so that read is charged per drafted token. A
*separate* draft model is its own small network. Charging the target's 680 MiB
head for gemma-4's 42 MiB assistant overstated each drafted step by **16x**.

**Cost is fitted to measurements.** Static traffic modelling cannot know
per-step latency or the draft model's own attention, which dominate at long
context. So the report now fits `pass cost = t_base + n x t_draft` by least
squares over the measured `ms/pass`, and inspects marginal cost first: on the
test machine a drafted step costs ~4.6 ms up to n=6 and ~36 ms at n=8, so a
single line through both regimes matches neither. Points past a step costing 3x
the median are excluded and reported as a cliff rather than extrapolated.

The result, against the same run:

| n_max | predicted | measured | previous model |
|---|---|---|---|
| 2 | 30.22 | 30.41 | 23.12 (-24%) |
| 4 | 28.25 | 28.81 | 20.34 (-29%) |
| 6 | 25.20 | 25.66 | 17.37 (-32%) |
| 8 | not extrapolated | 14.00 | 15.02 |

The pre-run table (before any measurement) remains, but is labelled an *upper
bound*: it charges traffic and cannot know per-step latency, so on gemma-4 it
predicts ~60 t/s where reality is ~26.

### Read the throughput numbers as the top of the range

`prefill` comes from `llama-bench` pp512 and `decode` from a depth-0 run, and the
sweep runs at whatever context its prompt fills (~600 tokens). Both fall as
context grows. A real 31K-context serving run of the same model and preset
measured 22-31 t/s decode against the 30.7 t/s the sweep reported, and prefill
of 560-909 t/s against pp512's 330.

Acceptance is workload dependent too: the same preset measured mean len 1.90 on
the sweep prompt and 1.61-2.80 across the two prompts in the run, while a real
long-context session showed 2.16-2.96. A higher acceptance shifts the optimum
toward a *larger* `--spec-draft-n-max`, so pointing `--prompt` at your own
workload matters more than any tuning the harness can do without it.

## Summarising a run and the generated model preset

Every run writes `llama-models-options.ini` next to its `summary.tsv`, and the
same summary can be reproduced later without touching the GPU:

```sh
./llmfit.sh --summary run_hemmingway --hf org/Repo-GGUF:Q4_K_M
# or, equivalently, at any time:
./llmfit.sh --summary run_hemmingway                     # section name comes from the run
./llmfit.sh --summary run_hemmingway --ini /tmp/tuned.ini   # custom destination
```

The preset is a standard llama.cpp **model preset** (`--models-preset`), not an
invented format — keys are command-line arguments without the leading dashes:

```ini
version = 1

[*]
device       = Vulkan0
flash-attn   = on
cache-type-k = q4_0
cache-type-v = q4_0

[org/Repo-GGUF:Q4_K_M]
ctx-size             = 32768
spec-type            = draft-mtp
spec-draft-n-max     = 4
spec-draft-p-min     = 0.0
spec-draft-type-k    = q4_0
spec-draft-type-v    = q4_0
spec-draft-device    = Vulkan0
temp                 = 1.0
```

Use it with:

```sh
llama-server --models-preset run_hemmingway/llama-models-options.ini
curl -X POST localhost:8080/models/load -d '{"model":"org/Repo-GGUF:Q4_K_M"}'
```

Two properties of the format are worth knowing, and the generator accounts for
both:

- **Section names that match an existing model merge into it.** The generator
  names the section after the HF repo id when `--hf` was used, so the preset
  attaches to the cached entry. When only `--model PATH` was given it emits an
  explicit `model = <absolute path>` instead, because the section name then has
  nothing to match against.
- **`[*]` applies to every model in the file, not just this one.** Verified
  behaviour: a second, unrelated cached model inherited the `[*]` keys when the
  preset was loaded. The generator therefore puts only model-independent values
  (device, flash-attn, KV types) in `[*]` and keeps `ctx-size`, batching and the
  speculative group in the model section.

The preset carries a comment header with the measured numbers, the ceiling
analysis, and a note that the sampling values were held fixed during the sweep
rather than tuned — so the file explains itself when read months later.

## Replaying a recorded OMP session

Everything above tunes against a synthetic prompt, and acceptance is workload
dependent -- so the most faithful workload is one you actually ran. OMP records
every session as JSONL under `~/.omp/agent/sessions/<project>/`, and those
recordings name the model that served them, which for this machine is the local
llama-server. Replay one directly:

```sh
./llmfit.sh --replay ~/.omp/agent/sessions/-src-myrepo/2026-09-22T13-13-42-159Z_....jsonl
./llmfit.sh --replay SESSION.jsonl --replay-turns 20 --replay-nmax 6
```

`--replay` skips the sweep. It extracts the recorded conversation, starts the
tuned server, and posts each turn **in order on a single slot**, so the server
reuses the shared prefix and the context grows exactly as it did live. Each
turn's prefill, decode and acceptance are then measured at the depths the real
workload used, rather than at a synthetic prompt's depth.

The recording format is one typed JSON object per line; the messages carry
`role` plus a content list of typed blocks, and a prompt is emitted before every
*assistant* message because that is precisely the text the server received.
Emitting only at user turns would miss the tool results, which dominate a coding
session's context.

```
  turn  prompt tok  prefill t/s  out tok  decode t/s  mean len
     0        4333        649.5      128       28.22      1.92
     1         872        530.5      128       36.96      2.52
     2         139        263.0      128       26.29      1.79
```

`prompt tok` counts only **new** tokens: turns 1 and 2 processed 872 and 139 of
their ~4.6K-token prompts because the prefix was already cached. That reuse is
what the real session experienced, so it belongs in the measurement.

Notes and limits:

- **The model is read from the recording.** A session may name several
  (`local-llama/gemma-4-E4B-it-Q4_0` alongside
  `local-llama/ggml-org/gemma-4-E4B-it-GGUF:Q4_0`); the harness picks the first
  that resolves locally. Pass `--hf`/`--model` to override.
- **Sampling is pinned** to the same values the sweeps use (temp 1.0, seed 42),
  not whatever the session originally used, so replays of different
  configurations stay comparable. Acceptance is only comparable at identical
  sampling settings.
- **Recorded timings are preserved but not comparable.** Assistant records carry
  `usage`, `ttft` and `duration` from the original run; `--show-usage` prints
  them, but they were taken with whatever configuration was live at the time.
- A prompt is capped at 400K characters (keep the recent tail) via
  `REPLAY_MAX_CHARS`; a long session otherwise grows without bound.

## Output files

| File | Contents |
|---|---|
| `probe.json` | everything phase 0 discovered: device, model facts, bandwidth, ceilings, prediction |
| `llama-models-options.ini` | ready-to-use llama.cpp model preset with the tuned options |
| `summary.tsv` | one row per run — the machine-readable result set the report reads |
| `<name>.log` | full `llama-server` log (`-lv 3`), including the timing and acceptance lines |
| `<name>.resp.json` | the generated completion, kept so outputs from different configurations can be diffed (with a fixed seed they should be identical) |
| `<name>.power` | GPU power / temperature / memory samples taken during the run |

## Why the sweep is built the way it is

Four properties are deliberate; each one silently corrupted results when it was
missing during development.

- **Fixed seed.** Acceptance length depends on the text being generated. With a
  random seed every configuration produces different output, so throughput
  differences reflect luck rather than merit. `SEED=42` is baked into every
  request.
- **Opposite-order rounds.** Decode throughput on a power-limited part drifts
  20–35 % between a cold first run and sustained load (AMD STAPM behaviour). A
  single sequential sweep therefore ranks configurations by position. `--rounds`
  runs the set twice in opposite order and averages.
- **One server at a time.** Two `llama-server` processes on one GPU thrash the
  shared memory window and roughly halve throughput with no error message.
  A stale server holding the port is killed before each run.
- **`curl -fsS` readiness.** `/health` returns `503 Loading model` while the
  model loads; a bare `curl -s` treats that as success, so requests fire too
  early and produce empty responses.

The report also groups repeats of the same configuration and only crowns a
winner among configurations measured in as many rounds as the rest, so a single
lucky round cannot win. A faster single-round result is printed as
`NOT CONFIRMED` instead.

Speculative decoding was verified not to change output: with a fixed seed, the
generated text is byte-identical (sha256) for MTP alone, MTP + `ngram-mod`, and
MTP + `ngram-simple`. Only the n-gram-only configuration (no MTP) produced
different text, which is why the harness always keeps `draft-mtp` in the
`--spec-type` list rather than replacing it.

## Worked example

`run_hemmingway/` is a real run: qwen3.8 27B dense Q4_K_M (model architecture
qwen3.5, 16.24 GiB, MTP head present) on a Radeon 890M iGPU with 128-bit
DDR5-5600.

```
-- raw benchmark (llama-bench) --
  prefill : 117.65 t/s
  decode  : 4.20 t/s
  achieved effective bandwidth : 73.2 GB/s
  theoretical bandwidth        : 89.6 GB/s  -> 82% of peak
  decode ceiling (theoretical) : 4.51 t/s
  decode efficiency            : 93% of the bandwidth ceiling

-- speculative decoding, workload 'varied' --
  config                       rounds decode t/s  acc_len  ms/pass
  n-max=4 +ngram-mod                1       8.92     3.34      375
  n-max=4 fa=off kv=f16             1       8.76     3.34      382
  n-max=4 p-min=0.0                 2       8.72     3.34      384
  n-max=2 p-min=0.0                 2       8.62     2.68      307
  n-max=6 p-min=0.0                 2       8.46     3.97      470
  n-max=8 p-min=0.0                 2       5.73     4.50      792

  BEST of 2 round(s): n-max=4 p-min=0.0 -> 8.72 t/s
  no-speculation control   : 4.18 t/s -> 2.09x

-- workload sensitivity (prompts other than 'varied') --
  n-max=4 prompt=repetitive                         12.15     4.54
  n-max=4 +ngram-mod prompt=repetitive              12.12     4.54
```

How to read it:

- Decode is already at 93 % of the bandwidth ceiling, so **speculation is the
  only remaining lever** — no flag will move un-speculated decode much.
- Prefill is compute-bound; nothing in the spec/batching sweep helped it.
  A smaller model is the only route to a large prefill gain.
- The `n=8` cliff is real (reproduced in both rounds), not noise.
- n-gram speculation bought **nothing** here: on varied input it is within noise
  of plain MTP, and on the repetitive prompt it is slightly *worse*. That is a
  useful negative — it is only worth enabling for workloads that echo their
  input (summarisation, transformation, large-scale reformatting).
- Acceptance is workload dependent: the same model accepts 3.34 tokens per pass
  on varied code but 4.54 on structured records, so the achievable decode rate
  moves with the task.

## Limitations

- **An APU's power cap is shared with the CPU, so `platform_profile` matters a
  lot -- and it must be A/B-ed with interleaved repeats.** On this machine,
  measured `b, p, p, b` and averaged:

  | profile | prefill | decode |
  |---|---|---|
  | balanced | 310.8 t/s | 14.31 t/s |
  | performance | 461.6 t/s | 20.52 t/s |
  | | **+48.5 %** | **+43.4 %** |

  Balanced was also much noisier run to run (253.95 vs 367.55 t/s, +45 %, versus
  424.11 vs 499.06 for performance). This is worth stating because a *sequential*
  before/after test suggested the opposite sign in both directions during
  development: -30 % for performance when balanced ran first, +22 % when it ran
  second. Any single pair of runs across a state change will mislead; interleave
  and repeat. The harness prints this and never changes the setting itself.

- **Machine state can move numbers further than any flag.** Two serving runs of
  the same model and preset, at the same context and with byte-identical
  acceptance (0.49, mean len 2.96), differed by 1.9x: 561 vs 294 t/s prefill and
  30.7 vs 20.1 t/s decode. Competing load and the power cap explain the gap. Any
  comparison made across different machine states - including the sweeps in this
  harness - is only valid because the candidates are interleaved and averaged.
- **Discrete GPUs need `--mem-bus-bits`** for a theoretical bandwidth; the
  achieved figure is still computed without it.
- **`--peak-tflops` is not auto-detected.** Without it the prefill ceiling and
  efficiency are omitted, since inventing a peak would make the efficiency
  number meaningless. Use the vendor's rating for the precision you are running
  at.
- **The prediction is a first-order model** (see the caveat above). It is a
  sanity check and a starting point for the candidate list, not a replacement
  for measurement.
- **One model at a time.** The harness assumes the whole model fits the device;
  it does not tune CPU/GPU layer splits for models that do not.
- Throughput figures are specific to the prompts in `prompts/` and to single
  stream, batch-1 serving. Concurrent requests change the picture entirely.
