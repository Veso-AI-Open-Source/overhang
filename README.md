# overhang

**Run massive models on your small Mac.**

overhang runs mixture-of-experts language models whose weights exceed your
machine's RAM. Gemma 4 26B runs in ~9 GB of resident memory; a 35B runs on a
24 GB MacBook today; the same engine design scales to 122B and beyond,
bounded by disk, not memory.

It works by combining four things that are usually separate projects:

1. **Expert streaming.** MoE models activate a small fraction of their weights
   per token. The active experts are read from an int8 container on disk
   through an LRU cache; RAM holds a working set, not the model.
2. **Quantized storage at every tier.** Weights are int8 on disk, int8 in
   cache, int8 in the resident dense layers, and dequantized only inside the
   compute kernel. Bytes moved, not FLOPs, decide decode speed.
3. **Apple GPU Neural Accelerators.** Prompt processing and batched expert
   math run on the M-series matrix units via Metal 4 tensor ops
   (`matmul2d`), compiled at runtime — no Xcode. Weights arrive there
   zero-copy: unified memory means a disk read lands at an address the GPU
   uses directly. See [VesoAi/mtlgemm](https://github.com/VesoAi/mtlgemm)
   for the standalone kernels and measurements.
4. **Architectures chosen for the constraint.** The Qwen3.5/3.6 engine uses
   Gated DeltaNet linear attention: prompt cost grows linearly with length
   and the recurrent state is constant-size, so long contexts do not erode
   the memory budget that streaming depends on.

Every engine is validated **token-exact** against its `transformers`
reference on a tiny-random model before it touches real weights. If
`make check-oracle` is green, the math is right.

## Measured (M5 MacBook, 10 cores, 24 GB RAM, macOS 26.5)

| model | on disk | resident RAM | decode |
|---|---|---|---|
| Gemma 4 26B-A4B (int8) | 26 GB | **~9 GB** | 5.5–7.5 tok/s warm |
| Qwen3.6-35B-A3B (int8) | 36 GB | ~17 GB | 1.2–1.5 tok/s sustained, ~5 warm |
| OLMoE-1B-7B (int8) | 6.9 GB | ~10 GB | 12 tok/s (**140 tok/s** GPU prefill, token-identical to CPU) |

The Gemma 4 figures came out of a measured tuning pass worth reading about in
WORKLOG.md: on macOS the memory compressor silently squeezes oversized caches,
so a small hot expert cache (24 slots) beats a large compressed one (48+) by
2x. Size to the *uncompressed* working set, not to free RAM.

See ROADMAP.md for the named next steps — int4 container, MTP speculative
decoding, GPU prefill for the Qwen engine — each with the measurement that
defines success.

## Quick start

```
make                     # builds qwen + olmoe + gemma engines (clang; brew install libomp)
make check-oracle        # qwen: 20/20 token-exact vs the transformers reference
make check-oracle-gemma  # gemma4: 28/28 teacher-forcing + 20/20 greedy, batched path
make gemma-metal         # gemma with Metal 4 GPU prefill

# convert a model (needs python + torch + safetensors, one-time):
python tools/convert_qwen.py  --model <hf snapshot> --out ./models/qwen36_i8
python tools/convert_gemma.py --model <hf snapshot> --out ./models/gemma4_26b_i8

# run a prompt directly:
python run/qwen_run.py "your prompt" --n 100 --cache 96

# or run the daemon + native app (OpenAI-compatible API on :11544):
cd daemon && cargo run --release      # /v1/chat/completions, /status, /engine/load
cd app    && cargo run --release      # GPUI chat + model library + live stats
```

## Layout

```
engines/     one file per architecture: the math, nothing else (qwen, olmoe, gemma4)
include/     safetensors reader, json, portability shims
metal/       Metal 4 GPU backend (tensor ops, in-kernel dequant, zero-copy arena)
tools/       converters (HF -> int8 container) and oracle generators
oracle/      tiny-random reference models + expected tokens
run/         prompt wrappers and an agentic capability harness
daemon/      overhangd (Rust/axum): OpenAI-compatible API, engine lifecycle,
             per-container chat templates, warm-append, measured capacity ladder
app/         native macOS app (Rust/GPUI): chat, model library, live stats
```

## How it's built

This codebase is developed with **agentic AI coding** end to end — the
engines, daemon, app, converters, and the performance work were written and
verified in an agentic loop against hard oracles: every engine must reproduce
its `transformers` reference token-exactly before real weights, every
optimization must show its number in a measurement, and the WORKLOG records
what was tried, what failed, and why. The oracle-first culture is what makes
that loop safe: correctness is a green test, not a code review opinion.

## Contributing

The roadmap lists bounded, claimable work items, each with its validation
criterion. The oracle harness removes the usual risk: your change either
reproduces the reference tokens or it does not. First contributor to land a
substantial PR gets write access — this project is built to be community-run.

## Credits

This project stands on specific prior work; see CREDITS.md. In particular,
the expert-streaming design extends **[colibrì](https://github.com/JustVugg/colibri)**
by JustVugg, which runs GLM-5.2 (744B) on 25 GB machines and established both
the memory-hierarchy approach and the oracle-first validation culture this
project follows.

Apache-2.0. See NOTICE.
