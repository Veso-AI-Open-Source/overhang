# Worklog

## 2026-07-17 (later) — decode 3.3 → 7.5 tok/s: the memory-compressor discovery

**Status: gemma 26B decodes at ~5.5–7.5 tok/s warm through the daemon
(was 2.4–3.3). Oracle still exact; 9/9 daemon tests. Ladder RAM honest.**

The investigation, in order (each step measured):
1. Hypothesis "IO-bound, raise QCACHE 32→64": hit rate 65%→92% but decode got
   SLOWER — the box went 12 GB into swap. Explicit 6 MB preads are cheaper
   than 4 KB swap faults.
2. `sample` profile showed threads waiting (cvwait 2.9k vs matmul 1.2k
   samples): ~1000 OMP micro-regions/token. Restructured the expert path to
   ONE coarse parallel region (serial collect+load → parallel per-hit compute
   with `matmul_q_st`/`matvec_st` → serial weighted reduce). Sync gone from
   profiles… but throughput barely moved. (OMP_WAIT_POLICY=active made it
   WORSE — spinning on 4P+6E asymmetric cores.)
3. Per-phase instrumentation (stderr `phases:` line per request) found the
   real sinks: **router-lookahead prefetch cost ~115 ms/token at decode**
   (macOS F_RDADVISE is expensive; LRU already ~85%) — now prefill-only
   (S>=8). route: 6.9s → 0.2s per 60 tok.
4. Expert "compute" was 143 ms/token for ~3 ms of int8 math → **macOS memory
   compressor**: LRU-cold expert slots get compressed under pressure, so a
   cache HIT pays a decompression fault costlier than the pread it saved.
   QCACHE sweep (60 warm tokens): 16→4.8 tok/s, **24→5.5**, 32→2.7, 48→2.8,
   64→swap-thrash. Small-and-hot beats big-and-compressed.
5. Speculation at 24 slots is a 4x LOSS (verify batches touch ~3x more
   experts/layer → LRU thrash): 41.8s vs 10.9s. Now **opt-in**
   (`GEMMA_SPEC=1`) — it genuinely helped at 48 slots, so it's a
   big-RAM-machine feature.

Final defaults: QCACHE cap/default 24 (gemma.c + models.rs
`effective_qcache` + app catalog 10 GB est). Measured through the daemon
after restart: **80 tok in 10.7s decode (7.5 tok/s), prefill 2.5s, hit 70%,
ladder 8.9 GB**. The per-phase `phases:` stderr line stays in as permanent
diagnostics.

Meta-lesson for all engines on macOS: RAM budgets must target the
UNCOMPRESSED working set, not "fits in free RAM" — the compressor turns
oversized caches into hidden swap. qwen's 96-slot config likely deserves the
same sweep.

## 2026-07-17 — Gemma 4 26B-A4B LIVE: converted, loaded, chatting end-to-end

**Status: real Gemma 4 26B answering correctly through daemon + app.
Container `colibri/models/gemma4_26b_i8` (26.04 GB, 8 shards + index).**

- Download saga: HF `snapshot_download` stalled silently TWICE mid-50GB-shard
  (connection dies, process lives), and its resume does NOT survive a restart
  (per-session `.incomplete` suffix — the 31 GB partial was orphaned).
  Fix that worked: rescue the partial into a plain dir and finish with
  `curl -C -` (+ `--speed-limit 1024 --speed-time 60` to kill silent stalls,
  retry loop). **Lesson: for single huge HF files, curl + byte-resume beats
  the python client.** Source snapshot kept at `colibri/_dl_gemma4_src/`
  (52 GB — delete to reclaim if never reconverting).
- Conversion: `convert_gemma.py` ran clean — 7,680 expert tensors split
  per-expert int8, 235 dense int8, vision tower dropped (356 tensors),
  `language_model.` prefix stripped. 52 GB bf16 → 26.04 GB int8.
- **Template correction (important)**: Gemma 4 is NOT gemma1-3 format. The
  canonical template (chat_template.jinja 2026-07-09) uses
  `<|turn>role\n...<turn|>` turns and a THOUGHT CHANNEL; non-thinking
  generation prompt = `<|turn>model\n<|channel>thought\n<channel|>` (empty
  thought → direct answer). First chat with the old `<start_of_turn>` template
  produced channel-marker garbage; with the canonical template output is
  clean. Stops: `<turn|>`, `<|channel>`, `<eos>`.
- **Measured on M5 (26 GB), engine `gemma-metal`, QCACHE 32**:
  load→ready **1.8 s**; prefill ~4–5 s (short prompts); decode **~3.3 tok/s**;
  expert-cache hit **65–67%**; phys_footprint **11 GB** (peak 12) — vs the
  ~52 GB bf16 nominal and our own 15 GB estimate. Multi-turn + SSE streaming
  verified; app Library shows it installed + in the slot.
- Daemon config gained `[engines] gemma4_text = "../gemma-metal"`.
- Known following-up: warm-append likely misses on gemma multi-turn (the
  thought-channel generation prefix means re-rendered history may not
  prefix-match engine state token-for-token — needs a token-level check);
  ladder RAM estimate uses config qcache (96) not the engine's cap (32), so
  the 22 GB shown overstates the real ~11 GB.

## 2026-07-16 (night) — performance pass: all six levers landed

**Status: oracle still 28/28 TF (now run as ONE batched pass) + 20/20 greedy;
9/9 daemon tests; GPU kernels numerically verified. 26B download in flight.**

1. **Batched prefill + expert batch-union** (`gemma.c` `step_batch`): S token
   per layer; gli expert si caricano per UNIONE sul batch (uno load per
   expert colpito, non 8 per token). Il gate dell'oracle ora esegue il TF in
   un singolo passo batched S=28 — attention/masking/union del percorso
   batched validati direttamente contro transformers.
2. **lm_head int8** (solo container): il vocab 262K×2816 f32 era ~3 GB di
   traffico per token decodificato; quantizzato per-riga al load. Ri-provato
   esatto: TF int8 (IDOT=0) combacia 28/28 con il riferimento python
   fake-quant CON lm_head quantizzato (untied).
3. **Router-lookahead prefetch**: al layer L si stima il routing di L+1 sul
   residuo corrente e si `st_prefetch`ano i suoi expert (readahead async).
4. **Warm-append nel daemon** (`engine.rs`): `history` degli id nello stato
   dell'engine; se i nuovi id la estendono esattamente → `reset:false` col
   solo suffisso (prefill per turno O(nuovi token), non O(storia intera));
   retry freddo automatico se il warm viene rifiutato prima di emettere
   (es. context full). Cleared su spawn/eject/mark_down/errore. Vale per
   TUTTI gli engine. Unit test `warm_append_reuses_prefix`.
5. **Decode speculativo n-gram** (`gemma.c --serve`): bozza di ≤4 token dal
   contesto (match di bigramma), verifica in UN passo batched con logits
   per-posizione, rollback di `m.pos` sul primo mismatch. GREEDY-EQUIVALENTE
   verificato: 40/40 token identici con GEMMA_SPEC=0 vs 1 su due richieste
   (inclusa una continuazione warm). Attenzione pura → il rollback e' solo
   `pos` (non fattibile cosi' su qwen/deltanet).
6. **Metal 4 GPU prefill** (`make gemma-metal`): om_metal esteso con kernel
   `gelu_mul`, `omm_moe_act(..., gelu)` (omm_moe = wrapper silu, olmoe-metal
   intatto) e `omm_dense_q` (dq_gemm su dense int8 residenti in arena;
   fallback -3 se K%64||N%32). gemma.c: scale in arena (slot_alloc), unione
   expert → un command buffer GPU quando S≥8. Verifica NUMERICA dei kernel
   vs riferimento CPU esatto: normalized RMSE ~0.001 (= rounding half, non
   bug di wiring; il max-rel-err e' fuorviante sotto cancellazione).

Nota misura: numeri di throughput reali arrivano col container 26B — il tiny
non e' indicativo. `prefill_s`/`decode_s`/`hit` gia' per-richiesta.

## 2026-07-16 (later) — Gemma 4 oracle + engines/gemma.c: ORACLE GREEN

**Status: `make check-oracle-gemma` = 28/28 teacher-forcing + 20/20 greedy
EXACT, first run after implementation. qwen oracle still green.**

- `engines/gemma.c` (~330 lines, pure f32 resident — the tiny is 0.61M params;
  int8 expert streaming comes with the real container). Validation main does
  BOTH teacher-forcing argmax over all 28 positions and the greedy-20 match,
  stricter than the qwen oracle gate.
- The math that had to be exactly right (all from modeling_gemma4.py, 5.13.1):
  embed × √hidden; **plain-weight RMSNorm** (not gemma2/3's 1+w), eps inside
  the mean; **values RMS-normalized** via weightless `v_norm`; full-attention
  layers reuse the **pre-norm k_proj output as values** (`attention_k_eq_v`,
  no v_proj tensor); **attention scaling = 1.0** (no 1/√d — q_norm/k_norm do
  the work); per-type rope (sliding: default θ10k over full head_dim; full:
  "proportional" θ1M with `int(0.25·32)//2 = 4` live frequencies, the rest
  zero → identity dims); router on the **pre-norm residual** (weightless RMS ×
  `router.scale` × D^-0.5, softmax f32, top-k renormalized to sum 1, then ×
  `per_expert_scale`); dense MLP and MoE run **in parallel** off the same
  residual and are summed after `post_feedforward_layernorm_{1,2}`; fused
  expert tensors (gate = first mi rows of `gate_up_proj[e]`); `layer_scalar`
  multiplies the whole layer output; tied lm_head + logit softcap 30.
- Makefile: `gemma` target + `check-oracle-gemma`; `all` now builds all three
  engines.

- Target: **google/gemma-4-26B-A4B** (128 experts, top-8, ~3.8B active) — the
  best new-model candidate from a web sweep: modern (2026), ~26 GB int8, and
  architecturally an OLMoE sibling (dense attention + routed FFN experts), so
  `olmoe.c` is the porting template. Real config.json fetched ungated from HF.
- `tools/make_gemma_oracle.py` (make_qwen_oracle.py methodology): tiny-random
  `gemma4_text` at toy dims mirroring the real config structurally — 5:1
  sliding/full layer pattern, per-layer-type rope (full: proportional +
  partial_rotary 0.25; sliding: default θ10k), `attention_k_eq_v`, global kv
  heads/dim on full layers, MoE block 8 experts top-2, gelu_pytorch_tanh,
  final_logit_softcapping 30, tied embeddings. `sliding_window=8` < the
  28-token reference so the window mask is actually exercised.
- Outputs: `oracle/gemma_tiny/` (0.61M params) + `oracle/ref_gemma.json`
  (greedy 20 + TF argmax). Deterministic across runs; reload-from-disk TF
  check baked into the script. Ran with the colibri venv
  (transformers 5.13.1 has gemma4).
- **Port notes from the tensor dump** (surprises vs qwen/olmoe):
  experts are FUSED 3D tensors (`experts.gate_up_proj`/`experts.down_proj`,
  like the 35B qwen container convert_qwen.py already handles — NOT
  OLMoE-style per-expert); router has `proj.weight` + `scale` +
  `per_expert_scale`; per-layer `layer_scalar`; five sandwich norms around
  the FFN (pre/post_feedforward, _1, _2 variants); dense `mlp.*` coexists
  with routed experts every layer; no lm_head tensor (tied); full-attention
  layers have one fewer tensor than sliding ones — read modeling_gemma4.py
  for k_eq_v semantics (both k_proj and v_proj exist in the snapshot).
- Next (engine math is DONE, see entry above): `convert_gemma.py` for the
  real 26B-A4B container (int8 experts + `.qs` scales, colibri style; decide
  fused-vs-per-expert layout), int8 expert streaming + `--serve` in gemma.c,
  batch prefill. NB `models.rs` est-RAM expert detection assumes
  `.experts.<id>.` names — the fused layout will need handling at conversion
  or scan time.

## 2026-07-16 — load-pipeline hardening: rollback, tok coherence, eject-wait, completeness gate

**Status: daemon builds clean (zero warnings), 8/8 unit tests, both app
integration tests pass against the mock daemon.**

### Done
- **Rollback-safe `/engine/load`** (`daemon/src/engine.rs`): a failed spawn now
  restores the previous `model_dir`/`engine_bin`, so the next request lazily
  respawns the *previous* model instead of retrying the broken container
  forever. This was a real brick: loading `olmoe_i4` (no `--serve`) killed the
  running qwen and 500'd every chat until a manual re-load. Restore is lazy by
  design (no eager re-spawn — seconds of RAM churn on a path where you're
  likely retrying a different container).
- **Eject waits for exit**: `eject()` is now async and awaits `child.wait()`
  after SIGKILL, so a load-swap never has two multi-GB engines resident at
  once. `mark_down` stays drop-only (child already dead there, respawn lazy).
- **Tokenizer/engine coherence**: chat captures ONE `Arc<Tok>` per request,
  read *inside* the engine mutex (encode moved there too — microseconds), and
  decodes with that same captured tok. Since `/engine/load` swaps tok while
  holding the engine mutex, tok and engine can now never be observed
  mismatched. No epoch counter needed.
- **Container completeness gate** (`models::verify_container`, called in the
  load handler *before* any eject): index-listed shards must exist with
  parseable safetensors headers (or ≥1 valid `.safetensors` when there's no
  index). Half-copied containers now 422 with a clear message instead of
  ejecting the working engine and crashing opaquely. 422 (not 404 — the app
  latches 404 as "endpoint unsupported").
- Tests: `engine.rs` test module (rollback with/without prior engine, eject
  no-op semantics) driven by `tests/mock_engine.py` via `engine_cmd`; the mock
  gained a SNAP-`*_bad` exit hook. Four `verify_container` unit tests in
  `models.rs`. New header-only fixture
  `tests/fixtures/models/mock/model.safetensors` (required — the mock
  container would otherwise fail the completeness gate).
- Verified live against a mock daemon on :11545 (the real daemon on :11544 was
  left untouched): load→chat, 404, 422 with engine surviving, and the exact
  brick scenario — failed load → 500 → streaming chat lazy-respawns the
  previous model, `/status` active again.

### Also done same day — olmoe.c `--serve` (the second tray is now loadable)
- Clicking Load on olmoe_i4 in the app exercised the rollback fix in prod: the
  engine exited before the ready line (`ref.json: No such file or directory` —
  run-mode treated `--serve`'s absence as an oracle run), daemon reverted, the
  subsequent qwen36_i8 load worked. Root cause was the known gap: olmoe.c had
  no serve mode.
- Ported the PROTOCOL.md v1 serve loop from qwen.c to olmoe.c: `--serve`
  dispatch in main, QCACHE env → cache slots (int8 colibri container, bits=8),
  KV allocated once at max_t=8192, `load_eos` (generation_config.json →
  config.json), same request/response lines. One improvement over qwen: olmoe's
  `step()` batches, so prefill runs in a single batched call instead of
  token-by-token.
- Verified live on :11544: `/engine/load olmoe_i4` → 200 ready in 0.7 s (eject
  of qwen waited for exit first — the new eject-wait path), chat 200 with
  prefill 2.7 s / decode ~8 tok/s / hit 0.83, `/status` reconciles
  (olmoe_i4 active, qwen36_i8 not).
- **Honest note**: chat quality through olmoe is off-template — the daemon
  hardcodes ChatML but OLMoE-instruct was trained on the zephyr-style
  `<|user|>`/`<|assistant|>` template, and its vocab contains anonymization
  literals (`|||IP_ADDRESS|||` showed up in a reply). Engine/protocol level is
  correct; per-container chat templates are a daemon feature for later.

### Also same day — app: status polling + visible load errors; olmoe_i4 removed
- **Root-caused the "loses state / needs manual refresh" UX**: (a) `load()` in
  `app/src/api.rs` only handled 404 — a 500/422 from the daemon was silently
  swallowed, so a failed load looked like "starts for a sec then nothing";
  (b) the app fetched `/status` only at startup and after its own commands, so
  any daemon-side change made by another client (curl, tests) stayed invisible
  until the manual refresh button.
- Fixes: backend command loop now re-polls `/status` every 2 s between
  commands (fingerprint gating means no repaint when nothing changed, so idle
  CPU is preserved); failed loads store `(model, message)` in
  `Shared.load_error` — parsed from the daemon's error envelope — and the
  failing tray's slot strip shows "✗ load failed: <msg>" until refresh/retry
  clears it; `fetch_status` no longer chains `fetch_system` (system re-probed
  on startup + explicit refresh only, not every 2 s poll).
- Verified: rebuilt (zero warnings), relaunched the app, loaded qwen36_i8 via
  curl (external client) — the footer showed "qwen36_i8 loaded" within the
  poll interval with no refresh click (screencapture-verified).
- **olmoe_i4 container deleted** (~6.9 GB, decision: int8 only). Ejected
  first, then removed `colibri/models/olmoe_i4`; ladder now shows qwen36_i8
  alone. The olmoe.c `--serve` code stays (works for any future int8 olmoe
  container; regenerate containers via colibri's convert_olmoe.py).

### Also same day — Model library redesign: list rows, full names, catalog
- Library tab rebuilt as **full-width list rows** (was a wrapped card grid
  that clipped badges/text at 300 px): one row per model — disc, full name,
  `container · disk · RAM` meta, then state + fits badge right-aligned.
  Shared `row_shell()`/`fits_badge()` builders; "This machine…" line removed
  (lives in the System tab).
- **Full model names**: rows show the real model name with the container id
  demoted to the meta line. Installed container verified against the HF cache
  + colibri WORKLOG: qwen36_i8 = **Qwen/Qwen3.6-35B-A3B** (int8 conversion).
  The ladder/engine `name` stays the container dir — display-only mapping via
  the catalog, so `active` reconciliation is untouched.
- **Catalog section** ("catalog — not installed"): models the shipped engines
  can actually run, with the colibri command that produces each container —
  Qwen3.5-122B-A10B (~61 GB, roadmap item), OLMoE-1B-7B-Instruct int8,
  GLM-5.2 (~700 GB container). Est. sizes marked with `~`; installed rows
  always show daemon-measured numbers instead.
- **Fit outline**: any row that doesn't fit gets a warn border + explicit
  reason ("won't fit — needs ~700 GB disk"); fit check mirrors the daemon
  (RAM ≤ 0.9 × total, disk ≤ free). GLM-5.2 renders as the worked example.
- Screenshot-verified (list3/library4): no clipping, sections read cleanly.
  Catalog entries are display-only (no downloader wired); est. RAM figures
  for non-installed rows are rough until a container exists to measure.

### Follow-up
- Surface `verify_container` in the `/status` ladder (`"complete": bool`) +
  app UI, so half-copied containers are visible before a load attempt.
- Wire a real download/convert flow behind the catalog rows (currently
  display-only pointers to colibri tools).
- Per-container chat template (render_chatml is hardcoded ChatML; wrong for
  OLMoE-instruct — moot while qwen36_i8 is the only container).

## 2026-07-15 — system discovery, /engine/load, measured capacity ladder

**Status: daemon + app build clean; all tests pass; verified end-to-end against
mock and real (qwen36_i8) engines.**

### Done
- **Removed all hardcoded device/model info from the daemon.** The `LADDER`
  const in `server.rs` (invented display names + RAM budgets) is gone. `/status`
  now builds the capacity ladder by scanning the model root: disk measured from
  file sizes, RAM estimated from safetensors headers (dense resident bytes +
  `n_layers × qcache × avg-expert-bytes` — the same accounting `qwen.c` prints
  at startup) + `config.json`. New `daemon/src/models.rs` (+ unit test).
- **Naming mismatch fixed**: ladder rows are named by container dir
  (`qwen36_i8`), identical to the engine ready-line model (SNAP basename), and
  `/status` now sets `active` per row — app trays show "in the slot" correctly.
- **System discovery** (`daemon/src/system.rs`, cached at startup): chip
  (sysctl), arch, OS/kernel, logical/physical cores, P/E split
  (hw.perflevel*), RAM, GPU cores (ioreg AGXAccelerator), Metal 4 (macOS 26+
  on arm64), unified memory. Exposed as `GET /system` with model-volume
  free/total + model root. On this machine: Apple M5, 10 cores (4P+6E),
  10 GPU cores, 26 GB, metal4=true.
- **`POST /engine/load {"model": name}`** implemented (was the top wiring gap):
  validates the name (400/404), picks the engine binary from the container's
  `model_type` (new `[engines]` config table, default family mapping
  qwen*/olmoe/glm as sibling of `engine_bin`), swaps the tokenizer to the
  container's `tokenizer.json`, ejects, spawns eagerly so failures surface in
  the response. Tokenizer is now behind `RwLock` (`AppState.tok()`).
- **App System tab** (`app/src/views/system.rs`): machine / storage / daemon
  panels rendered from `GET /system` (nothing hardcoded client-side), ✓ badges
  for Metal 4 + unified memory, rescan button; `system_unsupported` 404 latch
  like eject/load. `OVERHANG_TAB=system` debug hook.
- Integration tests (`app/tests/daemon.rs`): fixed stale >20-char assertion
  (predates the in-repo 3-token mock engine); added `system_discovery_and_load`
  (discovery report + load reconciles `active` by name). Both pass against
  the daemon running `tests/overhangd.mock.toml`.
- **Verified end-to-end with real weights**: `make check-oracle` 20/20 exact;
  daemon on `overhangd.toml` → measured ladder (olmoe_i4 7.4 GB disk / est
  10.5 GB RAM; qwen36_i8 39.0 GB disk / est 16.7 GB RAM — README's "36 GB"
  understates the container on disk); `/engine/load qwen36_i8` ready in 3.3 s;
  streamed chat at ~1.7 tok/s incl. prefill; eject OK.

### Known gaps / honest notes
- `engines/olmoe.c` has **no `--serve`** — only qwen speaks the protocol, so
  loading `olmoe_i4` fails cleanly (500 "engine exited before ready line",
  daemon survives). Port the serve loop to olmoe.c to make the second tray
  loadable.
- olmoe_i4 RAM estimate uses the f32-fallback path (int4 container's packing
  isn't modeled), so 10.5 GB is rough. Fine for fits/doesn't-fit at 26 GB.
- Load progress bar still keys off `resident_gb` from /events, which the
  engine only emits while generating — loads show indeterminate dots.
- `OVERHANG_TAB=system` didn't visibly select the tab in one manual launch
  (unconfirmed; tab button itself works). Worth a look next session.

## 2026-07-14 — app: gpui port, polish pass, CD-changer footer/library

**Status: gpui UI is the default build; egui kept as fallback. Nothing committed.**

### Done
- Ported `app/` from egui to pure gpui (zed pinned at rev `7a3a823`). `cargo run` opens
  the native "overhang" window; egui interim UI preserved under
  `cargo run --no-default-features --features egui-fallback` (`src/egui_legacy/`).
  `api.rs` unchanged by the port. Zero warnings on both feature sets.
- **Build gotcha worth remembering**: gpui renders *no text at all* unless the
  `gpui_platform` dependency has `features = ["font-kit"]` — no error, no warning,
  just blank glyphs. Cost an hour; now pinned in Cargo.toml.
- Theme tokens in `src/views/theme.rs` (dark, warm amber accent `#f08c3e`); no inline
  hex in views. Chat = bottom-aligned `list()` with bubbles, mono streaming text +
  block cursor, "reading prompt… N tok/s prefill" during dead air, tok/s pill in the
  composer. Stats = numeric tiles + amber gradient-fill sparkline (canvas/paint_path)
  fed by ws /events. Library = CD-changer trays (disc glyph, fits badge, recessed
  mono slot strip).
- CD-changer footer on all tabs: engine disc + name + state, live tok/s while
  generating, ghost Eject (warn hover). Eject verified against live daemon (200).
- Load flow: `Cmd::Load` → POST `/engine/load {"model": name}`; atomic (one engine op
  at a time — other trays show "slot busy", Eject/Send disabled mid-op); progress bar
  from /events `resident_gb` climbing toward the model's RAM budget, indeterminate
  dots when /events is silent. Verified end-to-end against the mock daemon.
- Perf: wake-driven, fingerprint-gated repaints (notify only when the visible
  snapshot changed). Idle CPU (release) ≈ 0.6% (0.12 s CPU / 20 s wall).
- api.rs additions (authorized): `Cmd::Eject`, `Cmd::Load`, `eject_unsupported` /
  `load_unsupported` latches (404 ⇒ "update overhangd" hints), `loading_model`;
  explicit refresh clears the latches so a daemon upgrade is picked up without
  restarting the app.

### Blocked / waiting on daemon (RESOLVED 2026-07-15: /engine/load landed, naming reconciled)

### Known stubs (app)
- Chat composer is a minimal key-handled div: no cursor movement, selection, or IME.
  Upgrade path: gpui `examples/input.rs` (EntityInputHandler).
- `gen_tok_s` includes prefill time in its denominator (reads low on long prompts).
- Debug/verification env hooks: `OVERHANG_ADDR`, `OVERHANG_TAB`, `OVERHANG_AUTOSEND`,
  `OVERHANG_AUTOEJECT`, `OVERHANG_AUTOLOAD`. Mock daemon (HTTP + SSE + ws /events,
  port 11545) lives in the session scratchpad as `mock_daemon_ws.py` — worth moving
  into `app/tests/` if we keep using it.

### Notes
- During endpoint probing I POSTed `/engine/eject` to the live daemon once (200) —
  engine was left unloaded.
- All UI verification was headless: per-window `screencapture -l`, no synthetic input.
