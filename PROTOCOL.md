# overhang engine serve protocol (v1)

Engines run as child processes of `overhangd`. Transport: JSON Lines over
stdin/stdout. One request at a time (no pipelining in v1). Stats and logs go
to stderr, never stdout.

## Launch

```
SNAP=<container dir> QCACHE=<slots> ./qwen --serve
```

On ready, engine prints exactly one line to stdout:

```json
{"ready":true,"model":"<SNAP basename>","n_layers":40,"vocab":248320}
```

## Request (daemon -> engine, one line)

```json
{"id":7,"ids":[1,2,3],"n":128,"reset":true}
```

- `id`     opaque integer, echoed on every response line
- `ids`    token ids to feed (daemon owns tokenization and chat template)
- `n`      max new tokens (greedy, v1)
- `reset`  true: clear KV/recurrent state, position 0 (new conversation).
           false: append `ids` after current state (warm continuation).

## Response (engine -> daemon, one line per token)

```json
{"id":7,"tok":9419}
{"id":7,"tok":30}
{"id":7,"done":true,"n_out":2,"prefill_s":1.42,"decode_s":0.31,"hit":0.86}
```

- Engine stops early on EOS (id from generation_config) or `n`; `done` line
  always sent, always last.
- `stop` request (`{"id":7,"stop":true}`) is accepted mid-generation and
  answered with the `done` line. v1 engines may ignore it between tokens only.

## Errors

```json
{"id":7,"error":"<message>"}
```

Fatal errors (bad container, OOM) may also exit non-zero; the daemon treats
process exit as fatal and restarts with `reset` semantics.

## Daemon HTTP surface (overhangd)

- `POST /v1/chat/completions` — OpenAI-compatible; `stream:true` = SSE
- `GET  /v1/models` — installed containers
- `GET  /status` — engine state + capacity ladder. The ladder is measured,
  never hardcoded: containers discovered under the model root (siblings of
  `model_dir`), disk from file sizes, RAM estimated from safetensors headers
  (dense resident + `n_layers × qcache × avg expert size`). Row `name` is the
  container dir name — identical to the engine ready-line `model`, so
  `active` reconciles.
- `GET  /system` — hardware discovery: chip, arch, OS/kernel, logical +
  physical cores, P/E core split, RAM, GPU core count, Metal 4 support,
  unified memory, model-volume free/total, model root. Discovered once at
  daemon startup (disk per call).
- `POST /engine/load` `{"model":"<container dir name>"}` — eject the current
  engine and spawn on the named container. Engine binary picked by the
  container's `model_type` (config `[engines]` table, else a sibling of
  `engine_bin` named after the family). Tokenizer swapped to the container's
  `tokenizer.json`. Errors: 400 (bad name), 404 (no such container),
  500 (engine failed to start; previous model stays ejected).
- `POST /engine/eject` — unload the model, free RAM; lazy respawn on next
  request
- `GET  /events` — WebSocket: per-token stats for the UI viz
  `{tok_s, hit_rate, resident_gb, streamed_mb_s}`
