#!/Users/x/.local/share/uv/tools/mlx-lm/bin/python
"""Shim engine: serves an MLX-quantized model through PROTOCOL.md v1.

Lets overhangd host models whose kernels live in MLX (e.g. PrismML's
Bonsai 1-bit/ternary family, model_type qwen3_5 dense) without a native C
engine. The daemon still owns tokenization and chat templating; this shim
consumes token ids and streams token ids back, greedy, exactly like
qwen/olmoe/gemma --serve.

Launch (via daemon [engines] mapping): SNAP=<container dir> bonsai_mlx.py --serve
Warm continuation supported: reset:false appends ids to the live KV cache.
"""
import json
import os
import sys
import time


def main():
    snap = os.environ.get("SNAP")
    if not snap:
        print("set SNAP=<model dir>", file=sys.stderr)
        return 1

    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load

    t0 = time.monotonic()
    model, tokenizer = load(snap)
    n_layers = getattr(getattr(model, "args", None), "num_hidden_layers", 0) or 0
    vocab = getattr(getattr(model, "args", None), "vocab_size", 0) or 0
    print(f"model loaded in {time.monotonic()-t0:.1f}s", file=sys.stderr, flush=True)

    eos = set(tokenizer.eos_token_ids or [])
    base = os.path.basename(os.path.normpath(snap))
    print(json.dumps({"ready": True, "model": base, "n_layers": n_layers,
                      "vocab": vocab}), flush=True)

    greedy = lambda logprobs: mx.argmax(logprobs, axis=-1)
    cache = None
    history = []  # ids currently represented in `cache`

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        rid = req.get("id")
        if req.get("stop"):
            print(json.dumps({"id": rid, "done": True, "n_out": 0,
                              "prefill_s": 0.0, "decode_s": 0.0, "hit": 0.0}), flush=True)
            continue
        ids = req.get("ids") or []
        n = int(req.get("n") or 0)
        reset = bool(req.get("reset"))

        if reset or cache is None:
            cache = make_prompt_cache(model)
            history = []
        history.extend(ids)

        # prefill happens inside generate_step's first yield; feeding only the
        # new ids against the retained cache is the warm-append path
        prompt = mx.array(ids if ids else [history[-1]])
        tp0 = time.monotonic()
        gen = generate_step(prompt, model, max_tokens=n, sampler=greedy,
                            prompt_cache=cache)
        n_out = 0
        prefill_s = decode_s = 0.0
        td0 = None
        for tok, _ in gen:
            tok = int(tok)
            now = time.monotonic()
            if td0 is None:
                prefill_s = now - tp0
                td0 = now
            print(json.dumps({"id": rid, "tok": tok}), flush=True)
            history.append(tok)
            n_out += 1
            if tok in eos or n_out >= n:
                break
        decode_s = (time.monotonic() - td0) if td0 else 0.0
        print(json.dumps({"id": rid, "done": True, "n_out": n_out,
                          "prefill_s": round(prefill_s, 3),
                          "decode_s": round(decode_s, 3), "hit": 0.0}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
