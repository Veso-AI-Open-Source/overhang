#!/usr/bin/env python3
"""Run an arbitrary prompt through the colibri OLMoE streaming C engine.

The C engine (c/olmoe) is a token-validation harness: it reads prompt_ids from a
ref.json and generates len(full_ids)-len(prompt_ids) new tokens greedily. This
wrapper synthesizes that ref.json from a text prompt, runs the engine, and
decodes the generated ids back to text.

Usage:
  ./.venv/bin/python olmoe_run.py "Why is the sky blue?" [--n 64] [--raw]
                                  [--model ./models/olmoe_i4] [--cap 16] [--bits 8]
"""
import argparse, json, os, re, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--n", type=int, default=64, help="tokens to generate")
    ap.add_argument("--model", default=str(HERE / "models/olmoe_i4"))
    ap.add_argument("--cap", type=int, default=16, help="expert cache slots/layer")
    ap.add_argument("--bits", type=int, default=8, help="expert quant bits (2..8)")
    ap.add_argument("--raw", action="store_true", help="raw completion, no chat template")
    ap.add_argument("--exact", action="store_true",
                    help="IDOT=0: scalar f32 kernels, token-exact vs the transformers oracle (~2.7x slower)")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok_path = Path(args.model) / "tokenizer.json"
    if not tok_path.is_file():
        sys.exit(f"tokenizer.json missing in {args.model} (copy it from the HF snapshot)")
    tok = Tokenizer.from_file(str(tok_path))

    if args.raw:
        text = args.prompt
    else:
        # OLMoE-1B-7B-0125-Instruct chat format
        text = f"<|endoftext|><|user|>\n{args.prompt}\n<|assistant|>\n"

    ids = tok.encode(text, add_special_tokens=False).ids
    ref = {
        "prompt": args.prompt,
        "prompt_ids": ids,
        "full_ids": ids + [0] * args.n,  # dummy targets: we ignore the match count
        "text": "",
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(ref, f)
        refpath = f.name

    env = dict(os.environ, SNAP=args.model)
    if args.exact:
        env["IDOT"] = "0"
    try:
        binname = os.environ.get("OLMOE_BIN", "c/olmoe")
        out = subprocess.run(
            [str(HERE / binname), str(args.cap), str(args.bits), refpath],
            env=env, capture_output=True, text=True,
        )
    finally:
        os.unlink(refpath)
    if out.returncode != 0:
        sys.exit(f"engine failed:\n{out.stdout}\n{out.stderr}")

    m = re.search(r"C engine : ([\d ]+)", out.stdout)
    if not m:
        sys.exit(f"could not parse engine output:\n{out.stdout}")
    gen_ids = [int(t) for t in m.group(1).split()]
    text_out = tok.decode(gen_ids, skip_special_tokens=False)
    # the harness generates a fixed n tokens with no EOS stop; cut at the first
    # end-of-turn marker or PII-mask artifact the model drifts into afterwards
    for stop in ("<|endoftext|>", "<|user|>", "<|assistant|>", "|||IP_ADDRESS|||",
                 "|||EMAIL_ADDRESS|||", "|||PHONE_NUMBER|||"):
        text_out = text_out.split(stop)[0]
    text_out = text_out.strip()

    stats = "\n".join(l for l in (out.stdout + out.stderr).splitlines()
                      if re.match(r"(Speed|PEAK RSS|Expert cache|resident|prefill|om_metal)", l))
    print(text_out)
    print("\n--- engine stats ---")
    print(stats)


if __name__ == "__main__":
    main()
