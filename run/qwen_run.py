#!/usr/bin/env python3
"""Run a prompt through the qwen.c streaming engine (Qwen3.6-35B-A3B).

Same trick as olmoe_run.py: synthesize a ref json with prompt_ids + dummy
targets, run the C engine, decode the emitted ids.

Usage:
  ./.venv/bin/python qwen_run.py "prompt" [--n 64] [--model ./models/qwen36_i8]
                                 [--cache 64] [--raw] [--exact]
"""
import argparse, json, os, re, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--model", default=str(HERE / "models/qwen36_i8"))
    ap.add_argument("--cache", type=int, default=64, help="expert cache slots/layer")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--exact", action="store_true", help="IDOT=0 scalar kernels")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))

    if args.raw:
        text = args.prompt
    else:
        text = f"<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = tok.encode(text, add_special_tokens=False).ids

    ref = {"prompt_ids": ids, "full_ids": ids + [0]*args.n}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(ref, f); refpath = f.name

    env = dict(os.environ, SNAP=args.model, QCACHE=str(args.cache))
    if args.exact: env["IDOT"] = "0"
    try:
        out = subprocess.run([str(HERE/"c/qwen"), refpath], env=env,
                             capture_output=True, text=True)
    finally:
        os.unlink(refpath)
    if out.returncode not in (0, 1):
        sys.exit(f"engine failed:\n{out.stdout}\n{out.stderr}")
    m = re.search(r"engine   : ([\d ]+)", out.stdout)
    if not m: sys.exit(f"no output parsed:\n{out.stdout}\n{out.stderr}")
    gen = [int(t) for t in m.group(1).split()]
    text_out = tok.decode(gen, skip_special_tokens=False)
    for stop in ("<|im_end|>", "<|endoftext|>"):
        text_out = text_out.split(stop)[0]
    print(text_out.strip())
    for line in out.stderr.splitlines():
        if re.search(r"streaming|cache|prefill|decode", line): print("·", line)

if __name__ == "__main__":
    main()
