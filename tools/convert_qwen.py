#!/usr/bin/env python3
"""Convert Qwen3.6/Qwen3.5-MoE HF checkpoint to the colibri int8 container.

Routed expert weights -> row-wise int8 + f32 scales ("name" + "name.qs").
Everything else (dense, deltanet, shared expert, norms, embed, lm_head) stays
bf16 as-is — the engine reads bf16 -> f32 on load. Shard-by-shard, resumable.

Usage:
  convert_qwen.py --model <hf snapshot dir> --out ./models/qwen36_i8
"""
import argparse, json, re, shutil, sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# 35B multimodal checkpoint: fused 3D experts under model.language_model.*
FUSED_RE = re.compile(r"(model\.(?:language_model\.)?layers\.\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")
EXPERT_RE = re.compile(r"model\.(?:language_model\.)?layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$")
def outname(name):                       # engine expects model.layers.*
    return name.replace("model.language_model.", "model.")

def quantize_row(w):
    w = w.float()
    s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
    q = (w / s).round().clamp(-128, 127).to(torch.int8)
    return q, s.squeeze(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out = Path(a.model), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for aux in src.glob("*.json"):
        shutil.copy2(aux, out / aux.name)
    for aux in ("tokenizer.json", "merges.txt", "vocab.json"):
        if (src / aux).exists(): shutil.copy2(src / aux, out / aux)

    shards = sorted(src.glob("*.safetensors"))
    if not shards: sys.exit(f"no safetensors in {src}")
    tot_in = tot_out = 0
    for i, sh in enumerate(shards, 1):
        dst = out / sh.name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[{i}/{len(shards)}] {sh.name} already converted, skip"); continue
        t = load_file(str(sh))
        o = {}
        for name, w in t.items():
            if name.startswith("model.visual"):        # vision tower: text engine skips it
                continue
            fm = FUSED_RE.search(name)
            if fm:                                     # [E, 2I|D, D|I] -> per-expert int8
                base, kind = outname(fm.group(1)), fm.group(2)
                E = w.shape[0]
                for e in range(E):
                    if kind == "gate_up_proj":
                        I2 = w.shape[1] // 2
                        for sub, ww in (("gate_proj", w[e,:I2,:]), ("up_proj", w[e,I2:,:])):
                            q, s = quantize_row(ww)
                            o[f"{base}.mlp.experts.{e}.{sub}.weight"] = q
                            o[f"{base}.mlp.experts.{e}.{sub}.weight.qs"] = s.float()
                    else:
                        q, s = quantize_row(w[e])
                        o[f"{base}.mlp.experts.{e}.down_proj.weight"] = q
                        o[f"{base}.mlp.experts.{e}.down_proj.weight.qs"] = s.float()
                tot_in += w.numel()*w.element_size(); tot_out += w.numel()
            elif EXPERT_RE.search(name):               # already per-expert (tiny model)
                q, s = quantize_row(w)
                o[outname(name)] = q; o[outname(name) + ".qs"] = s.float()
                tot_in += w.numel()*w.element_size(); tot_out += q.numel() + s.numel()*4
            else:
                o[outname(name)] = w
        save_file(o, str(dst))
        print(f"[{i}/{len(shards)}] {sh.name} ok", flush=True)
    if tot_in: print(f"experts: {tot_in/1e9:.1f} GB -> {tot_out/1e9:.1f} GB")
    print(f"container ready: {out}")

if __name__ == "__main__":
    main()
