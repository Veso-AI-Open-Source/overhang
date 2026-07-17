#!/usr/bin/env python3
"""Convert a Gemma 4 MoE HF checkpoint to the overhang int8 container.

Layout produced (matches engines/gemma.c streaming mode + daemon models.rs):
  - fused HF expert tensors [E,2mi,D]/[E,D,mi] are SPLIT per expert:
      model.layers.N.experts.E.gate_up_proj  int8 [2mi,D] + .qs f32 [2mi]
      model.layers.N.experts.E.down_proj     int8 [D,mi]  + .qs f32 [D]
    (`.experts.<id>.` naming keeps models.rs RAM estimation working)
  - dense 2D matrices (q/k/v/o, mlp gate/up/down, router.proj) -> int8 + .qs
  - embed_tokens, norms, router scales, layer_scalar stay as-is (bf16/f32;
    the engine reads them f32 via st_read_f32)
  - vision/audio towers dropped; multimodal `language_model.` prefix stripped
  - config.json flattened to the text config (model_type gemma4_text);
    tokenizer.json + generation_config.json copied
  - sharded model-XXXXX-of-XXXXX.safetensors + model.safetensors.index.json
    (the index makes the daemon's verify_container completeness gate real)

Usage:
  convert_gemma.py --model <hf snapshot dir> --out ./models/gemma4_26b_i8
"""
import argparse, json, re, shutil, sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SHARD_BYTES = 3_500_000_000  # ~3.5 GB per output shard

DENSE_INT8_RE = re.compile(
    r"\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj|router\.proj)\.weight$"
)


def quantize_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise symmetric int8: scale = amax/127 (mirrors engine quantize_rows)."""
    w = w.float()
    amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scales = amax / 127.0
    q = (w / scales).round().clamp(-128, 127).to(torch.int8)
    return q, scales.squeeze(1).contiguous()


class ShardWriter:
    def __init__(self, out: Path):
        self.out = out
        self.tensors, self.nbytes, self.shards, self.weight_map = {}, 0, [], {}

    def add(self, name: str, t: torch.Tensor):
        self.tensors[name] = t.contiguous()
        self.nbytes += t.numel() * t.element_size()
        if self.nbytes >= SHARD_BYTES:
            self.flush()

    def flush(self):
        if not self.tensors:
            return
        self.shards.append(dict(self.tensors))
        self.tensors, self.nbytes = {}, 0

    def save(self):
        self.flush()
        n = len(self.shards)
        total = 0
        for i, tensors in enumerate(self.shards, 1):
            fname = f"model-{i:05d}-of-{n:05d}.safetensors"
            save_file(tensors, str(self.out / fname))
            for name, t in tensors.items():
                self.weight_map[name] = fname
                total += t.numel() * t.element_size()
            print(f"  wrote {fname} ({sum(t.numel()*t.element_size() for t in tensors.values())/1e9:.2f} GB)")
        (self.out / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": total}, "weight_map": self.weight_map}, indent=0))
        print(f"container total: {total/1e9:.2f} GB in {n} shards")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF snapshot dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out = Path(a.model), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # config: flatten to the text config the engine reads
    cfg = json.loads((src / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    text["model_type"] = "gemma4_text"
    text.pop("dtype", None)
    (out / "config.json").write_text(json.dumps(text, indent=1))
    for extra in ("tokenizer.json", "generation_config.json", "tokenizer_config.json"):
        if (src / extra).exists():
            shutil.copy2(src / extra, out / extra)
    print(f"config (flattened text_config) + tokenizer -> {out}")

    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        sys.exit(f"no safetensors in {src}")

    w = ShardWriter(out)
    n_experts_split = n_dense_q = n_kept = n_dropped = 0

    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for name in f.keys():
                # drop multimodal towers; strip the language_model prefix
                if any(s in name for s in ("vision", "audio", "multi_modal", "multimodal")):
                    n_dropped += 1
                    continue
                cname = name
                if "language_model." in cname:
                    cname = "model." + cname.split("language_model.", 1)[1]
                if not cname.startswith("model.") and cname != "lm_head.weight":
                    cname = "model." + cname
                if cname == "lm_head.weight":  # tied to embed in gemma4
                    n_dropped += 1
                    continue

                t = f.get_tensor(name)
                if cname.endswith("experts.gate_up_proj") or cname.endswith("experts.down_proj"):
                    base = cname.rsplit(".", 1)  # ("model.layers.N.experts", proj)
                    prefix, proj = cname[: cname.rfind(".experts.")], cname.rsplit(".", 1)[1]
                    for e in range(t.shape[0]):
                        q, s = quantize_rows(t[e])
                        w.add(f"{prefix}.experts.{e}.{proj}", q)
                        w.add(f"{prefix}.experts.{e}.{proj}.qs", s)
                    n_experts_split += t.shape[0]
                    del base
                elif DENSE_INT8_RE.search(cname):
                    q, s = quantize_rows(t)
                    w.add(cname, q)
                    w.add(cname + ".qs", s)
                    n_dense_q += 1
                else:
                    w.add(cname, t)
                    n_kept += 1
        print(f"{shard.name}: done")

    w.save()
    print(f"experts split: {n_experts_split}, dense int8: {n_dense_q}, "
          f"kept as-is: {n_kept}, dropped: {n_dropped}")
    print(f"\nRun: SNAP={out} ./gemma --serve")


if __name__ == "__main__":
    main()
