#!/usr/bin/env python3
"""Tiny-random Gemma 4 MoE (gemma4_text) oracle for the C engine.

Same methodology as make_qwen_oracle.py: build the REAL architecture at toy
dimensions with seeded random weights, save the snapshot, and record greedy +
teacher-forcing reference token ids in ref_gemma.json. The C engine must
reproduce these exactly (TF) / greedily before touching the 26B-A4B.

Mirrors google/gemma-4-26B-A4B config.json structurally:
  - layer_types: 5x sliding_attention then 1x full_attention (interval 6)
  - per-layer-type rope (full: proportional + partial_rotary 0.25; sliding:
    default theta 10k), sliding_window small enough that the 28-token
    reference actually exercises the window mask
  - attention_k_eq_v (K/V shared), separate global kv heads/dim on full layers
  - MoE block: routed experts + top_k, gelu_pytorch_tanh
  - final_logit_softcapping, tie_word_embeddings

Usage: make_gemma_oracle.py [--out oracle/gemma_tiny] [--ref oracle/ref_gemma.json]
"""
import argparse, json
from pathlib import Path

import torch
from transformers.models.gemma4 import Gemma4TextConfig, Gemma4ForCausalLM

torch.manual_seed(42)

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="oracle/gemma_tiny")
ap.add_argument("--ref", default="oracle/ref_gemma.json")
a = ap.parse_args()

LAYER_TYPES = (["sliding_attention"] * 5 + ["full_attention"]) * 2  # 12 layers

cfg = Gemma4TextConfig(
    vocab_size=512,
    hidden_size=64,
    intermediate_size=48,
    num_hidden_layers=len(LAYER_TYPES),
    layer_types=LAYER_TYPES,
    sliding_window=8,                      # < ref length: window mask exercised
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    num_global_key_value_heads=1,
    global_head_dim=32,
    attention_k_eq_v=True,
    rope_parameters={
        "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0,
                           "partial_rotary_factor": 0.25},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
    },
    enable_moe_block=True,
    num_experts=8,
    top_k_experts=2,
    moe_intermediate_size=16,
    hidden_activation="gelu_pytorch_tanh",
    final_logit_softcapping=30.0,
    rms_norm_eps=1e-6,
    max_position_embeddings=4096,
    tie_word_embeddings=True,
    hidden_size_per_layer_input=0,         # per-layer input embeddings off (as 26B-A4B)
    use_bidirectional_attention=None,      # text-only oracle: no vision blocks
    pad_token_id=0,
)

model = Gemma4ForCausalLM(cfg).eval().float()
print(f"tiny model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
model.save_pretrained(out, safe_serialization=True)
print(f"snapshot -> {out}")

prompt = [3, 1, 4, 1, 5, 9, 2, 6]
n_new = 20
with torch.no_grad():
    ids = torch.tensor([prompt])
    gen = model.generate(ids, max_new_tokens=n_new, do_sample=False,
                         use_cache=True, pad_token_id=0)
    full = gen[0].tolist()
    # teacher-forcing: argmax at every position of the full sequence
    logits = model(torch.tensor([full])).logits[0]
    tf = logits.argmax(-1).tolist()

# reload from disk: the saved snapshot (what the C engine will read) must
# reproduce the same TF argmax, proving the safetensors round-trip
with torch.no_grad():
    m2 = Gemma4ForCausalLM.from_pretrained(out).eval().float()
    tf2 = m2(torch.tensor([full])).logits[0].argmax(-1).tolist()
assert tf2 == tf, "reloaded snapshot diverges from in-memory model"
print("reload check: TF argmax identical from saved snapshot")

ref = {"prompt_ids": prompt, "full_ids": full, "tf_argmax": tf,
       "note": "tiny-random gemma4_text MoE; greedy 20 + TF argmax over full_ids"}
Path(a.ref).write_text(json.dumps(ref, indent=1))
print(f"ref -> {a.ref}")
print("greedy continuation:", full[len(prompt):])
