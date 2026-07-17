#!/usr/bin/env python3
"""Tiny-random Qwen3.6 (qwen3_5_moe text) oracle for the C engine.

Same methodology as make_glm_oracle.py: build the REAL architecture at toy
dimensions with seeded random weights, save the snapshot, and record greedy +
teacher-forcing reference token ids in ref_qwen.json. The C engine must
reproduce these exactly (TF) / greedily before touching the 35B.

Usage: make_qwen_oracle.py [--out c/qwen_tiny] [--ref ref_qwen.json]
"""
import argparse, json
from pathlib import Path

import torch
from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM

torch.manual_seed(42)

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="c/qwen_tiny")
ap.add_argument("--ref", default="ref_qwen.json")
a = ap.parse_args()

cfg = Qwen3_5MoeTextConfig(
    vocab_size=512,
    hidden_size=64,
    num_hidden_layers=8,                  # pattern: full attention every 4th
    full_attention_interval=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    partial_rotary_factor=0.25,           # rotary dim 8
    rope_parameters={"rope_type": "default", "rope_theta": 10000000,
                     "partial_rotary_factor": 0.25,
                     "mrope_interleaved": True, "mrope_section": [2, 1, 1]},
    attn_output_gate=True,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    intermediate_size=32,
    rms_norm_eps=1e-6,
    max_position_embeddings=4096,
    tie_word_embeddings=False,
    mtp_num_hidden_layers=0,
)

model = Qwen3_5MoeForCausalLM(cfg).eval().float()
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

ref = {"prompt_ids": prompt, "full_ids": full, "tf_argmax": tf,
       "note": "tiny-random qwen3_5_moe; greedy 20 + TF argmax over full_ids"}
Path(a.ref).write_text(json.dumps(ref, indent=1))
print(f"ref -> {a.ref}")
print("greedy continuation:", full[len(prompt):])
