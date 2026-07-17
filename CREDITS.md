# Credits

overhang is an integration of ideas with clear owners. Nods where they are due:

- **colibrì — JustVugg** (https://github.com/JustVugg/colibri)
  The origin of this project. Expert streaming from disk with an LRU cache,
  the int8/int4 container philosophy, treating VRAM/RAM/storage as one
  hierarchy, and oracle-first token-exact validation are colibrì's designs.
  The olmoe engine here is derived from colibrì's `c/olmoe.c` (Apache-2.0);
  our fixes are offered upstream.
- **Eliseev & Mazur, "Fast Inference of Mixture-of-Experts Language Models
  with Offloading"** (arXiv:2312.17238) — prior art on MoE expert offloading
  and caching.
- **liuliu/example_matmul_metal4** (https://github.com/liuliu/example_matmul_metal4)
  The working matmul2d example that documents the descriptor tile semantics.
- **Rigel — Ramchand Kumaresan** (arXiv:2606.12765) — empirical
  characterization of the Metal 4.1 tensor path on M4; our M5 measurements
  extend this line of work.
- **ggml / llama.cpp** (https://github.com/ggml-org/llama.cpp) — the Q8_0
  quantization conventions and the reference open-source inference culture.
- **Qwen team (Alibaba)** for Qwen3.5/3.6 and the Gated DeltaNet hybrid;
  **Ai2** for OLMoE (fully open weights and data); **Zhipu AI** for GLM.
- **Apple MLX team** — TensorOps groundwork on M5
  (https://machinelearning.apple.com/research/exploring-llms-mlx-m5).
