# Roadmap — claimable work

Each item is bounded and has a measurement that defines done. The oracle
harness (`make check-oracle`) is the correctness gate for all of them.
Claim by opening an issue; first substantial landed PR gets write access.

| item | definition of done | est. effort |
|---|---|---|
| **int4 packed expert container** | Qwen3.6-35B experts 32 GB -> 16 GB; sustained decode >= 4 tok/s on 24 GB; oracle green at int4 tolerance | ~2 days |
| **MTP speculative decoding** | Qwen3.6 checkpoint ships a 1-layer MTP head (`mtp.*` tensors); wire draft+verify; report acceptance rate and net tok/s | ~3 days |
| **GPU prefill for the qwen engine** | metal/om_metal.mm already does this for olmoe (8x measured); port the batch-union path to qwen.c prefill | ~1 day |
| **Qwen3.5-122B-A10B run** | same engine, bigger container (~61 GB int8). First 122B on a consumer Mac: post your numbers | download + patience |
| **Router-lookahead prefetch** | colibrì's PILOT idea: prefetch next layer's experts during compute; measure hit-rate delta | ~2 days |
| **Kernel lane extraction** | carve qmm.h (quantized matvec) and expertstream.{h,c} (cache+container) out of the engines so the next architecture port is an afternoon | ~1 day |
| **Linux port** | engines are POSIX; needs Makefile branch + AVX2 path check (kernels have scalar fallback) | ~1 day |
| **New architecture: your pick** | write the oracle (tools/make_qwen_oracle.py is the template, ~60 lines), then the engine math. GLM-4.5-Air and Kimi-Linear are natural fits | ~2-4 days |

Hardware benchmark rows wanted for the README table: M5 Pro/Max, M4
generation, 32-64 GB configurations. Open an issue with your numbers.
