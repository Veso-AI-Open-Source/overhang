/* om_metal.h — Metal 4 tensor-op backend for olmoe.c prefill (Apple Silicon).
 * Experts stay int8 (colibri container semantics); dequant happens in-kernel
 * (threadgroup tiles) feeding mpp::tensor_ops::matmul2d on the GPU Neural
 * Accelerators. C API so olmoe.c stays pure C. */
#ifndef OM_METAL_H
#define OM_METAL_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* 1 = GPU ready, 0 = unavailable (no device / shader compile failed) */
int omm_init(void);

/* GPU-visible allocation (shared MTLBuffer arena). Falls back to malloc
 * semantics — pointer is normal CPU memory too (unified). */
void *omm_alloc(size_t n);

/* one routed expert applied to a batch of positions */
typedef struct {
    const int8_t *g, *u, *d;        /* int8 weights  [I,D],[I,D],[D,I] (arena ptrs) */
    const float  *gs, *us, *ds;     /* per-row scales (arena ptrs) */
    int mpos;                       /* number of positions routed here */
    const int    *pos;              /* position indices into x/out   [mpos] */
    const float  *w;                /* router weights                [mpos] */
} omm_job;

/* out[pos[r]] += w[r] * SwiGLU_expert(x[pos[r]]) for every job, one command
 * buffer. x,out are f32 [S,D]. Returns 0 on success. */
int omm_moe(const omm_job *jobs, int nj, const float *x, int S, int D, int I, float *out);

/* same, with selectable gate activation: 0 = silu (olmoe/qwen), 1 = gelu_tanh
 * (gemma4). omm_moe() == omm_moe_act(..., 0). */
int omm_moe_act(const omm_job *jobs, int nj, const float *x, int S, int D, int I,
                float *out, int gelu);

/* int8 dense GEMM: out[S,N] = x[S,K] @ dequant(Wq)[N,K]^T. Wq + per-row scales
 * must be arena pointers (omm_alloc). Falls back (-3) when K%64 || N%32. */
int omm_dense_q(const int8_t *Wq, const float *scales, int N, int K,
                const float *x, int S, float *out);

/* dense GEMM: out[S,N] = x[S,K] @ W[N,K]^T. W is the engine's resident f32
 * weight; a transposed half copy is cached GPU-side on first use (keyed by
 * pointer). Returns 0 on success. */
int omm_dense(const float *W, int N, int K, const float *x, int S, float *out);

#ifdef __cplusplus
}
#endif
#endif
