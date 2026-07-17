/* om_metal.mm — implementation. See om_metal.h.
 * Kernel = the validated fusion-harness dq_gemm generalized to runtime M:
 * int8 weight tiles dequant into threadgroup memory, matmul2d (32x32 tile,
 * K-chunk 64, multiply_accumulate) on the M5 Neural Accelerators. */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <dispatch/dispatch.h>
#include "om_metal.h"

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_q;
static id<MTLComputePipelineState> g_psQ, g_psSM, g_psGM, g_psH;

/* ---- arena: 256 MB shared blocks; unified memory = CPU ptr == GPU ptr ---- */
#define BLK (256u<<20)
typedef struct { id<MTLBuffer> buf; size_t used; } Block;
static Block *g_blk; static int g_nblk, g_capblk;

void *omm_alloc(size_t n) {
    n = (n + 63) & ~(size_t)63;
    if (n > BLK) {                                     /* oversized: own buffer */
        id<MTLBuffer> b=[g_dev newBufferWithLength:n options:MTLResourceStorageModeShared];
        if (!b) return NULL;
        if (g_nblk==g_capblk){g_capblk=g_capblk?g_capblk*2:64;g_blk=(Block*)realloc(g_blk,g_capblk*sizeof(Block));}
        g_blk[g_nblk++] = (Block){b, n};
        return b.contents;
    }
    if (!g_nblk || g_blk[g_nblk-1].used + n > BLK) {
        id<MTLBuffer> b=[g_dev newBufferWithLength:BLK options:MTLResourceStorageModeShared];
        if (!b) return NULL;
        if (g_nblk==g_capblk){g_capblk=g_capblk?g_capblk*2:64;g_blk=(Block*)realloc(g_blk,g_capblk*sizeof(Block));}
        g_blk[g_nblk++] = (Block){b, 0};
    }
    Block *bl = &g_blk[g_nblk-1];
    void *p = (char*)bl->buf.contents + bl->used;
    bl->used += n;
    return p;
}
/* resolve an arena pointer to (buffer, offset) for setBuffer */
static bool arena_find(const void *p, id<MTLBuffer> *buf, size_t *off) {
    for (int i = 0; i < g_nblk; i++) {
        char *base = (char*)g_blk[i].buf.contents;
        if ((const char*)p >= base && (const char*)p < base + g_blk[i].buf.length) {
            *buf = g_blk[i].buf; *off = (const char*)p - base; return true;
        }
    }
    return false;
}

static NSString *kSrc = @R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct P { uint K; uint N; uint M; };
/* C[M,N] += X[M,K] x dequant(W)[K,N]; W int8 [N,K] row-major + per-row scales */
kernel void dq_gemm(device const char  *W   [[buffer(0)]],
                    device const float *Wsc [[buffer(1)]],
                    device half        *X   [[buffer(2)]],
                    device half        *C   [[buffer(3)]],
                    constant P         &p   [[buffer(4)]],
                    uint2 tgid [[threadgroup_position_in_grid]],
                    uint  lid  [[thread_index_in_threadgroup]])
{
    const uint K = p.K, N = p.N, M = p.M;
    threadgroup half Bt[64*32];
    auto A  = tensor<device half, dextents<int32_t,2>, tensor_inline>(X, dextents<int32_t,2>((int)K, (int)M));
    auto Ct = tensor<device half, dextents<int32_t,2>, tensor_inline>(C, dextents<int32_t,2>((int)N, (int)M));
    auto Bt_t = tensor<threadgroup half, extents<int32_t,32,64>, tensor_inline>(Bt, extents<int32_t,32,64>());
    constexpr auto d = matmul2d_descriptor(32, 32, 64, false, false, false,
                                           matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d, execution_simdgroups<4>> op;
    const uint n0 = tgid.x * 32, m0 = tgid.y * 32;
    for (uint k0 = 0; k0 < K; k0 += 64) {
        for (uint i = lid; i < 64*32; i += 128) {
            uint kk = i >> 5, nn = i & 31;
            uint gn = n0 + nn;
            Bt[i] = (half)((float)W[(ulong)gn*K + k0 + kk] * Wsc[gn]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = A.slice<64,32>((int)k0, (int)m0);
        auto mC = Ct.slice<32,32>((int)n0, (int)m0);
        op.run(mA, Bt_t, mC);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
/* dense: C[M,N] += X[M,K] x Wt[K,N] (Wt already transposed half, device tensor) */
kernel void h_gemm(device half *Wt [[buffer(0)]],
                   device half *X  [[buffer(1)]],
                   device half *C  [[buffer(2)]],
                   constant P  &p  [[buffer(3)]],
                   uint2 tgid [[threadgroup_position_in_grid]])
{
    const uint K = p.K, N = p.N, M = p.M;
    auto A  = tensor<device half, dextents<int32_t,2>, tensor_inline>(X,  dextents<int32_t,2>((int)K, (int)M));
    auto B  = tensor<device half, dextents<int32_t,2>, tensor_inline>(Wt, dextents<int32_t,2>((int)N, (int)K));
    auto Ct = tensor<device half, dextents<int32_t,2>, tensor_inline>(C,  dextents<int32_t,2>((int)N, (int)M));
    constexpr auto d = matmul2d_descriptor(32, 32, 64, false, false, false,
                                           matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d, execution_simdgroups<4>> op;
    const uint n0 = tgid.x * 32, m0 = tgid.y * 32;
    for (uint k0 = 0; k0 < K; k0 += 64) {
        auto mA = A.slice<64,32>((int)k0, (int)m0);
        auto mB = B.slice<32,64>((int)n0, (int)k0);
        auto mC = Ct.slice<32,32>((int)n0, (int)m0);
        op.run(mA, mB, mC);
    }
}
kernel void silu_mul(device const half *g [[buffer(0)]], device const half *u [[buffer(1)]],
                     device half *h [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    float z = (float)g[i];
    h[i] = (half)(z/(1.f+exp(-z)) * (float)u[i]);
}
kernel void gelu_mul(device const half *g [[buffer(0)]], device const half *u [[buffer(1)]],
                     device half *h [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    float z = (float)g[i];
    float t = 0.5f*z*(1.f + precise::tanh(0.7978845608028654f*(z + 0.044715f*z*z*z)));
    h[i] = (half)(t * (float)u[i]);
}
)MSL";

int omm_init(void) {
    static int done = -1;
    if (done >= 0) return done;
    done = 0;
    g_dev = MTLCreateSystemDefaultDevice();
    if (!g_dev) return 0;
    g_q = [g_dev newCommandQueue];
    NSError *err = nil;
    MTLCompileOptions *opt = [MTLCompileOptions new];
    if (@available(macOS 26.0, *)) opt.languageVersion = MTLLanguageVersion4_0;
    id<MTLLibrary> lib = [g_dev newLibraryWithSource:kSrc options:opt error:&err];
    if (!lib) { fprintf(stderr, "om_metal: shader compile failed: %s\n",
                        err.localizedDescription.UTF8String); return 0; }
    g_psQ  = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dq_gemm"]  error:&err];
    g_psSM = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"silu_mul"] error:&err];
    g_psGM = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gelu_mul"] error:&err];
    g_psH  = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"h_gemm"]   error:&err];
    if (!g_psQ || !g_psSM || !g_psGM || !g_psH) return 0;
    fprintf(stderr, "om_metal: GPU prefill on %s (Metal 4 tensor ops)\n", g_dev.name.UTF8String);
    done = 1;
    return 1;
}

/* grow-only scratch */
static id<MTLBuffer> g_bX, g_bCg, g_bCu, g_bHh, g_bY;
static void ensure(id<MTLBuffer> *b, size_t n) {
    if (*b && (*b).length >= n) return;
    *b = [g_dev newBufferWithLength:n options:MTLResourceStorageModeShared];
}

int omm_moe(const omm_job *jobs, int nj, const float *x, int S, int D, int I, float *out) {
    return omm_moe_act(jobs, nj, x, S, D, I, out, 0);
}

int omm_moe_act(const omm_job *jobs, int nj, const float *x, int S, int D, int I,
                float *out, int gelu) {
    (void)S;
    /* padded row offsets */
    size_t rows = 0;
    size_t *roff = (size_t*)alloca(nj * sizeof(size_t));
    for (int j = 0; j < nj; j++) { roff[j] = rows; rows += (size_t)((jobs[j].mpos + 31) & ~31); }
    if (!rows) return 0;
    ensure(&g_bX,  rows * D * 2); ensure(&g_bCg, rows * I * 2);
    ensure(&g_bCu, rows * I * 2); ensure(&g_bHh, rows * I * 2);
    ensure(&g_bY,  rows * D * 2);
    if (!g_bX || !g_bCg || !g_bCu || !g_bHh || !g_bY) return -1;

    /* gather x rows -> half, zero the padding rows; accumulators start at 0 */
    memset(g_bCg.contents, 0, rows * I * 2);
    memset(g_bCu.contents, 0, rows * I * 2);
    memset(g_bY.contents,  0, rows * D * 2);
    __fp16 *X = (__fp16*)g_bX.contents;
    dispatch_apply((size_t)nj, DISPATCH_APPLY_AUTO, ^(size_t j){
        const omm_job *jb = &jobs[j];
        size_t mp = (size_t)((jb->mpos + 31) & ~31);
        for (int r = 0; r < jb->mpos; r++) {
            const float *src = x + (size_t)jb->pos[r] * D;
            __fp16 *dst = X + (roff[j] + r) * D;
            for (int k = 0; k < D; k++) dst[k] = (__fp16)src[k];
        }
        memset(X + (roff[j] + jb->mpos) * D, 0, (mp - jb->mpos) * D * 2);
    });

    id<MTLCommandBuffer> cb = [g_q commandBuffer];
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
    for (int j = 0; j < nj; j++) {
        const omm_job *jb = &jobs[j];
        uint32_t Mp = (uint32_t)((jb->mpos + 31) & ~31);
        id<MTLBuffer> bw; size_t ow, os;
        uint32_t pGU[3] = {(uint32_t)D, (uint32_t)I, Mp};
        uint32_t pDN[3] = {(uint32_t)I, (uint32_t)D, Mp};
        size_t xo = roff[j] * D * 2, co = roff[j] * I * 2, yo = roff[j] * D * 2;

        [ce setComputePipelineState:g_psQ];
        if (!arena_find(jb->g, &bw, &ow) ) return -2;
        [ce setBuffer:bw offset:ow atIndex:0];
        if (!arena_find(jb->gs, &bw, &os)) return -2;
        [ce setBuffer:bw offset:os atIndex:1];
        [ce setBuffer:g_bX offset:xo atIndex:2];
        [ce setBuffer:g_bCg offset:co atIndex:3];
        [ce setBytes:pGU length:12 atIndex:4];
        [ce dispatchThreadgroups:MTLSizeMake(I/32, Mp/32, 1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];

        if (!arena_find(jb->u, &bw, &ow) ) return -2;
        [ce setBuffer:bw offset:ow atIndex:0];
        if (!arena_find(jb->us, &bw, &os)) return -2;
        [ce setBuffer:bw offset:os atIndex:1];
        [ce setBuffer:g_bCu offset:co atIndex:3];
        [ce dispatchThreadgroups:MTLSizeMake(I/32, Mp/32, 1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];

        [ce setComputePipelineState:(gelu ? g_psGM : g_psSM)];
        [ce setBuffer:g_bCg offset:co atIndex:0];
        [ce setBuffer:g_bCu offset:co atIndex:1];
        [ce setBuffer:g_bHh offset:co atIndex:2];
        [ce dispatchThreads:MTLSizeMake((size_t)Mp*I,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];

        [ce setComputePipelineState:g_psQ];
        if (!arena_find(jb->d, &bw, &ow) ) return -2;
        [ce setBuffer:bw offset:ow atIndex:0];
        if (!arena_find(jb->ds, &bw, &os)) return -2;
        [ce setBuffer:bw offset:os atIndex:1];
        [ce setBuffer:g_bHh offset:co atIndex:2];
        [ce setBuffer:g_bY offset:yo atIndex:3];
        [ce setBytes:pDN length:12 atIndex:4];
        [ce dispatchThreadgroups:MTLSizeMake(D/32, Mp/32, 1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    }
    [ce endEncoding]; [cb commit]; [cb waitUntilCompleted];

    /* scatter-add: out[pos] += w * y  (jobs touch disjoint rows; positions may
     * repeat across jobs -> serialize per position via per-job loop on one thread
     * per output row chunk: simplest correct = single-threaded over jobs) */
    const __fp16 *Y = (const __fp16*)g_bY.contents;
    for (int j = 0; j < nj; j++) {
        const omm_job *jb = &jobs[j];
        for (int r = 0; r < jb->mpos; r++) {
            float wgt = jb->w[r];
            float *o = out + (size_t)jb->pos[r] * D;
            const __fp16 *y = Y + (roff[j] + r) * D;
            for (int k = 0; k < D; k++) o[k] += wgt * (float)y[k];
        }
    }
    return 0;
}

/* ---- dense GEMM: half transposed weight cache keyed by f32 pointer ---- */
typedef struct { const float *key; id<MTLBuffer> buf; } WEnt;
static WEnt *g_w; static int g_nw, g_capw;
static id<MTLBuffer> wcache(const float *W, int N, int K) {
    for (int i = 0; i < g_nw; i++) if (g_w[i].key == W) return g_w[i].buf;
    id<MTLBuffer> b = [g_dev newBufferWithLength:(size_t)N*K*2 options:MTLResourceStorageModeShared];
    if (!b) return nil;
    __fp16 *T = (__fp16*)b.contents;                  /* [K,N] <- W[N,K] */
    dispatch_apply((size_t)K, DISPATCH_APPLY_AUTO, ^(size_t k){
        for (int n = 0; n < N; n++) T[k*(size_t)N + n] = (__fp16)W[(size_t)n*K + k];
    });
    if (g_nw==g_capw){g_capw=g_capw?g_capw*2:128;g_w=(WEnt*)realloc(g_w,g_capw*sizeof(WEnt));}
    g_w[g_nw++] = (WEnt){W, b};
    return b;
}
static id<MTLBuffer> g_bXd, g_bYd;

/* int8 dense: dq_gemm sul peso quantizzato residente (arena) */
int omm_dense_q(const int8_t *Wq, const float *scales, int N, int K,
                const float *x, int S, float *out) {
    if ((K & 63) || (N & 31)) return -3;
    id<MTLBuffer> bw, bs; size_t ow, os;
    if (!arena_find(Wq, &bw, &ow) || !arena_find(scales, &bs, &os)) return -2;
    uint32_t Mp = (uint32_t)((S + 31) & ~31);
    ensure(&g_bXd, (size_t)Mp*K*2); ensure(&g_bYd, (size_t)Mp*N*2);
    if (!g_bXd || !g_bYd) return -1;
    __fp16 *X = (__fp16*)g_bXd.contents;
    for (size_t i = 0; i < (size_t)S*K; i++) X[i] = (__fp16)x[i];
    memset(X + (size_t)S*K, 0, ((size_t)Mp - S)*K*2);
    memset(g_bYd.contents, 0, (size_t)Mp*N*2);
    uint32_t p[3] = {(uint32_t)K, (uint32_t)N, Mp};
    id<MTLCommandBuffer> cb = [g_q commandBuffer];
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
    [ce setComputePipelineState:g_psQ];
    [ce setBuffer:bw offset:ow atIndex:0];
    [ce setBuffer:bs offset:os atIndex:1];
    [ce setBuffer:g_bXd offset:0 atIndex:2];
    [ce setBuffer:g_bYd offset:0 atIndex:3];
    [ce setBytes:p length:12 atIndex:4];
    [ce dispatchThreadgroups:MTLSizeMake(N/32, Mp/32, 1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    [ce endEncoding]; [cb commit]; [cb waitUntilCompleted];
    const __fp16 *Y = (const __fp16*)g_bYd.contents;
    for (size_t i = 0; i < (size_t)S*N; i++) out[i] = (float)Y[i];
    return 0;
}

int omm_dense(const float *W, int N, int K, const float *x, int S, float *out) {
    id<MTLBuffer> bw = wcache(W, N, K);
    if (!bw) return -1;
    uint32_t Mp = (uint32_t)((S + 31) & ~31);
    ensure(&g_bXd, (size_t)Mp*K*2); ensure(&g_bYd, (size_t)Mp*N*2);
    if (!g_bXd || !g_bYd) return -1;
    __fp16 *X = (__fp16*)g_bXd.contents;
    for (size_t i = 0; i < (size_t)S*K; i++) X[i] = (__fp16)x[i];
    memset(X + (size_t)S*K, 0, ((size_t)Mp - S)*K*2);
    memset(g_bYd.contents, 0, (size_t)Mp*N*2);
    uint32_t p[3] = {(uint32_t)K, (uint32_t)N, Mp};
    id<MTLCommandBuffer> cb = [g_q commandBuffer];
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
    [ce setComputePipelineState:g_psH];
    [ce setBuffer:bw offset:0 atIndex:0];
    [ce setBuffer:g_bXd offset:0 atIndex:1];
    [ce setBuffer:g_bYd offset:0 atIndex:2];
    [ce setBytes:p length:12 atIndex:3];
    [ce dispatchThreadgroups:MTLSizeMake(N/32, Mp/32, 1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    [ce endEncoding]; [cb commit]; [cb waitUntilCompleted];
    const __fp16 *Y = (const __fp16*)g_bYd.contents;
    for (size_t i = 0; i < (size_t)S*N; i++) out[i] = (float)Y[i];
    return 0;
}
