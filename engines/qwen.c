/* qwen.c — Qwen3.6 / Qwen3.5-MoE (qwen3_5_moe text) engine in pure C.
 *
 * Stadio A (this file): faithful replica of the transformers forward
 * (modeling_qwen3_5_moe.py, torch fallback path) validated token-exact
 * against ref_qwen.json from c/tools/make_qwen_oracle.py.
 *
 * Architecture: 3:1 Gated DeltaNet (linear attention, recurrent state) :
 * gated full attention (GQA, partial interleaved RoPE, sigmoid output gate);
 * MoE with softmax->top-k(renorm) router + sigmoid-gated shared expert.
 *
 * This stage keeps ALL weights f32-resident (tiny model). Expert streaming +
 * int8 cache (olmoe.c-style) is Stadio B once exactness is proven.
 *
 * build: make qwen    run: SNAP=c/qwen_tiny ./qwen ref_qwen.json
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>
#include "st.h"

/* ---------- config (read from config.json, nested text_config or flat) ---------- */
typedef struct {
    int hidden, n_layers, full_interval;
    int n_heads, n_kv, head_dim, rot_dim; float theta;
    int lin_kh, lin_vh, lin_kd, lin_vd, conv_k;
    int n_exp, topk, moe_inter, shared_inter;
    int vocab; float eps;
} Cfg;

typedef struct { int8_t *q; float *s; } QW;   /* int8 dense (stream mode) */

typedef struct {           /* dense per-layer weights */
    int is_full;           /* 1 = full attention, 0 = gated deltanet */
    float *in_ln, *post_ln;
    /* full attention */
    float *q, *k, *v, *o, *qn, *kn;
    /* deltanet */
    float *qkv, *z, *b, *a, *conv_w, *A_log, *dt_bias, *gnorm, *outp;
    /* moe */
    float *gate, *sh_g, *sh_u, *sh_d, *sh_gate;
    float **eg, **eu, **ed;               /* [n_exp] expert weights f32 (tiny) */
    /* int8 twins (stream mode): big matmuls go through matmul_q */
    QW Qq,Qk,Qv,Qo, Qqkv,Qz,Qout, Qgate,Qsg,Qsu,Qsd;
} Layer;

typedef struct { int eid; int8_t *g,*u,*d; float *gs,*us,*ds; uint64_t used; } Slot;
typedef struct { Slot *slots; int n, cap; } LCache;

typedef struct {
    Cfg c; shards S;
    int stream;                           /* 1 = int8 container, experts via LRU */
    LCache *cache; uint64_t clock, hits, miss;
    float *embed, *lm_head, *final_norm;
    Layer *L;
    /* per full-attn layer KV cache */
    float **K, **V; int max_t;
    QW Qlmh;
    /* per deltanet layer: conv history [conv_dim, conv_k-1] and state [vh][kd][vd] */
    float **conv_st, **rec_st;
    int pos;                              /* absolute position of next token */
} Model;

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static float *falloc(int64_t n){ float *p=calloc(n,sizeof(float)); if(!p){fprintf(stderr,"OOM %lld\n",(long long)n);exit(1);} return p; }

static void matvec(float *y, const float *x, const float *W, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o*I; float acc = 0.f;
        for (int i = 0; i < I; i++) acc += x[i]*w[i];
        y[o] = acc;
    }
}
/* Qwen3_5MoeRMSNorm is ZERO-CENTERED: gamma = 1 + weight (weight stored ~0) */
#if defined(__ARM_NEON)
#include <arm_neon.h>
static inline int32_t dot_i8_16(const int8_t *a, const int8_t *b) {
    int32x4_t acc = vdupq_n_s32(0);
    int8x16_t va = vld1q_s8(a), vb = vld1q_s8(b);
#if defined(__ARM_FEATURE_DOTPROD)
    acc = vdotq_s32(acc, va, vb);
#else
    acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(va),  vget_low_s8(vb)));
    acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(va), vget_high_s8(vb)));
#endif
    return vaddvq_s32(acc);
}
#endif
/* y[O] = x[I] @ Wq^T, Wq int8 row-major + per-row scale (Q8_0 activations, IDOT=0 scalar) */
static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
#if defined(__ARM_NEON)
    static int idot = -1;
    if (idot < 0) { const char *e = getenv("IDOT"); idot = !(e && *e == '0'); }
    if (idot && I % 16 == 0 && I <= 4096) {
        int nb = I/16; int8_t xi[4096]; float xs[256];
        for (int b = 0; b < nb; b++) {
            const float *xb = x + b*16;
            float am = 0.f; for (int i = 0; i < 16; i++) { float a = fabsf(xb[i]); if (a > am) am = a; }
            float s = am/127.f; if (s < 1e-12f) s = 1e-12f;
            xs[b] = s; float inv = 1.f/s;
            for (int i = 0; i < 16; i++) xi[b*16+i] = (int8_t)lrintf(xb[i]*inv);
        }
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o*I; float acc = 0.f;
            for (int b = 0; b < nb; b++) acc += xs[b]*(float)dot_i8_16(xi+b*16, w+b*16);
            y[o] = acc*scale[o];
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o*I; float acc = 0.f;
        for (int i = 0; i < I; i++) acc += x[i]*(float)w[i];
        y[o] = acc*scale[o];
    }
}

static void quantize_rows(const float *w, int8_t *q, float *scale, int O, int I) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *wr = w + (int64_t)o*I;
        float am = 0.f; for (int i = 0; i < I; i++) { float a = fabsf(wr[i]); if (a > am) am = a; }
        float s = am/127.f; if (s < 1e-8f) s = 1e-8f;
        scale[o] = s;
        int8_t *qr = q + (int64_t)o*I;
        for (int i = 0; i < I; i++) {
            int v = (int)lrintf(wr[i]/s);
            if (v > 127) v = 127; if (v < -128) v = -128;
            qr[i] = (int8_t)v;
        }
    }
}
#define DM(m,y,x,W,QWF,I,O) do { \
    if ((m)->stream) matmul_q((y),(x),(QWF).q,(QWF).s,(I),(O)); \
    else matvec((y),(x),(W),(I),(O)); } while (0)
static QW qw_make(float **pw, int O, int I) {   /* quantize + free the f32 original */
    QW w; w.q = malloc((int64_t)O*I); w.s = malloc(O*sizeof(float));
    quantize_rows(*pw, w.q, w.s, O, I);
    free(*pw); *pw = NULL;
    return w;
}
static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    double ms = 0; for (int i = 0; i < n; i++) ms += (double)x[i]*x[i];
    float r = 1.f/sqrtf((float)(ms/n)+eps);
    for (int i = 0; i < n; i++) out[i] = x[i]*r*(1.f+w[i]);
}
static void l2norm_row(float *x, int n) {                 /* eps 1e-6, matches l2norm() */
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i]*x[i];
    float r = 1.f/sqrtf((float)ss + 1e-6f);
    for (int i = 0; i < n; i++) x[i] *= r;
}
static float sigmoidf_(float z){ return 1.f/(1.f+expf(-z)); }
static float siluf_(float z){ return z*sigmoidf_(z); }
static float softplusf_(float z){ return z>20.f ? z : log1pf(expf(z)); }
static void softmax_(float *x, int n) {
    float m=-1e30f; for(int i=0;i<n;i++) if(x[i]>m)m=x[i];
    double s=0; for(int i=0;i<n;i++){ x[i]=expf(x[i]-m); s+=x[i]; }
    for(int i=0;i<n;i++) x[i]/= (float)s;
}

/* ---------- loading ---------- */
#include "json.h"
static jval *cfg_root;
static double cget(jval *r, const char *k, double dflt) {
    jval *v = json_get(r, k); return v ? v->num : dflt;
}
static void load_cfg(Cfg *c, const char *snap) {
    char path[2048]; snprintf(path,sizeof(path),"%s/config.json",snap);
    FILE *f=fopen(path,"rb"); if(!f){perror(path);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf=malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    char *arena=NULL; jval *r=json_parse(buf,&arena);
    jval *t=json_get(r,"text_config"); if(t) r=t;      /* 35B nests under text_config */
    cfg_root=r;
    c->hidden=(int)cget(r,"hidden_size",0); c->n_layers=(int)cget(r,"num_hidden_layers",0);
    c->full_interval=(int)cget(r,"full_attention_interval",4);
    c->n_heads=(int)cget(r,"num_attention_heads",0); c->n_kv=(int)cget(r,"num_key_value_heads",0);
    c->head_dim=(int)cget(r,"head_dim",0);
    jval *rp=json_get(r,"rope_parameters");
    double prf = rp? cget(rp,"partial_rotary_factor",cget(r,"partial_rotary_factor",1.0))
                   : cget(r,"partial_rotary_factor",1.0);
    c->rot_dim=(int)(c->head_dim*prf);
    c->theta = rp? (float)cget(rp,"rope_theta",10000000.0) : 10000000.f;
    c->lin_kh=(int)cget(r,"linear_num_key_heads",0); c->lin_vh=(int)cget(r,"linear_num_value_heads",0);
    c->lin_kd=(int)cget(r,"linear_key_head_dim",0);  c->lin_vd=(int)cget(r,"linear_value_head_dim",0);
    c->conv_k=(int)cget(r,"linear_conv_kernel_dim",4);
    c->n_exp=(int)cget(r,"num_experts",0); c->topk=(int)cget(r,"num_experts_per_tok",0);
    c->moe_inter=(int)cget(r,"moe_intermediate_size",0);
    c->shared_inter=(int)cget(r,"shared_expert_intermediate_size",0);
    c->vocab=(int)cget(r,"vocab_size",0); c->eps=(float)cget(r,"rms_norm_eps",1e-6);
}
static float *ld(Model *m, const char *fmt, ...) {
    char name[512]; va_list ap; va_start(ap,fmt); vsnprintf(name,sizeof(name),fmt,ap); va_end(ap);
    int64_t n = st_numel(&m->S, name);
    if (n < 0) { fprintf(stderr,"missing tensor %s\n",name); exit(1); }
    float *p = falloc(n); st_read_f32(&m->S, name, p, 0); return p;
}
static float *ld_opt(Model *m, const char *fmt, ...) {
    char name[512]; va_list ap; va_start(ap,fmt); vsnprintf(name,sizeof(name),fmt,ap); va_end(ap);
    int64_t n = st_numel(&m->S, name);
    if (n < 0) return NULL;
    float *p = falloc(n); st_read_f32(&m->S, name, p, 0); return p;
}

static void model_init(Model *m, const char *snap, int max_t) {
    load_cfg(&m->c, snap); st_init(&m->S, snap);
    Cfg *c = &m->c;
    m->max_t = max_t; m->pos = 0;
    {   /* container? expert 0/0 stored as int8 => streaming with LRU cache */
        st_tensor *t = st_find(&m->S, "model.layers.0.mlp.experts.0.gate_proj.weight");
        m->stream = (t && t->dtype == 3);
    }
    if (m->stream) {
        int cap = getenv("QCACHE") ? atoi(getenv("QCACHE")) : 64;
        m->cache = calloc(c->n_layers, sizeof(LCache));
        for (int i = 0; i < c->n_layers; i++) {
            m->cache[i].cap = cap;
            m->cache[i].slots = calloc(cap, sizeof(Slot));
        }
        fprintf(stderr, "streaming experts: int8 container, cache %d/layer (%.1f GB)\n",
                cap, (double)c->n_layers*cap*(3.0*c->moe_inter*c->hidden+ (2.0*c->moe_inter+c->hidden)*4)/1e9);
    }
    m->embed = ld(m, "model.embed_tokens.weight");
    m->lm_head = ld_opt(m, "lm_head.weight");
    if (!m->lm_head) m->lm_head = m->embed;               /* tied */
    if (m->stream) { float *lm = m->lm_head; m->Qlmh = qw_make(&lm, c->vocab, c->hidden);
                     if (m->lm_head != m->embed) m->lm_head = NULL; }
    m->final_norm = ld(m, "model.norm.weight");
    m->L = calloc(c->n_layers, sizeof(Layer));
    m->K = calloc(c->n_layers, sizeof(float*)); m->V = calloc(c->n_layers, sizeof(float*));
    m->conv_st = calloc(c->n_layers, sizeof(float*)); m->rec_st = calloc(c->n_layers, sizeof(float*));
    int conv_dim = 2*c->lin_kh*c->lin_kd + c->lin_vh*c->lin_vd;
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        l->is_full = ((i+1) % c->full_interval) == 0;
        l->in_ln  = ld(m,"model.layers.%d.input_layernorm.weight",i);
        l->post_ln= ld(m,"model.layers.%d.post_attention_layernorm.weight",i);
        if (l->is_full) {
            l->q = ld(m,"model.layers.%d.self_attn.q_proj.weight",i);
            l->k = ld(m,"model.layers.%d.self_attn.k_proj.weight",i);
            l->v = ld(m,"model.layers.%d.self_attn.v_proj.weight",i);
            l->o = ld(m,"model.layers.%d.self_attn.o_proj.weight",i);
            l->qn= ld(m,"model.layers.%d.self_attn.q_norm.weight",i);
            l->kn= ld(m,"model.layers.%d.self_attn.k_norm.weight",i);
            if (m->stream) {
                l->Qq = qw_make(&l->q, c->n_heads*c->head_dim*2, c->hidden);
                l->Qk = qw_make(&l->k, c->n_kv*c->head_dim, c->hidden);
                l->Qv = qw_make(&l->v, c->n_kv*c->head_dim, c->hidden);
                l->Qo = qw_make(&l->o, c->hidden, c->n_heads*c->head_dim);
            }
            m->K[i]=falloc((int64_t)c->n_kv*max_t*c->head_dim);
            m->V[i]=falloc((int64_t)c->n_kv*max_t*c->head_dim);
        } else {
            l->qkv   = ld(m,"model.layers.%d.linear_attn.in_proj_qkv.weight",i);
            l->z     = ld(m,"model.layers.%d.linear_attn.in_proj_z.weight",i);
            l->b     = ld(m,"model.layers.%d.linear_attn.in_proj_b.weight",i);
            l->a     = ld(m,"model.layers.%d.linear_attn.in_proj_a.weight",i);
            l->conv_w= ld(m,"model.layers.%d.linear_attn.conv1d.weight",i);   /* [conv_dim,1,k] */
            l->A_log = ld(m,"model.layers.%d.linear_attn.A_log",i);
            l->dt_bias=ld(m,"model.layers.%d.linear_attn.dt_bias",i);
            l->gnorm = ld(m,"model.layers.%d.linear_attn.norm.weight",i);     /* [vd] */
            l->outp  = ld(m,"model.layers.%d.linear_attn.out_proj.weight",i);
            if (m->stream) {
                int cd = 2*c->lin_kh*c->lin_kd + c->lin_vh*c->lin_vd, vdm = c->lin_vh*c->lin_vd;
                l->Qqkv = qw_make(&l->qkv, cd, c->hidden);
                l->Qz   = qw_make(&l->z, vdm, c->hidden);
                l->Qout = qw_make(&l->outp, c->hidden, vdm);
            }
            m->conv_st[i]=falloc((int64_t)conv_dim*(c->conv_k-1));
            m->rec_st[i]=falloc((int64_t)c->lin_vh*c->lin_kd*c->lin_vd);
        }
        l->gate  = ld(m,"model.layers.%d.mlp.gate.weight",i);
        l->sh_g  = ld(m,"model.layers.%d.mlp.shared_expert.gate_proj.weight",i);
        l->sh_u  = ld(m,"model.layers.%d.mlp.shared_expert.up_proj.weight",i);
        l->sh_d  = ld(m,"model.layers.%d.mlp.shared_expert.down_proj.weight",i);
        l->sh_gate=ld(m,"model.layers.%d.mlp.shared_expert_gate.weight",i);
        if (m->stream) {
            l->Qgate = qw_make(&l->gate, c->n_exp, c->hidden);
            l->Qsg = qw_make(&l->sh_g, c->shared_inter, c->hidden);
            l->Qsu = qw_make(&l->sh_u, c->shared_inter, c->hidden);
            l->Qsd = qw_make(&l->sh_d, c->hidden, c->shared_inter);
        }
        if (!m->stream) {
            l->eg=calloc(c->n_exp,sizeof(float*)); l->eu=calloc(c->n_exp,sizeof(float*)); l->ed=calloc(c->n_exp,sizeof(float*));
            for (int e = 0; e < c->n_exp; e++) {
                l->eg[e]=ld(m,"model.layers.%d.mlp.experts.%d.gate_proj.weight",i,e);
                l->eu[e]=ld(m,"model.layers.%d.mlp.experts.%d.up_proj.weight",i,e);
                l->ed[e]=ld(m,"model.layers.%d.mlp.experts.%d.down_proj.weight",i,e);
            }
        }
    }
}

/* ---------- gated deltanet, one token (torch_recurrent_gated_delta_rule) ---------- */
static void deltanet_step(Model *m, Layer *l, int li, const float *x, float *out) {
    Cfg *c = &m->c;
    int kh=c->lin_kh, vh=c->lin_vh, kd=c->lin_kd, vd=c->lin_vd, K=c->conv_k;
    int key_dim = kh*kd, val_dim = vh*vd, conv_dim = 2*key_dim + val_dim;
    float *qkv = falloc(conv_dim);
    DM(m, qkv, x, l->qkv, l->Qqkv, c->hidden, conv_dim);
    /* causal conv1d update (kernel K, groups=conv_dim, no bias) + silu */
    float *hist = m->conv_st[li];                          /* [conv_dim, K-1] oldest..newest */
    float *cv = falloc(conv_dim);
    #pragma omp parallel for schedule(static)
    for (int ch = 0; ch < conv_dim; ch++) {
        const float *w = l->conv_w + (int64_t)ch*K;        /* [K] */
        float *h = hist + (int64_t)ch*(K-1);
        float acc = 0.f;
        for (int j = 0; j < K-1; j++) acc += w[j]*h[j];
        acc += w[K-1]*qkv[ch];
        for (int j = 0; j < K-2; j++) h[j]=h[j+1];         /* shift history */
        h[K-2]=qkv[ch];
        cv[ch] = siluf_(acc);
    }
    float *q = cv, *k = cv + key_dim, *v = cv + 2*key_dim;
    /* z, beta, g */
    float *z = falloc(val_dim);   DM(m, z, x, l->z, l->Qz, c->hidden, val_dim);
    float *bb = falloc(vh);       matvec(bb, x, l->b, c->hidden, vh);
    float *aa = falloc(vh);       matvec(aa, x, l->a, c->hidden, vh);
    /* l2norm q,k per head + scale q */
    for (int h = 0; h < kh; h++) { l2norm_row(q+h*kd, kd); l2norm_row(k+h*kd, kd); }
    float scale = 1.f/sqrtf((float)kd);
    for (int i = 0; i < key_dim; i++) q[i]*=scale;
    /* per v-head recurrence; k/q head index = h / (vh/kh) (repeat_interleave) */
    int rep = vh/kh;
    float *S = m->rec_st[li];                              /* [vh][kd][vd] */
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < vh; h++) {
        const float *qh = q + (h/rep)*kd, *kt = k + (h/rep)*kd;
        const float *vt = v + h*vd;
        float g = -expf(l->A_log[h]) * softplusf_(aa[h] + l->dt_bias[h]);
        float ge = expf(g), beta = sigmoidf_(bb[h]);
        float *Sh = S + (int64_t)h*kd*vd;
        float kv[512], delta[512];                         /* vd <= 512 */
        for (int j = 0; j < vd; j++) kv[j]=0.f;
        for (int i = 0; i < kd; i++) {
            float *Si = Sh + (int64_t)i*vd; float ki = kt[i];
            for (int j = 0; j < vd; j++) { Si[j]*=ge; kv[j]+=Si[j]*ki; }
        }
        for (int j = 0; j < vd; j++) delta[j]=(vt[j]-kv[j])*beta;
        float *oh = out + h*vd;                            /* reuse out as core_attn buffer */
        for (int j = 0; j < vd; j++) oh[j]=0.f;
        for (int i = 0; i < kd; i++) {
            float *Si = Sh + (int64_t)i*vd; float ki=kt[i], qi=qh[i];
            for (int j = 0; j < vd; j++) { Si[j]+=ki*delta[j]; oh[j]+=Si[j]*qi; }
        }
        /* gated RMSNorm per head: rms(out)*w * silu(z) */
        double ms=0; for(int j=0;j<vd;j++) ms+=(double)oh[j]*oh[j];
        float r=1.f/sqrtf((float)(ms/vd)+c->eps);
        const float *zh = z + h*vd;
        for (int j = 0; j < vd; j++) oh[j]=oh[j]*r*l->gnorm[j]*siluf_(zh[j]);
    }
    /* out_proj in place: out currently [val_dim] -> project to hidden */
    float *proj = falloc(c->hidden);
    DM(m, proj, out, l->outp, l->Qout, val_dim, c->hidden);
    memcpy(out, proj, c->hidden*sizeof(float));
    free(proj); free(qkv); free(cv); free(z); free(bb); free(aa);
}

/* ---------- gated full attention, one token ---------- */
static void attn_step(Model *m, Layer *l, int li, const float *x, int pos, float *out) {
    Cfg *c = &m->c;
    int H=c->n_heads, KV=c->n_kv, hd=c->head_dim, rot=c->rot_dim, grp=H/KV;
    float *qg = falloc((int64_t)H*hd*2);
    DM(m, qg, x, l->q, l->Qq, c->hidden, H*hd*2);
    float *kk = falloc((int64_t)KV*hd); DM(m, kk, x, l->k, l->Qk, c->hidden, KV*hd);
    float *vv = falloc((int64_t)KV*hd); DM(m, vv, x, l->v, l->Qv, c->hidden, KV*hd);
    /* per-head split: q_proj row-major gives per head [hd q | hd gate] (chunk on last dim of
     * view [.., H, 2*hd]) */
    float *q = falloc((int64_t)H*hd), *gate = falloc((int64_t)H*hd);
    for (int h = 0; h < H; h++) {
        memcpy(q   +h*hd, qg + (int64_t)h*2*hd,      hd*sizeof(float));
        memcpy(gate+h*hd, qg + (int64_t)h*2*hd + hd, hd*sizeof(float));
    }
    for (int h = 0; h < H; h++)  rmsnorm(q+h*hd, q+h*hd, l->qn, hd, c->eps);
    for (int h = 0; h < KV; h++) rmsnorm(kk+h*hd, kk+h*hd, l->kn, hd, c->eps);
    /* partial RoPE (non-interleaved halves within rot dims; text mrope == default) */
    int half = rot/2;
    for (int h = 0; h < H+KV; h++) {
        float *vec = h < H ? q+h*hd : kk+(h-H)*hd;
        for (int j = 0; j < half; j++) {
            float inv = powf(c->theta, -(float)(2*j)/(float)rot);
            float ang = pos*inv, cs=cosf(ang), sn=sinf(ang);
            float a0=vec[j], b0=vec[j+half];
            vec[j]      = a0*cs - b0*sn;
            vec[j+half] = b0*cs + a0*sn;
        }
    }
    /* write KV cache */
    for (int h = 0; h < KV; h++) {
        memcpy(m->K[li] + ((int64_t)h*m->max_t + pos)*hd, kk+h*hd, hd*sizeof(float));
        memcpy(m->V[li] + ((int64_t)h*m->max_t + pos)*hd, vv+h*hd, hd*sizeof(float));
    }
    float scale = 1.f/sqrtf((float)hd);
    float *ctx = falloc((int64_t)H*hd);
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < H; h++) {
        const float *qh = q+h*hd;
        const float *Kc = m->K[li] + (int64_t)(h/grp)*m->max_t*hd;
        const float *Vc = m->V[li] + (int64_t)(h/grp)*m->max_t*hd;
        float sc[8192];
        for (int t = 0; t <= pos; t++) {
            const float *kv = Kc + (int64_t)t*hd; float acc=0;
            for (int j = 0; j < hd; j++) acc += qh[j]*kv[j];
            sc[t]=acc*scale;
        }
        softmax_(sc, pos+1);
        float *oh = ctx+h*hd;
        for (int j = 0; j < hd; j++) oh[j]=0.f;
        for (int t = 0; t <= pos; t++) {
            const float *vvv = Vc + (int64_t)t*hd; float w=sc[t];
            for (int j = 0; j < hd; j++) oh[j]+=w*vvv[j];
        }
    }
    /* output gate then o_proj */
    for (int i = 0; i < H*hd; i++) ctx[i]*=sigmoidf_(gate[i]);
    DM(m, out, ctx, l->o, l->Qo, H*hd, c->hidden);
    free(qg); free(kk); free(vv); free(q); free(gate); free(ctx);
}

/* ---------- expert LRU (int8 container: name + name.qs) ---------- */
static Slot *expert_get(Model *m, int layer, int eid) {
    LCache *lc = &m->cache[layer]; Cfg *c = &m->c;
    for (int i = 0; i < lc->n; i++) if (lc->slots[i].eid-1 == eid) {
        m->hits++; lc->slots[i].used = ++m->clock; return &lc->slots[i];
    }
    m->miss++;
    int64_t ng = (int64_t)c->moe_inter*c->hidden, nd = (int64_t)c->hidden*c->moe_inter;
    Slot *s;
    if (lc->n < lc->cap) {
        s = &lc->slots[lc->n++];
        s->g=malloc(ng); s->u=malloc(ng); s->d=malloc(nd);
        s->gs=falloc(c->moe_inter); s->us=falloc(c->moe_inter); s->ds=falloc(c->hidden);
    } else { int lru=0; for (int i=1;i<lc->n;i++) if (lc->slots[i].used<lc->slots[lru].used) lru=i; s=&lc->slots[lru]; }
    char nm[320], qs[336];
    #define RD(W,SC,proj) do { \
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d." proj ".weight",layer,eid); \
        snprintf(qs,sizeof(qs),"%s.qs",nm); \
        st_read_raw(&m->S,nm,W,1); st_read_f32(&m->S,qs,SC,1); } while(0)
    RD(s->g,s->gs,"gate_proj"); RD(s->u,s->us,"up_proj"); RD(s->d,s->ds,"down_proj");
    #undef RD
    s->eid = eid+1; s->used = ++m->clock;
    return s;
}

/* ---------- MoE, one token ---------- */
static void moe_step(Model *m, Layer *l, const float *x, float *out) {
    Cfg *c = &m->c; int D=c->hidden, E=c->n_exp, K=c->topk, MI=c->moe_inter, SI=c->shared_inter;
    /* shared expert with sigmoid gate */
    float *g = falloc(SI>MI?SI:MI), *u = falloc(SI>MI?SI:MI);
    DM(m, g, x, l->sh_g, l->Qsg, D, SI); DM(m, u, x, l->sh_u, l->Qsu, D, SI);
    for (int i = 0; i < SI; i++) g[i]=siluf_(g[i])*u[i];
    DM(m, out, g, l->sh_d, l->Qsd, SI, D);
    float sg = 0; for (int i = 0; i < D; i++) sg += l->sh_gate[i]*x[i];
    sg = sigmoidf_(sg);
    for (int i = 0; i < D; i++) out[i]*=sg;
    /* router: softmax(f32) -> topk -> renorm */
    float *pr = falloc(E);
    DM(m, pr, x, l->gate, l->Qgate, D, E);
    softmax_(pr, E);
    int idx[64]; float val[64];
    for (int kk = 0; kk < K; kk++) {
        int best=-1; float bv=-1e30f;
        for (int e = 0; e < E; e++) {
            int taken=0; for (int j = 0; j < kk; j++) if (idx[j]==e){taken=1;break;}
            if (!taken && pr[e]>bv){bv=pr[e];best=e;}
        }
        idx[kk]=best; val[kk]=bv;
    }
    float sm=0; for (int kk=0;kk<K;kk++) sm+=val[kk];
    for (int kk=0;kk<K;kk++) val[kk]/=sm;
    float *hh = falloc(D);
    int li = (int)(l - m->L);
    for (int kk = 0; kk < K; kk++) {
        int e = idx[kk];
        if (m->stream) {
            Slot *s = expert_get(m, li, e);
            matmul_q(g, x, s->g, s->gs, D, MI); matmul_q(u, x, s->u, s->us, D, MI);
            for (int i = 0; i < MI; i++) g[i]=siluf_(g[i])*u[i];
            matmul_q(hh, g, s->d, s->ds, MI, D);
        } else {
            matvec(g, x, l->eg[e], D, MI); matvec(u, x, l->eu[e], D, MI);
            for (int i = 0; i < MI; i++) g[i]=siluf_(g[i])*u[i];
            matvec(hh, g, l->ed[e], MI, D);
        }
        for (int i = 0; i < D; i++) out[i]+=val[kk]*hh[i];
    }
    free(g); free(u); free(pr); free(hh);
}

static double g_t_delta, g_t_attn, g_t_moe, g_t_head; static int g_prof = -1;
/* ---------- one token through the stack; returns logits (malloc) ---------- */
static float *step(Model *m, int id) {
    Cfg *c = &m->c; int D = c->hidden;
    float *x = falloc(D);
    memcpy(x, m->embed + (int64_t)id*D, D*sizeof(float));
    float *nrm = falloc(D);
    int val_dim = c->lin_vh*c->lin_vd;
    float *tmp = falloc(val_dim > D ? val_dim : D);        /* deltanet uses [val_dim] scratch */
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        rmsnorm(nrm, x, l->in_ln, D, c->eps);
        double tt = now_s();
        if (l->is_full) { attn_step(m, l, i, nrm, m->pos, tmp); g_t_attn += now_s()-tt; }
        else            { deltanet_step(m, l, i, nrm, tmp);     g_t_delta += now_s()-tt; }
        for (int j = 0; j < D; j++) x[j]+=tmp[j];
        rmsnorm(nrm, x, l->post_ln, D, c->eps);
        tt = now_s();
        moe_step(m, l, nrm, tmp);
        g_t_moe += now_s()-tt;
        for (int j = 0; j < D; j++) x[j]+=tmp[j];
    }
    m->pos++;
    rmsnorm(x, x, m->final_norm, D, c->eps);
    float *logit = falloc(c->vocab);
    double tt = now_s();
    if (m->stream) matmul_q(logit, x, m->Qlmh.q, m->Qlmh.s, D, c->vocab);
    else matvec(logit, x, m->lm_head, D, c->vocab);
    g_t_head += now_s()-tt;
    free(x); free(nrm); free(tmp);
    return logit;
}

/* ---------- serve mode (PROTOCOL.md v1: JSON lines su stdin/stdout) ---------- */
static int g_eos[8]; static int g_n_eos = 0;
static void load_eos(const char *snap) {
    /* generation_config.json se c'e', altrimenti config.json (cfg_root gestisce text_config) */
    char path[2048]; snprintf(path,sizeof(path),"%s/generation_config.json",snap);
    jval *v = NULL;
    FILE *f = fopen(path,"rb");
    if (f) {
        fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
        char *buf=malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
        char *arena=NULL; jval *r=json_parse(buf,&arena);
        v = json_get(r,"eos_token_id");
    }
    if (!v || v->t==J_NULL) v = cfg_root ? json_get(cfg_root,"eos_token_id") : NULL;
    if (!v) return;
    if (v->t==J_NUM) g_eos[g_n_eos++]=(int)v->num;
    else if (v->t==J_ARR) for (int i=0;i<v->len && g_n_eos<8;i++)
        if (v->kids[i]->t==J_NUM) g_eos[g_n_eos++]=(int)v->kids[i]->num;
}
static int is_eos(int t){ for(int i=0;i<g_n_eos;i++) if(g_eos[i]==t) return 1; return 0; }

static int serve(const char *snap) {
    Model m; memset(&m,0,sizeof(m));
    model_init(&m, snap, 8192);
    load_eos(snap);
    Cfg *c = &m.c;
    char nb[1024]; snprintf(nb,sizeof(nb),"%s",snap);       /* basename, slash finali tolti */
    size_t bl=strlen(nb); while (bl>1 && nb[bl-1]=='/') nb[--bl]=0;
    char *base = strrchr(nb,'/'); base = base? base+1 : nb;
    printf("{\"ready\":true,\"model\":\"%s\",\"n_layers\":%d,\"vocab\":%d}\n", base, c->n_layers, c->vocab);
    fflush(stdout);
    int conv_dim = 2*c->lin_kh*c->lin_kd + c->lin_vh*c->lin_vd;
    char *line=NULL; size_t lcap=0; ssize_t got;
    while ((got=getline(&line,&lcap,stdin)) > 0) {
        if (got==1 && line[0]=='\n') continue;
        char *arena=NULL; jval *r=json_parse(line,&arena);
        jval *jid=json_get(r,"id"); long rid = jid? (long)jid->num : 0;
        jval *js=json_get(r,"stop");
        jval *jids=json_get(r,"ids"), *jn=json_get(r,"n"), *jr=json_get(r,"reset");
        if (js && js->boolean) {                             /* stop tra un token e l'altro: gia' fermi */
            printf("{\"id\":%ld,\"done\":true,\"n_out\":0,\"prefill_s\":0.0,\"decode_s\":0.0,\"hit\":0.0}\n",rid);
            fflush(stdout); continue;
        }
        int n = jn? (int)jn->num : 0;
        int nids = (jids && jids->t==J_ARR) ? jids->len : 0;
        int reset = jr && (jr->t==J_BOOL ? jr->boolean : jr->num!=0);
        if (reset) {
            m.pos = 0;
            for (int i = 0; i < c->n_layers; i++) if (!m.L[i].is_full) {
                memset(m.conv_st[i],0,(size_t)conv_dim*(c->conv_k-1)*sizeof(float));
                memset(m.rec_st[i],0,(size_t)c->lin_vh*c->lin_kd*c->lin_vd*sizeof(float));
            }   /* KV non serve azzerarlo: pos limita le letture */
        }
        if (m.pos + nids + n >= m.max_t) {
            printf("{\"id\":%ld,\"error\":\"context full\"}\n",rid); fflush(stdout); continue;
        }
        float *logit=NULL;
        double tp0=now_s();
        for (int i = 0; i < nids; i++) { free(logit); logit = step(&m,(int)jids->kids[i]->num); }
        double prefill=now_s()-tp0, td0=now_s();
        int n_out=0;
        for (int i = 0; i < n && logit; i++) {
            int best=0; float bv=logit[0];
            for (int v = 1; v < c->vocab; v++) if (logit[v]>bv){bv=logit[v];best=v;}
            free(logit); logit=NULL;
            printf("{\"id\":%ld,\"tok\":%d}\n",rid,best); fflush(stdout);
            n_out++;
            logit = step(&m,best);                           /* anche i generati avanzano lo stato */
            if (is_eos(best)) break;
        }
        free(logit);
        double hit = m.stream && (m.hits+m.miss) ? (double)m.hits/(double)(m.hits+m.miss) : 0.0;
        printf("{\"id\":%ld,\"done\":true,\"n_out\":%d,\"prefill_s\":%.3f,\"decode_s\":%.3f,\"hit\":%.3f}\n",
               rid, n_out, prefill, now_s()-td0, hit);
        fflush(stdout);
    }
    return 0;
}

/* ---------- validation main ---------- */
static int *read_ints(jval *o, const char *key, int *n) {
    jval *a = json_get(o,key); if(!a){*n=0;return NULL;}
    int *r = malloc(a->len*sizeof(int));
    for (int i = 0; i < a->len; i++) r[i]=(int)a->kids[i]->num;
    *n=a->len; return r;
}
int main(int argc, char **argv) {
    const char *snap = getenv("SNAP");
    if (!snap) { fprintf(stderr,"set SNAP=<model dir>\n"); return 1; }
    if (argc>1 && !strcmp(argv[1],"--serve")) return serve(snap);
    const char *refpath = argc>1 ? argv[1] : "ref_qwen.json";
    FILE *f=fopen(refpath,"rb"); if(!f){perror(refpath);return 1;}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf=malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    char *arena=NULL; jval *ref=json_parse(buf,&arena);
    int np,nf; int *prompt=read_ints(ref,"prompt_ids",&np); int *full=read_ints(ref,"full_ids",&nf);
    int n_new = nf-np;

    Model m; memset(&m,0,sizeof(m)); double t0=now_s();
    model_init(&m, snap, nf+8);
    printf("qwen3.6 engine: %d layers (%d full-attn), loaded in %.1fs\n",
           m.c.n_layers, m.c.n_layers/m.c.full_interval, now_s()-t0);

    /* greedy: feed prompt token by token (recurrent path), then generate */
    int cur=-1, match=0;
    float *logit=NULL;
    double tp0=now_s();
    for (int i = 0; i < np; i++) { free(logit); logit = step(&m, prompt[i]); }
    double tp=now_s()-tp0;
    fprintf(stderr,"prefill: %d tok in %.2fs (%.2f tok/s)\n", np, tp, np/tp);
    double td0=now_s();
    printf("reference: "); for (int i = np; i < nf; i++) printf("%d ", full[i]);
    printf("\nengine   : ");
    for (int i = 0; i < n_new; i++) {
        int best=0; float bv=logit[0];
        for (int v = 1; v < m.c.vocab; v++) if (logit[v]>bv){bv=logit[v];best=v;}
        printf("%d ",best); fflush(stdout);
        if (best==full[np+i]) match++;
        cur=best; free(logit);
        if (i<n_new-1) logit=step(&m,cur);
    }
    double td=now_s()-td0;
    fprintf(stderr,"decode: %d tok in %.2fs (%.2f tok/s)\n", n_new, td, n_new/td);
    fprintf(stderr,"phase totals: delta=%.1fs attn=%.1fs moe=%.1fs lm_head=%.1fs\n",
            g_t_delta, g_t_attn, g_t_moe, g_t_head);
    if (m.stream) fprintf(stderr,"expert cache: %.1f%% hit (%llu/%llu)\n",
        100.0*m.hits/(m.hits+m.miss+1e-9),(unsigned long long)m.hits,(unsigned long long)(m.hits+m.miss));
    printf("\ngreedy match: %d/%d %s\n", match, n_new, match==n_new?"EXACT":"");
    return match==n_new?0:1;
}
