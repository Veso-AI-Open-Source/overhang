/* Motore Gemma 4 MoE (gemma4_text) in C puro, con EXPERT-STREAMING dal disco.
 *
 * Due modalita', rilevate dal container:
 *  - ORACLE (tiny f32): expert fusi [E,2mi,D]/[E,D,mi] residenti f32 — usato
 *    da `make check-oracle-gemma` per validare la matematica vs transformers.
 *  - CONTAINER (convert_gemma.py): dense int8+scale residenti (embed f32,
 *    lm_head int8 quantizzato al load), expert per-singolo
 *    (`model.layers.N.experts.E.gate_up_proj` int8 + .qs) letti dal disco
 *    on-demand con cache LRU per-layer (QCACHE, cap interno).
 *
 * Prestazioni (stessa filosofia streaming, vedi WORKLOG 2026-07-16):
 *  - PREFILL BATCHED: step_batch processa S token per layer; gli expert si
 *    caricano per UNIONE sul batch (ogni expert colpito una volta sola, non
 *    8 volte per token) — l'ammortamento e' il punto dello streaming MoE.
 *  - ROUTER-LOOKAHEAD: mentre si calcola il layer L si stima il routing di
 *    L+1 sul residuo corrente e si fa st_prefetch dei suoi expert (readahead
 *    async: i miss trovano la page cache gia' calda).
 *  - LM_HEAD INT8 (solo container): il vocab 262K x 2816 in f32 sarebbe
 *    ~3 GB di traffico per token; quantizzato per-riga al load -> 4x meno.
 *  - DECODE SPECULATIVO n-gram (--serve): bozza di k token dal contesto,
 *    verifica in UN passo batched (stessa ammortizzazione degli expert);
 *    greedy-equivalente per costruzione (si accetta solo cio' che combacia).
 *    GEMMA_SPEC=0 per disattivare.
 *
 * Architettura (da modeling_gemma4.py, transformers 5.13.1):
 *  - embed * sqrt(hidden); RMSNorm "plain weight" (NON 1+w), eps dentro la media
 *  - layer sliding (window causale) e full alternati via config.layer_types
 *  - full attention: k_eq_v -> v_proj assente, value = k_proj PRE-norm;
 *    global kv heads/dim propri; rope "proportional" parziale (freq zero oltre
 *    rope_angles). sliding: rope default su tutto head_dim
 *  - q_norm/k_norm per-testa (con peso); v_norm SENZA peso (RMS puro)
 *  - attention scaling = 1.0 (niente 1/sqrt(d))
 *  - FFN: mlp denso e blocco MoE IN PARALLELO sullo stesso residuo, sommati
 *    dopo le rispettive post-norm; router sul residuo pre-norm con
 *    rms_puro * scale * hidden^-0.5, softmax f32, topk rinormalizzato * per_expert_scale
 *  - x *= layer_scalar a fine layer; lm_head legato all'embedding; softcap 30
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "st.h"
#include "json.h"

#ifdef OM_METAL
#include "om_metal.h"
static int g_omm = -1;      /* -1 unset, 0 off, 1 GPU ready. OM_METAL=0 disabilita */
static int omm_on(void) {
    if (g_omm < 0) {
        const char *e = getenv("OM_METAL");
        g_omm = (e && *e == '0') ? 0 : omm_init();
    }
    return g_omm;
}
static void *slot_alloc(size_t n) { return omm_on() ? omm_alloc(n) : malloc(n); }
#else
static void *slot_alloc(size_t n) { return malloc(n); }
#endif

/* ---------- config ---------- */
typedef struct {
    int hidden, n_layers, n_heads, n_kv, head_dim;   /* sliding attention */
    int n_gkv, ghead_dim;                            /* full attention (k_eq_v) */
    int n_experts, topk, moe_inter, inter, vocab;
    int window;
    float eps, softcap;
    float theta_slide, theta_full, partial_full;
    int *is_full;                                    /* [n_layers] */
} Cfg;

static void load_cfg(Cfg *c, const char *snap) {
    char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if(!f){perror(path);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf = malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    char *arena=NULL; jval *r = json_parse(buf, &arena);
    c->hidden    = (int)json_get(r,"hidden_size")->num;
    c->n_layers  = (int)json_get(r,"num_hidden_layers")->num;
    c->n_heads   = (int)json_get(r,"num_attention_heads")->num;
    c->n_kv      = (int)json_get(r,"num_key_value_heads")->num;
    c->head_dim  = (int)json_get(r,"head_dim")->num;
    c->n_gkv     = (int)json_get(r,"num_global_key_value_heads")->num;
    c->ghead_dim = (int)json_get(r,"global_head_dim")->num;
    c->n_experts = (int)json_get(r,"num_experts")->num;
    c->topk      = (int)json_get(r,"top_k_experts")->num;
    c->moe_inter = (int)json_get(r,"moe_intermediate_size")->num;
    c->inter     = (int)json_get(r,"intermediate_size")->num;
    c->vocab     = (int)json_get(r,"vocab_size")->num;
    c->window    = (int)json_get(r,"sliding_window")->num;
    c->eps       = (float)json_get(r,"rms_norm_eps")->num;
    jval *sc = json_get(r,"final_logit_softcapping");
    c->softcap = (sc && sc->t==J_NUM) ? (float)sc->num : 0.f;
    jval *rp = json_get(r,"rope_parameters");
    c->theta_full   = (float)json_get(json_get(rp,"full_attention"),"rope_theta")->num;
    c->partial_full = (float)json_get(json_get(rp,"full_attention"),"partial_rotary_factor")->num;
    c->theta_slide  = (float)json_get(json_get(rp,"sliding_attention"),"rope_theta")->num;
    jval *lt = json_get(r,"layer_types");
    c->is_full = calloc(c->n_layers, sizeof(int));
    for (int i = 0; i < c->n_layers && i < lt->len; i++)
        c->is_full[i] = strcmp(lt->kids[i]->str, "full_attention") == 0;
    free(buf); free(arena);
}

/* ---------- matrice residente: f32 O int8 per-riga + scala ---------- */
typedef struct { float *f; int8_t *q; float *s; } Mat;

/* ---------- cache LRU degli expert streamati (int8 + scale) ---------- */
typedef struct {
    int eid; uint64_t used;
    int8_t *gq, *dq; float *gs, *ds;   /* gate_up [2mi,D] / down [D,mi] */
} Slot;
typedef struct { Slot *slots; int n, cap; } LCache;

typedef struct {
    float *in_ln, *post_attn_ln, *pre_ffw, *post_ffw;
    float *post_ffw1, *post_ffw2, *pre_ffw2;  /* norme del blocco MoE */
    float layer_scalar;
    Mat q, k, v, o, mg, mu, md, rproj;        /* v assente sui layer full */
    float *qn, *kn, *rscale, *pes;
    float *eg, *ed;                           /* modalita' oracle: fusi f32 */
    int is_full, hd, nkv;
} Layer;

typedef struct {
    Cfg c;
    shards S;
    int stream;                               /* container int8 per-expert */
    float *embed, *final_norm;
    int8_t *lm8; float *lm8s;                 /* lm_head int8 (solo stream) */
    Layer *L;
    LCache *cache;                            /* [n_layers], solo stream */
    uint64_t clock, hits, miss;
    float **K, **V; int pos, max_t;
    float *invs, *invf;                       /* inv_freq per tipo di layer */
} Model;

static float *falloc(int64_t n){ float *p=malloc(n*sizeof(float)); if(!p){fprintf(stderr,"OOM %ld\n",(long)n);exit(1);} return p; }

/* profiling per fase (GPROF=1): accumulatori azzerati a ogni done-line */
static double g_t_attn, g_t_dense, g_t_route, g_t_eload, g_t_ecomp, g_t_head;
static double prof_now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec*1e-9;
}

static float *load_f32(Model *m, const char *nm) {
    int64_t n = st_numel(&m->S, nm);
    if (n < 0) { fprintf(stderr, "missing %s\n", nm); exit(1); }
    float *p = falloc(n);
    st_read_f32(&m->S, nm, p, 0);
    return p;
}

/* carica una matrice: dtype I8 nel container -> int8+.qs, altrimenti f32 */
static void mat_load(Model *m, Mat *w, const char *nm, int rows) {
    st_tensor *t = st_find(&m->S, nm);
    if (!t) { fprintf(stderr, "missing %s\n", nm); exit(1); }
    memset(w, 0, sizeof(*w));
    if (t->dtype == 3) {
        /* slot_alloc: con OM_METAL peso e scale vivono nell'arena condivisa
         * GPU-visibile (zero-copy per dq_gemm) */
        w->q = slot_alloc(t->nbytes);
        w->s = slot_alloc(rows*sizeof(float));
        if(!w->q || !w->s){fprintf(stderr,"OOM %s\n",nm);exit(1);}
        st_read_raw(&m->S, nm, w->q, 0);
        char qs[300]; snprintf(qs, sizeof(qs), "%s.qs", nm);
        st_read_f32(&m->S, qs, w->s, 0);
    } else {
        w->f = load_f32(m, nm);
    }
}

/* ---------- primitive ---------- */
/* y[O] = W[O,I] @ x[I], f32 */
static void matvec(float *y, const float *W, const float *x, int O, int I) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o*I;
        float acc = 0.f;
        for (int i = 0; i < I; i++) acc += w[i]*x[i];
        y[o] = acc;
    }
}

/* y[O] = W_q[O,I] @ x con W int8 per-riga + scala (via int8xint8 NEON) */
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

/* variante SERIALE (nessuna regione omp): per l'interno di regioni parallele
 * a grana grossa (un thread = un expert), dove il costo di sync di ~1000
 * micro-regioni per token dominava il decode (misurato col sampler). */
static void matmul_q_st(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
#if defined(__ARM_NEON)
    static int idot = -1;
    if (idot < 0) { const char *e = getenv("IDOT"); idot = !(e && *e == '0'); }
    if (idot && I % 16 == 0 && I <= 8192) {
        int nb = I / 16;
        int8_t xi[8192]; float xs[512];
        for (int b = 0; b < nb; b++) {
            const float *xb = x + b * 16;
            float amax = 0.f; for (int i = 0; i < 16; i++) { float a = fabsf(xb[i]); if (a > amax) amax = a; }
            float s = amax / 127.f; if (s < 1e-12f) s = 1e-12f;
            xs[b] = s; float inv = 1.f / s;
            for (int i = 0; i < 16; i++) xi[b * 16 + i] = (int8_t)lrintf(xb[i] * inv);
        }
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o * I;
            float acc = 0.f;
            for (int b = 0; b < nb; b++) acc += xs[b] * (float)dot_i8_16(xi + b * 16, w + b * 16);
            y[o] = acc * scale[o];
        }
        return;
    }
#endif
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        float acc = 0.f;
        for (int i = 0; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
}

static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
#if defined(__ARM_NEON)
    static int idot = -1;
    if (idot < 0) { const char *e = getenv("IDOT"); idot = !(e && *e == '0'); }
    if (idot && I % 16 == 0 && I <= 8192) {
        int nb = I / 16;
        int8_t xi[8192]; float xs[512];
        for (int b = 0; b < nb; b++) {
            const float *xb = x + b * 16;
            float amax = 0.f; for (int i = 0; i < 16; i++) { float a = fabsf(xb[i]); if (a > amax) amax = a; }
            float s = amax / 127.f; if (s < 1e-12f) s = 1e-12f;
            xs[b] = s; float inv = 1.f / s;
            for (int i = 0; i < 16; i++) xi[b * 16 + i] = (int8_t)lrintf(xb[i] * inv);
        }
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o * I;
            float acc = 0.f;
            for (int b = 0; b < nb; b++) acc += xs[b] * (float)dot_i8_16(xi + b * 16, w + b * 16);
            y[o] = acc * scale[o];
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        float acc = 0.f;
        for (int i = 0; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
}

/* y[O] = W[O,I] @ x[I], f32 seriale */
static void matvec_st(float *y, const float *W, const float *x, int O, int I) {
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o*I;
        float acc = 0.f;
        for (int i = 0; i < I; i++) acc += w[i]*x[i];
        y[o] = acc;
    }
}

static void mat_apply(float *y, const Mat *w, const float *x, int O, int I) {
    if (w->f) matvec(y, w->f, x, O, I);
    else matmul_q(y, x, w->q, w->s, I, O);
}

/* y[S,O] = x[S,D] @ W^T batched. Hook GPU (stream, S>=8): dq_gemm su int8. */
static void mat_apply_b(float *y, const Mat *w, const float *x, int S, int O, int I) {
#ifdef OM_METAL
    if (w->q && S >= 8 && omm_on() &&
        omm_dense_q(w->q, w->s, O, I, x, S, y) == 0) return;
#endif
    for (int s = 0; s < S; s++) mat_apply(y + (int64_t)s*O, w, x + (int64_t)s*I, O, I);
}

/* quantizzazione per-riga int8 (replica quantize_rows del converter) */
static void quantize_rows_c(const float *w, int8_t *q, float *scale, int64_t O, int I) {
    #pragma omp parallel for schedule(static)
    for (int64_t o = 0; o < O; o++) {
        const float *wr = w + o*I;
        float amax = 0.f; for (int i = 0; i < I; i++) { float a = fabsf(wr[i]); if (a > amax) amax = a; }
        float s = amax / 127.f; if (s < 1e-12f) s = 1e-12f;
        scale[o] = s;
        int8_t *qr = q + o*I;
        for (int i = 0; i < I; i++) {
            int v = (int)lrintf(wr[i] / s);
            if (v > 127) v = 127; if (v < -128) v = -128;
            qr[i] = (int8_t)v;
        }
    }
}

/* Gemma4RMSNorm: x * (mean(x^2)+eps)^-0.5 [* w]; peso "plain", NON 1+w */
static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    double ms = 0; for (int i = 0; i < n; i++) ms += (double)x[i]*x[i];
    float r = powf((float)(ms/n) + eps, -0.5f);
    for (int i = 0; i < n; i++) out[i] = x[i]*r*(w ? w[i] : 1.f);
}

static float gelu_tanh(float x) {
    return 0.5f*x*(1.f + tanhf(0.7978845608028654f*(x + 0.044715f*x*x*x)));
}

/* rope stile rotate_half: coppie (j, j+half); freq nulle = identita' */
static void rope(float *x, int dim, int pos, const float *inv_freq) {
    int half = dim/2;
    for (int j = 0; j < half; j++) {
        float a = pos*inv_freq[j], cs = cosf(a), sn = sinf(a);
        float x1 = x[j], x2 = x[j+half];
        x[j]      = x1*cs - x2*sn;
        x[j+half] = x2*cs + x1*sn;
    }
}

/* ---------- expert streaming: LRU per layer, come olmoe.c ---------- */
static void expert_get(Model *m, int layer, int eid, Slot **out) {
    LCache *lc = &m->cache[layer];
    for (int i = 0; i < lc->n; i++) if (lc->slots[i].eid == eid) {
        m->hits++; lc->slots[i].used = ++m->clock; *out = &lc->slots[i]; return;
    }
    m->miss++;
    Cfg *c = &m->c;
    int64_t ng = (int64_t)2*c->moe_inter*c->hidden, nd = (int64_t)c->hidden*c->moe_inter;
    Slot *s;
    if (lc->n < lc->cap) {
        s = &lc->slots[lc->n++];
        s->gq = slot_alloc(ng); s->dq = slot_alloc(nd);
        s->gs = slot_alloc(2*c->moe_inter*sizeof(float)); s->ds = slot_alloc(c->hidden*sizeof(float));
        if (!s->gq || !s->dq || !s->gs || !s->ds) { fprintf(stderr,"OOM expert slot\n"); exit(1); }
    } else {
        int lru = 0;
        for (int i = 1; i < lc->n; i++) if (lc->slots[i].used < lc->slots[lru].used) lru = i;
        s = &lc->slots[lru];
    }
    char nm[300];
    snprintf(nm,sizeof(nm),"model.layers.%d.experts.%d.gate_up_proj",layer,eid);
    st_read_raw(&m->S, nm, s->gq, 1);
    snprintf(nm,sizeof(nm),"model.layers.%d.experts.%d.gate_up_proj.qs",layer,eid);
    st_read_f32(&m->S, nm, s->gs, 1);
    snprintf(nm,sizeof(nm),"model.layers.%d.experts.%d.down_proj",layer,eid);
    st_read_raw(&m->S, nm, s->dq, 1);
    snprintf(nm,sizeof(nm),"model.layers.%d.experts.%d.down_proj.qs",layer,eid);
    st_read_f32(&m->S, nm, s->ds, 1);
    s->eid = eid; s->used = ++m->clock;
    *out = s;
}

/* readahead async degli expert di un layer (router-lookahead) */
static void expert_prefetch(Model *m, int layer, const int *eids, int n) {
    char nm[300];
    for (int i = 0; i < n; i++) {
        snprintf(nm,sizeof(nm),"model.layers.%d.experts.%d.gate_up_proj",layer,eids[i]);
        st_prefetch(&m->S, nm);
        snprintf(nm,sizeof(nm),"model.layers.%d.experts.%d.down_proj",layer,eids[i]);
        st_prefetch(&m->S, nm);
    }
}

/* ---------- caricamento ---------- */
#define LNM(suffix) (snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i), nm)

static void model_init(Model *m, const char *snap, int max_t, int qcache) {
    memset(m, 0, sizeof(*m));
    load_cfg(&m->c, snap);
    st_init(&m->S, snap);
    Cfg *c = &m->c;
    m->stream = st_has(&m->S, "model.layers.0.experts.0.gate_up_proj");
    m->embed = load_f32(m, "model.embed_tokens.weight");
    m->final_norm = load_f32(m, "model.norm.weight");
    if (m->stream) {
        /* lm_head legato: copia int8 per il matvec sul vocab (4x meno traffico) */
        m->lm8 = slot_alloc((int64_t)c->vocab*c->hidden);
        m->lm8s = falloc(c->vocab);
        if (!m->lm8) { fprintf(stderr,"OOM lm8\n"); exit(1); }
        quantize_rows_c(m->embed, m->lm8, m->lm8s, c->vocab, c->hidden);
    }
    m->L = calloc(c->n_layers, sizeof(Layer));
    char nm[256];
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        l->is_full = c->is_full[i];
        l->hd  = l->is_full ? c->ghead_dim : c->head_dim;
        l->nkv = l->is_full ? c->n_gkv : c->n_kv;
        l->in_ln        = load_f32(m, LNM("input_layernorm.weight"));
        l->post_attn_ln = load_f32(m, LNM("post_attention_layernorm.weight"));
        l->pre_ffw      = load_f32(m, LNM("pre_feedforward_layernorm.weight"));
        l->post_ffw     = load_f32(m, LNM("post_feedforward_layernorm.weight"));
        l->post_ffw1    = load_f32(m, LNM("post_feedforward_layernorm_1.weight"));
        l->post_ffw2    = load_f32(m, LNM("post_feedforward_layernorm_2.weight"));
        l->pre_ffw2     = load_f32(m, LNM("pre_feedforward_layernorm_2.weight"));
        float *ls = load_f32(m, LNM("layer_scalar"));
        l->layer_scalar = ls[0]; free(ls);
        mat_load(m, &l->q, LNM("self_attn.q_proj.weight"), c->n_heads*l->hd);
        l->qn = load_f32(m, LNM("self_attn.q_norm.weight"));
        mat_load(m, &l->k, LNM("self_attn.k_proj.weight"), l->nkv*l->hd);
        l->kn = load_f32(m, LNM("self_attn.k_norm.weight"));
        if (!l->is_full)                       /* k_eq_v sui layer full */
            mat_load(m, &l->v, LNM("self_attn.v_proj.weight"), l->nkv*l->hd);
        mat_load(m, &l->o, LNM("self_attn.o_proj.weight"), c->hidden);
        mat_load(m, &l->mg, LNM("mlp.gate_proj.weight"), c->inter);
        mat_load(m, &l->mu, LNM("mlp.up_proj.weight"), c->inter);
        mat_load(m, &l->md, LNM("mlp.down_proj.weight"), c->hidden);
        mat_load(m, &l->rproj, LNM("router.proj.weight"), c->n_experts);
        l->rscale = load_f32(m, LNM("router.scale"));
        l->pes    = load_f32(m, LNM("router.per_expert_scale"));
        if (!m->stream) {
            l->eg = load_f32(m, LNM("experts.gate_up_proj"));
            l->ed = load_f32(m, LNM("experts.down_proj"));
        }
    }
    if (m->stream) {
        m->cache = calloc(c->n_layers, sizeof(LCache));
        for (int i = 0; i < c->n_layers; i++) {
            m->cache[i].cap = qcache;
            m->cache[i].slots = calloc(qcache, sizeof(Slot));
        }
    }
    m->max_t = max_t;
    m->K = calloc(c->n_layers, sizeof(float*));
    m->V = calloc(c->n_layers, sizeof(float*));
    for (int i = 0; i < c->n_layers; i++) {
        m->K[i] = falloc((int64_t)m->L[i].nkv * max_t * m->L[i].hd);
        m->V[i] = falloc((int64_t)m->L[i].nkv * max_t * m->L[i].hd);
    }
    /* inv_freq per tipo (compute_default / _proportional) */
    m->invs = falloc(c->head_dim/2);
    for (int j = 0; j < c->head_dim/2; j++)
        m->invs[j] = 1.f/powf(c->theta_slide, (float)(2*j)/c->head_dim);
    int rope_angles = (int)(c->partial_full*c->ghead_dim)/2;
    m->invf = falloc(c->ghead_dim/2);
    for (int j = 0; j < c->ghead_dim/2; j++)
        m->invf[j] = j < rope_angles ? 1.f/powf(c->theta_full, (float)(2*j)/c->ghead_dim) : 0.f;
}

/* routing di un token: probs softmax -> topk rinormalizzato * per_expert_scale */
static void route_token(Model *m, Layer *l, const float *xrow, float *nrm,
                        int *top, float *tw) {
    Cfg *c = &m->c;
    rmsnorm(nrm, xrow, NULL, c->hidden, c->eps);
    float rroot = 1.f/sqrtf((float)c->hidden);
    for (int i = 0; i < c->hidden; i++) nrm[i] *= l->rscale[i]*rroot;
    float probs[512];
    mat_apply(probs, &l->rproj, nrm, c->n_experts, c->hidden);
    float pmx = -1e30f;
    for (int e = 0; e < c->n_experts; e++) if (probs[e] > pmx) pmx = probs[e];
    float psum = 0.f;
    for (int e = 0; e < c->n_experts; e++) { probs[e] = expf(probs[e]-pmx); psum += probs[e]; }
    float wsum = 0.f;
    for (int k = 0; k < c->topk; k++) {
        int best = -1; float bv = -1.f;
        for (int e = 0; e < c->n_experts; e++) {
            int taken = 0;
            for (int j = 0; j < k; j++) if (top[j] == e) taken = 1;
            if (!taken && probs[e] > bv) { bv = probs[e]; best = e; }
        }
        top[k] = best; tw[k] = bv/psum; wsum += bv/psum;
    }
    for (int k = 0; k < c->topk; k++) tw[k] = tw[k]/wsum*l->pes[top[k]];
}

/* ---------- S token attraverso lo stack (batched) ----------
 * ids[S] entra alle posizioni m->pos .. m->pos+S-1. Ritorna i logits delle
 * ultime n_tail posizioni (malloc, [n_tail, vocab]). */
static float *step_batch(Model *m, const int *ids, int S, int n_tail) {
    Cfg *c = &m->c; int D = c->hidden;
    int p0 = m->pos;
    float *x = falloc((int64_t)S*D);
    float scale_e = sqrtf((float)D);
    for (int s = 0; s < S; s++)
        for (int i = 0; i < D; i++)
            x[(int64_t)s*D+i] = m->embed[(int64_t)ids[s]*D + i]*scale_e;

    int maxhd = c->ghead_dim > c->head_dim ? c->ghead_dim : c->head_dim;
    int maxkv = c->n_kv > c->n_gkv ? c->n_kv : c->n_gkv;
    float *nrm = falloc((int64_t)S*D);
    float *qb  = falloc((int64_t)S*c->n_heads*maxhd);
    float *kb  = falloc((int64_t)S*maxkv*maxhd);
    int imax = c->inter > 2*c->moe_inter ? c->inter : 2*c->moe_inter;
    float *t1 = falloc((int64_t)S*imax), *t2 = falloc((int64_t)S*imax);
    float *ffn = falloc((int64_t)S*D), *moe = falloc((int64_t)S*D), *tmp = falloc((int64_t)S*D);
    int *top = malloc(S*c->topk*sizeof(int));
    float *tw = malloc(S*c->topk*sizeof(float));
    int *la_top = malloc(c->topk*sizeof(int));
    float *la_tw = malloc(c->topk*sizeof(float));

    for (int li = 0; li < c->n_layers; li++) {
        Layer *l = &m->L[li];
        int hd = l->hd, nkv = l->nkv, groups = c->n_heads/nkv;
        const float *inv = l->is_full ? m->invf : m->invs;

        /* --- attention: prima K/V di tutto il batch, poi i punteggi --- */
        double tp_ = prof_now();
        for (int s = 0; s < S; s++) rmsnorm(nrm+(int64_t)s*D, x+(int64_t)s*D, l->in_ln, D, c->eps);
        mat_apply_b(qb, &l->q, nrm, S, c->n_heads*hd, D);
        mat_apply_b(kb, &l->k, nrm, S, nkv*hd, D);
        for (int s = 0; s < S; s++) {
            int pos = p0 + s;
            for (int h = 0; h < c->n_heads; h++) {
                float *qh = qb + ((int64_t)s*c->n_heads + h)*hd;
                rmsnorm(qh, qh, l->qn, hd, c->eps);
                rope(qh, hd, pos, inv);
            }
            for (int h = 0; h < nkv; h++) {
                float *kh = kb + ((int64_t)s*nkv + h)*hd;
                float *Krow = m->K[li] + ((int64_t)h*m->max_t + pos)*hd;
                float *Vrow = m->V[li] + ((int64_t)h*m->max_t + pos)*hd;
                if (l->is_full) {
                    /* k_eq_v: value = k_proj PRE k_norm, poi v_norm senza peso */
                    rmsnorm(Vrow, kh, NULL, hd, c->eps);
                }
                rmsnorm(kh, kh, l->kn, hd, c->eps);
                rope(kh, hd, pos, inv);
                memcpy(Krow, kh, hd*sizeof(float));
            }
        }
        if (!l->is_full) {
            mat_apply_b(kb, &l->v, nrm, S, nkv*hd, D);
            for (int s = 0; s < S; s++)
                for (int h = 0; h < nkv; h++) {
                    float *Vrow = m->V[li] + ((int64_t)h*m->max_t + p0 + s)*hd;
                    rmsnorm(Vrow, kb + ((int64_t)s*nkv + h)*hd, NULL, hd, c->eps);
                }
        }
        #pragma omp parallel for schedule(static) collapse(2)
        for (int s = 0; s < S; s++) {
            for (int h = 0; h < c->n_heads; h++) {
                int pos = p0 + s;
                int start = l->is_full ? 0 : (pos - c->window + 1 < 0 ? 0 : pos - c->window + 1);
                const float *qh = qb + ((int64_t)s*c->n_heads + h)*hd;
                int kvh = h/groups;
                const float *Kh = m->K[li] + (int64_t)kvh*m->max_t*hd;
                const float *Vh = m->V[li] + (int64_t)kvh*m->max_t*hd;
                float sc[8192]; float mx = -1e30f;
                for (int t = start; t <= pos; t++) {
                    float acc = 0.f;
                    for (int j = 0; j < hd; j++) acc += qh[j]*Kh[(int64_t)t*hd + j];
                    sc[t-start] = acc;                    /* scaling = 1.0 */
                    if (acc > mx) mx = acc;
                }
                float sum = 0.f;
                for (int t = 0; t <= pos-start; t++) { sc[t] = expf(sc[t]-mx); sum += sc[t]; }
                /* output riusa qb (q non serve piu' per questa testa) */
                float *oh = qb + ((int64_t)s*c->n_heads + h)*hd;
                float acc[512];
                memset(acc, 0, hd*sizeof(float));
                for (int t = 0; t <= pos-start; t++) {
                    float p = sc[t]/sum;
                    const float *vr = Vh + (int64_t)(t+start)*hd;
                    for (int j = 0; j < hd; j++) acc[j] += p*vr[j];
                }
                memcpy(oh, acc, hd*sizeof(float));
            }
        }
        mat_apply_b(tmp, &l->o, qb, S, D, c->n_heads*hd);
        for (int s = 0; s < S; s++) {
            rmsnorm(tmp+(int64_t)s*D, tmp+(int64_t)s*D, l->post_attn_ln, D, c->eps);
            for (int i = 0; i < D; i++) x[(int64_t)s*D+i] += tmp[(int64_t)s*D+i];
        }
        g_t_attn += prof_now()-tp_; tp_ = prof_now();

        /* --- routing per token (sul residuo x) + LOOKAHEAD del layer L+1 --- */
        for (int s = 0; s < S; s++)
            route_token(m, l, x+(int64_t)s*D, nrm+(int64_t)s*D, top+s*c->topk, tw+s*c->topk);
        if (m->stream && li+1 < c->n_layers && S >= 8) {
            /* stima del routing di L+1 sul residuo corrente dell'ultimo token:
             * approssimata ma correlata; il readahead scalda la page cache.
             * SOLO in prefill: a decode (S piccolo) l'F_RDADVISE di macOS
             * costava ~115 ms/token (misurato) contro un LRU gia' a ~85%. */
            route_token(m, &m->L[li+1], x+(int64_t)(S-1)*D, nrm, la_top, la_tw);
            expert_prefetch(m, li+1, la_top, c->topk);
        }
        g_t_route += prof_now()-tp_; tp_ = prof_now();

        /* --- mlp denso (batched) --- */
        for (int s = 0; s < S; s++) rmsnorm(nrm+(int64_t)s*D, x+(int64_t)s*D, l->pre_ffw, D, c->eps);
        mat_apply_b(t1, &l->mg, nrm, S, c->inter, D);
        mat_apply_b(t2, &l->mu, nrm, S, c->inter, D);
        for (int64_t i = 0; i < (int64_t)S*c->inter; i++) t1[i] = gelu_tanh(t1[i])*t2[i];
        mat_apply_b(ffn, &l->md, t1, S, D, c->inter);
        for (int s = 0; s < S; s++)
            rmsnorm(ffn+(int64_t)s*D, ffn+(int64_t)s*D, l->post_ffw1, D, c->eps);  /* h1 */
        g_t_dense += prof_now()-tp_; tp_ = prof_now();

        /* --- expert per UNIONE sul batch: ogni expert colpito si carica e
         * processa UNA volta per tutti i token che lo hanno scelto ---
         * (con OM_METAL e S>=8 l'intera unione va in un command buffer GPU) */
        int mi = c->moe_inter;
        for (int s = 0; s < S; s++)
            rmsnorm(nrm+(int64_t)s*D, x+(int64_t)s*D, l->pre_ffw2, D, c->eps);  /* input expert */
        memset(moe, 0, (int64_t)S*D*sizeof(float));
        int gpu_done = 0;
#ifdef OM_METAL
        if (m->stream && S >= 8 && omm_on()) {
            /* unione sul batch, un solo command buffer (gelu_tanh in-kernel) */
            omm_job jobs[512]; int nj = 0;
            int *pbuf = malloc(S*c->topk*sizeof(int));
            float *wbuf = malloc(S*c->topk*sizeof(float));
            int cur = 0;
            for (int e = 0; e < c->n_experts; e++) {
                int j0 = cur;
                for (int s = 0; s < S; s++) for (int k = 0; k < c->topk; k++)
                    if (top[s*c->topk+k] == e) { pbuf[cur]=s; wbuf[cur]=tw[s*c->topk+k]; cur++; }
                if (cur == j0) continue;
                Slot *sl; expert_get(m, li, e, &sl);
                jobs[nj++] = (omm_job){sl->gq, sl->gq + (int64_t)mi*D, sl->dq,
                                       sl->gs, sl->gs + mi, sl->ds,
                                       cur - j0, pbuf + j0, wbuf + j0};
            }
            if (omm_moe_act(jobs, nj, nrm, S, D, mi, moe, 1) == 0) gpu_done = 1;
            free(pbuf); free(wbuf);
        }
#endif
        if (!gpu_done) {
            /* fase 1 (SERIALE): raccolta hit + caricamento expert — expert_get
             * fa IO e muta la LRU, non e' thread-safe. cap >= topk*? no: i
             * loaded di questo layer non vengono sfrattati (nessun expert_get
             * durante la fase 2). */
            typedef struct { const int8_t *gq, *dq; const float *gs, *ds;
                             const float *gu, *dn; int s; float w; } EHit;
            int nh = 0;
            EHit *hits = malloc((size_t)S*c->topk*sizeof(EHit));
            for (int e = 0; e < c->n_experts; e++) {
                Slot *sl = NULL;
                const float *gu = NULL, *dn = NULL;
                int loaded = 0;
                for (int s = 0; s < S; s++) for (int k = 0; k < c->topk; k++) {
                    if (top[s*c->topk+k] != e) continue;
                    if (!loaded) {
                        if (m->stream) {
                            double te_ = prof_now();
                            expert_get(m, li, e, &sl);
                            g_t_eload += prof_now()-te_;
                        } else { gu = l->eg + (int64_t)e*2*mi*D; dn = l->ed + (int64_t)e*D*mi; }
                        loaded = 1;
                    }
                    hits[nh++] = m->stream
                        ? (EHit){sl->gq, sl->dq, sl->gs, sl->ds, NULL, NULL, s, tw[s*c->topk+k]}
                        : (EHit){NULL, NULL, NULL, NULL, gu, dn, s, tw[s*c->topk+k]};
                }
            }
            /* fase 2: UNA regione parallela a grana grossa (un task = un
             * expert-hit intero, matmul seriali) — il decode passava piu'
             * tempo nei barrier di ~1000 micro-regioni che nei matmul. */
            float *hbuf = falloc((int64_t)nh*D);
            #pragma omp parallel for schedule(dynamic)
            for (int h = 0; h < nh; h++) {
                float a[mi], b[mi];                  /* VLA: scratch per-thread */
                const float *xin = nrm + (int64_t)hits[h].s*D;
                if (hits[h].gq) {
                    matmul_q_st(a, xin, hits[h].gq, hits[h].gs, D, mi);
                    matmul_q_st(b, xin, hits[h].gq + (int64_t)mi*D, hits[h].gs + mi, D, mi);
                    for (int i = 0; i < mi; i++) a[i] = gelu_tanh(a[i])*b[i];
                    matmul_q_st(hbuf + (int64_t)h*D, a, hits[h].dq, hits[h].ds, mi, D);
                } else {
                    matvec_st(a, hits[h].gu, xin, mi, D);                 /* gate */
                    matvec_st(b, hits[h].gu + (int64_t)mi*D, xin, mi, D); /* up */
                    for (int i = 0; i < mi; i++) a[i] = gelu_tanh(a[i])*b[i];
                    matvec_st(hbuf + (int64_t)h*D, hits[h].dn, a, D, mi);
                }
            }
            /* fase 3 (SERIALE): riduzione pesata — piu' hit sullo stesso s */
            for (int h = 0; h < nh; h++) {
                float *mo = moe + (int64_t)hits[h].s*D;
                const float *hb = hbuf + (int64_t)h*D;
                for (int i = 0; i < D; i++) mo[i] += hits[h].w*hb[i];
            }
            free(hbuf); free(hits);
        }
        g_t_ecomp += prof_now()-tp_;
        for (int s = 0; s < S; s++) {
            float *mo = moe + (int64_t)s*D, *f1 = ffn + (int64_t)s*D, *xs = x + (int64_t)s*D;
            rmsnorm(mo, mo, l->post_ffw2, D, c->eps);      /* h2 */
            for (int i = 0; i < D; i++) f1[i] += mo[i];    /* h1 + h2 */
            rmsnorm(f1, f1, l->post_ffw, D, c->eps);
            for (int i = 0; i < D; i++) xs[i] = (xs[i] + f1[i])*l->layer_scalar;
        }
    }
    m->pos = p0 + S;

    if (n_tail > S) n_tail = S;
    double th_ = prof_now();
    float *logits = falloc((int64_t)n_tail*c->vocab);
    for (int t = 0; t < n_tail; t++) {
        float *xs = x + (int64_t)(S-n_tail+t)*D;
        rmsnorm(xs, xs, m->final_norm, D, c->eps);
        float *lo = logits + (int64_t)t*c->vocab;
        if (m->lm8) matmul_q(lo, xs, m->lm8, m->lm8s, D, c->vocab);
        else matvec(lo, m->embed, xs, c->vocab, D);        /* lm_head legato */
        if (c->softcap > 0)
            for (int v = 0; v < c->vocab; v++) lo[v] = c->softcap*tanhf(lo[v]/c->softcap);
    }

    g_t_head += prof_now()-th_;
    free(nrm); free(qb); free(kb); free(t1); free(t2);
    free(ffn); free(moe); free(tmp); free(top); free(tw);
    free(la_top); free(la_tw); free(x);
    return logits;
}

static float *step(Model *m, int id) { return step_batch(m, &id, 1, 1); }

/* ---------- serve mode (PROTOCOL.md v1: JSON lines su stdin/stdout) ---------- */
static int g_eos[8]; static int g_n_eos = 0;
static void load_eos(const char *snap) {
    const char *files[2] = {"generation_config.json", "config.json"};
    for (int fi = 0; fi < 2 && !g_n_eos; fi++) {
        char path[2048]; snprintf(path,sizeof(path),"%s/%s",snap,files[fi]);
        FILE *f = fopen(path,"rb"); if (!f) continue;
        fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
        char *buf=malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
        char *arena=NULL; jval *r=json_parse(buf,&arena);
        jval *v = json_get(r,"eos_token_id");
        if (v && v->t==J_NUM) g_eos[g_n_eos++]=(int)v->num;
        else if (v && v->t==J_ARR) for (int i=0;i<v->len && g_n_eos<8;i++)
            if (v->kids[i]->t==J_NUM) g_eos[g_n_eos++]=(int)v->kids[i]->num;
        free(buf); free(arena);
    }
}
static int is_eos(int t){ for(int i=0;i<g_n_eos;i++) if(g_eos[i]==t) return 1; return 0; }

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec*1e-9;
}

static int argmax(const float *v, int n) {
    int b = 0; for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i; return b;
}

/* bozza n-gram: cerca l'ultimo bigramma nel contesto e copia la
 * continuazione. Ritorna quanti token di bozza (0 = nessuna). */
#define SPEC_K 4
static int ngram_draft(const int *hist, int hlen, int *draft) {
    if (hlen < 3) return 0;
    int a = hist[hlen-2], b = hist[hlen-1];
    for (int q = hlen-3; q >= 1; q--) {
        if (hist[q-1] == a && hist[q] == b) {
            int k = 0;
            while (k < SPEC_K && q+1+k < hlen) { draft[k] = hist[q+1+k]; k++; }
            return k;
        }
    }
    return 0;
}

static int serve(const char *snap) {
    const char *qc = getenv("QCACHE");
    int cap = qc ? atoi(qc) : 24; if (cap < 1) cap = 24;
    /* 128 expert da ~6 MB int8. Sweep misurato su M5 26 GB (2026-07-17),
     * 60 token warm: 16 slot = 4.8 tok/s; 24 = 5.5 tok/s; 32 = 2.7; 48 = 2.8;
     * 64 = swap. Sopra ~24 il memory compressor di macOS comprime gli slot
     * LRU e ogni "hit" paga una decompression fault piu' cara del pread —
     * la cache piccola e CALDA batte la cache grande e compressa. */
    if (cap > 24) { fprintf(stderr, "gemma: QCACHE %d capped to 24\n", cap); cap = 24; }
    /* Speculazione OPT-IN (GEMMA_SPEC=1): i batch di verifica toccano ~3x
     * piu' expert per layer e sfondano la cache da 24 slot (misurato:
     * 41.8s vs 10.9s sui 60 token warm). Torna utile solo con cache grande
     * su macchine con piu' RAM. */
    const char *sp = getenv("GEMMA_SPEC");
    int spec = sp && *sp == '1';
    Model m; model_init(&m, snap, 8192, cap);
    load_eos(snap);
    Cfg *c = &m.c;
    int *hist = malloc(m.max_t*sizeof(int)); int hlen = 0;
    char nb[1024]; snprintf(nb,sizeof(nb),"%s",snap);       /* basename */
    size_t bl=strlen(nb); while (bl>1 && nb[bl-1]=='/') nb[--bl]=0;
    char *base = strrchr(nb,'/'); base = base? base+1 : nb;
    printf("{\"ready\":true,\"model\":\"%s\",\"n_layers\":%d,\"vocab\":%d}\n", base, c->n_layers, c->vocab);
    fflush(stdout);
    char *line=NULL; size_t lcap=0; ssize_t got;
    while ((got=getline(&line,&lcap,stdin)) > 0) {
        if (got==1 && line[0]=='\n') continue;
        char *arena=NULL; jval *r=json_parse(line,&arena);
        jval *jid=json_get(r,"id"); long rid = jid? (long)jid->num : 0;
        jval *js=json_get(r,"stop");
        jval *jids=json_get(r,"ids"), *jn=json_get(r,"n"), *jr=json_get(r,"reset");
        if (js && js->boolean) {
            printf("{\"id\":%ld,\"done\":true,\"n_out\":0,\"prefill_s\":0.0,\"decode_s\":0.0,\"hit\":0.0}\n",rid);
            fflush(stdout); continue;
        }
        int n = jn? (int)jn->num : 0;
        int nids = (jids && jids->t==J_ARR) ? jids->len : 0;
        int reset = jr && (jr->t==J_BOOL ? jr->boolean : jr->num!=0);
        if (reset) { m.pos = 0; hlen = 0; }   /* attention pura: pos limita le letture KV */
        if (m.pos + nids + n + SPEC_K >= m.max_t) {
            printf("{\"id\":%ld,\"error\":\"context full\"}\n",rid); fflush(stdout); continue;
        }
        float *logit=NULL;
        double tp0=now_s();
        if (nids) {
            int *in = malloc(nids*sizeof(int));
            for (int i = 0; i < nids; i++) { in[i]=(int)jids->kids[i]->num; hist[hlen++]=in[i]; }
            logit = step_batch(&m, in, nids, 1);  /* PREFILL batched, union expert */
            free(in);
        }
        double prefill=now_s()-tp0, td0=now_s();
        int n_out=0;
        while (n_out < n && logit) {
            int best = argmax(logit, c->vocab);
            free(logit); logit=NULL;
            printf("{\"id\":%ld,\"tok\":%d}\n",rid,best); fflush(stdout);
            hist[hlen++]=best; n_out++;
            if (is_eos(best)) break;
            if (n_out >= n) break;

            /* decode speculativo: bozza n-gram verificata in un passo batched;
             * greedy-equivalente (si accettano solo argmax combacianti) */
            int draft[SPEC_K];
            int k = spec ? ngram_draft(hist, hlen, draft) : 0;
            if (k > n - n_out - 1) k = n - n_out - 1;
            if (k > 0) {
                int batch[1+SPEC_K]; batch[0]=best;
                for (int i = 0; i < k; i++) batch[1+i]=draft[i];
                int pconf = m.pos;                     /* stato confermato */
                float *ls = step_batch(&m, batch, 1+k, 1+k);
                int acc = 0;
                for (; acc < k; acc++) {
                    int want = argmax(ls + (int64_t)acc*c->vocab, c->vocab);
                    if (want != draft[acc]) break;
                    printf("{\"id\":%ld,\"tok\":%d}\n",rid,want); fflush(stdout);
                    hist[hlen++]=want; n_out++;
                    if (is_eos(want) || n_out >= n) { acc++; break; }
                }
                /* stato valido fino a batch[acc]; le posizioni oltre si
                 * sovrascrivono (attention pura: pos limita le letture) */
                m.pos = pconf + 1 + acc;
                logit = falloc(c->vocab);
                memcpy(logit, ls + (int64_t)acc*c->vocab, c->vocab*sizeof(float));
                free(ls);
                if ((n_out && is_eos(hist[hlen-1])) || n_out >= n) { free(logit); logit=NULL; }
            } else {
                logit = step(&m, best);
            }
        }
        free(logit);
        double tot = m.hits + m.miss;
        double hit = tot ? (double)m.hits/tot : 0.0;
        printf("{\"id\":%ld,\"done\":true,\"n_out\":%d,\"prefill_s\":%.3f,\"decode_s\":%.3f,\"hit\":%.3f}\n",
               rid, n_out, prefill, now_s()-td0, hit);
        fflush(stdout);
        fprintf(stderr, "phases: attn=%.1fs dense=%.1fs route=%.1fs eload=%.1fs ecomp=%.1fs head=%.1fs\n",
                g_t_attn, g_t_dense, g_t_route, g_t_eload, g_t_ecomp-g_t_eload, g_t_head);
        g_t_attn=g_t_dense=g_t_route=g_t_eload=g_t_ecomp=g_t_head=0;
    }
    return 0;
}

/* ---------- validation main (TF + greedy vs ref_gemma.json) ---------- */
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
    const char *refpath = argc>1 ? argv[1] : "oracle/ref_gemma.json";
    FILE *f=fopen(refpath,"rb"); if(!f){perror(refpath);return 1;}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf=malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    char *arena=NULL; jval *ref=json_parse(buf,&arena);
    int np,nf,ntf;
    int *prompt=read_ints(ref,"prompt_ids",&np);
    int *full  =read_ints(ref,"full_ids",&nf);
    int *tf    =read_ints(ref,"tf_argmax",&ntf);
    int n_new = nf-np;

    Model m; model_init(&m, snap, nf+8, 16);
    int nfull_layers = 0;
    for (int i = 0; i < m.c.n_layers; i++) nfull_layers += m.c.is_full[i];
    printf("gemma4 engine: %d layers, %d/%d full-attn, %d experts topk %d%s\n",
           m.c.n_layers, nfull_layers, m.c.n_layers, m.c.n_experts, m.c.topk,
           m.stream ? " (int8 streaming)" : " (f32 resident)");

    /* teacher forcing IN UN PASSO BATCHED: valida attention/masking/union
     * del percorso batched, non solo il token-per-token */
    int gdbg = getenv("GDBG") != NULL;
    int tf_match = 0, ncheck = ntf < nf ? ntf : nf;
    m.pos = 0;
    float *ls = step_batch(&m, full, nf, nf);
    if (gdbg) fprintf(stderr, "tf_c : ");
    for (int i = 0; i < ncheck; i++) {
        int a = argmax(ls + (int64_t)i*m.c.vocab, m.c.vocab);
        if (gdbg) fprintf(stderr, "%d ", a);
        if (a == tf[i]) tf_match++;
    }
    if (gdbg) fprintf(stderr, "\n");
    free(ls);
    printf("teacher-forcing (batched): %d/%d argmax match\n", tf_match, ncheck);

    /* greedy: prefill batched del prompt, poi decode token-per-token */
    m.pos = 0;
    float *logit = step_batch(&m, prompt, np, 1);
    int match = 0;
    printf("reference: "); for (int i = np; i < nf; i++) printf("%d ", full[i]);
    printf("\nengine   : ");
    for (int i = 0; i < n_new; i++) {
        int best = argmax(logit, m.c.vocab);
        printf("%d ", best); fflush(stdout);
        if (best == full[np+i]) match++;
        free(logit); logit = NULL;
        if (i < n_new-1) logit = step(&m, best);
    }
    free(logit);
    printf("\ngreedy match: %d/%d %s\n", match, n_new, match==n_new?"EXACT":"");
    return (match==n_new && tf_match==ncheck) ? 0 : 1;
}
