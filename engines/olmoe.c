/* Motore di inferenza OLMoE in C puro, con EXPERT-STREAMING dal disco.
 * Porting del motore Python (engine.py). Obiettivo Stadio A: produrre gli STESSI
 * token id del riferimento (ref.json) -> valida il core prima di scalare a GLM-5.2.
 *
 * Densa (embed, attn, router, norme, lm_head) residente in RAM (float32).
 * Expert letti dal disco on-demand via pread+fadvise(DONTNEED), cache LRU per-layer.
 * Matmul multi-thread con OpenMP (niente BLAS).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#if defined(__APPLE__) || defined(__linux__)
#include <sys/resource.h>
#endif
#include "st.h"

#ifdef OM_METAL
#include "om_metal.h"
static int g_omm = -1;      /* -1 unset, 0 off, 1 GPU ready. OM_METAL=0 disables */
static int omm_on(void) {
    if (g_omm < 0) {
        const char *e = getenv("OM_METAL");
        g_omm = (e && *e == '0') ? 0 : omm_init();
    }
    return g_omm;
}
/* expert-slot memory must be GPU-visible when the backend is live */
static void *slot_alloc(size_t n) { return omm_on() ? omm_alloc(n) : malloc(n); }
#else
static void *slot_alloc(size_t n) { return malloc(n); }
#endif

/* ---------- config ---------- */
typedef struct {
    int hidden, n_layers, n_heads, n_kv_heads, head_dim;
    int n_experts, topk, inter, vocab;
    float theta, eps; int norm_topk;
} Cfg;

/* ---------- pesi densi per-layer ---------- */
typedef struct {
    float *in_ln, *post_ln, *q, *k, *v, *o, *qn, *kn, *gate;
} Layer;

/* ---------- cache LRU degli expert (pesi QUANTIZZATI) ----------
 * Ogni weight [out,in] tenuto come int8 (per-riga) + scala float per riga.
 * Cosi' la RAM-cache scende da 4 byte/param (f32) a 1 byte/param: e' il
 * meccanismo che fa stare GLM-5.2 nei 15 GB. dequant-on-use nel matmul. */
typedef struct { int eid; int8_t *g, *u, *d; float *gs, *us, *ds; uint64_t used; } Slot;
typedef struct { Slot *slots; int n, cap; } LCache;

typedef struct {
    Cfg c;
    shards S;
    int quant_bits;        /* bit di quantizzazione degli expert (2..8); storage int8, niente f32 (#134) */
    float *embed, *lm_head, *final_norm;
    Layer *L;
    LCache *cache;          /* [n_layers] */
    uint64_t clock, hits, miss;
    /* kv-cache per-layer: K,V come [H * maxT * head_dim] */
    float **K, **V; int kv_len, max_t;
    double dense_load_s;
} Model;

/* ---------- utility ---------- */
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0); }
static float *falloc(int64_t n) { float *p = malloc(n*sizeof(float)); if(!p){fprintf(stderr,"OOM %ld\n",(long)n);exit(1);} return p; }

/* y[S,O] = x[S,I] @ W^T,  W e' [O,I] row-major */
static void matmul(float *y, const float *x, const float *W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float acc = 0.f;
            for (int i = 0; i < I; i++) acc += xs[i] * w[i];
            y[(int64_t)s * O + o] = acc;
        }
    }
}

/* y[1,O] = x[1,I] @ W^T con W quantizzato: q[O,I] int8 + scala per riga.
 * W[o,i] ~= q[o,i]*scale[o]  ->  y[o] = scale[o] * sum_i x[i]*q[o,i]. */
#if defined(__ARM_NEON)
#include <arm_neon.h>
/* dot int8x16 -> int32 (un blocco Q8_0 da 16) */
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

static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
#if defined(__ARM_NEON)
    /* via int8xint8 (stile glm.c IDOT): attivazione quantizzata Q8_0 (scala per
     * blocco di 32) una volta sola, poi dot interi NEON per ogni riga di W.
     * IDOT=0 per la via scalare f32. */
    static int idot = -1;
    if (idot < 0) { const char *e = getenv("IDOT"); idot = !(e && *e == '0'); }
    if (idot && I % 16 == 0) {
        int nb = I / 16;
        int8_t xi[4096]; float xs[256];          /* I<=4096: OLMoE hidden 2048, inter 1024 */
        if (I <= 4096) {
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

/* quantizza un weight f32 [O,I] -> int8 q[O,I] + scala[O], simmetrica per riga.
 * Replica quant_dequant() del Python: scale = amax(|w|, riga)/qmax, q = round(w/scale). */
static void quantize_rows(const float *w, int8_t *q, float *scale, int O, int I, int bits) {
    int qmax = (1 << (bits - 1)) - 1;     /* 8->127, 4->7, 2->1 */
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *wr = w + (int64_t)o * I;
        float amax = 0.f; for (int i = 0; i < I; i++) { float a = fabsf(wr[i]); if (a > amax) amax = a; }
        float s = amax / qmax; if (s < 1e-8f) s = 1e-8f;
        scale[o] = s;
        int8_t *qr = q + (int64_t)o * I;
        for (int i = 0; i < I; i++) {
            int v = (int)lrintf(wr[i] / s);
            if (v >  qmax) v =  qmax;
            if (v < -qmax-1) v = -qmax-1;
            qr[i] = (int8_t)v;
        }
    }
}

/* rmsnorm su una riga di lunghezza D, in-place su out (out puo' essere == x) */
static void rmsnorm_row(float *out, const float *x, const float *w, int D, float eps) {
    double ms = 0; for (int i = 0; i < D; i++) ms += (double)x[i]*x[i];
    float r = 1.f / sqrtf((float)(ms / D) + eps);
    for (int i = 0; i < D; i++) out[i] = x[i] * r * w[i];
}

static void softmax_row(float *x, int n) {
    float m = -1e30f; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* ---------- caricamento ---------- */
static void load_cfg(Cfg *c, const char *snap) {
    char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if(!f){perror(path);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf = malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    char *arena=NULL; jval *r = json_parse(buf, &arena);
    c->hidden    = (int)json_get(r,"hidden_size")->num;
    c->n_layers  = (int)json_get(r,"num_hidden_layers")->num;
    c->n_heads   = (int)json_get(r,"num_attention_heads")->num;
    c->n_kv_heads= (int)json_get(r,"num_key_value_heads")->num;
    c->n_experts = (int)json_get(r,"num_experts")->num;
    c->topk      = (int)json_get(r,"num_experts_per_tok")->num;
    c->inter     = (int)json_get(r,"intermediate_size")->num;
    c->vocab     = (int)json_get(r,"vocab_size")->num;
    c->head_dim  = c->hidden / c->n_heads;
    jval *th = json_get(r,"rope_theta");  c->theta = th ? (float)th->num : 10000.f;
    jval *ep = json_get(r,"rms_norm_eps"); c->eps   = ep ? (float)ep->num : 1e-5f;
    jval *nt = json_get(r,"norm_topk_prob"); c->norm_topk = (nt && nt->t==J_BOOL) ? nt->boolean : 0;
    free(buf); free(arena);
}

static float *load_t(Model *m, const char *name) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) { fprintf(stderr, "missing %s\n", name); exit(1); }
    float *p = falloc(n);
    st_read_f32(&m->S, name, p, 0);   /* densa: niente DONTNEED, resta residente */
    return p;
}

static void model_init(Model *m, const char *snap, int cap, int bits) {
    memset(m, 0, sizeof(*m));
    m->quant_bits = bits;
    load_cfg(&m->c, snap);
    st_init(&m->S, snap);
    Cfg *c = &m->c;
    double t0 = now_s();
    m->embed      = load_t(m, "model.embed_tokens.weight");
    m->lm_head    = load_t(m, "lm_head.weight");
    m->final_norm = load_t(m, "model.norm.weight");
    m->L = calloc(c->n_layers, sizeof(Layer));
    char nm[256];
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        #define LD(field, suffix) snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); l->field = load_t(m,nm)
        LD(in_ln,  "input_layernorm.weight");
        LD(post_ln,"post_attention_layernorm.weight");
        LD(q, "self_attn.q_proj.weight"); LD(k, "self_attn.k_proj.weight");
        LD(v, "self_attn.v_proj.weight"); LD(o, "self_attn.o_proj.weight");
        LD(qn,"self_attn.q_norm.weight"); LD(kn,"self_attn.k_norm.weight");
        LD(gate, "mlp.gate.weight");
        #undef LD
    }
    m->cache = calloc(c->n_layers, sizeof(LCache));
    for (int i = 0; i < c->n_layers; i++) { m->cache[i].cap = cap; m->cache[i].slots = calloc(cap, sizeof(Slot)); }
    m->dense_load_s = now_s() - t0;
}

/* legge un weight dal disco (streaming) e lo quantizza in q[O,I]+scale[O].
 * Container pre-quantizzato (convert_olmoe.py: int8 + scale f32 in "name.qs"):
 * lettura raw diretta, meta' I/O e zero quantize_rows a runtime. */
static void load_expert_w(Model *m, const char *name, int8_t *q, float *scale, int O, int I, float *tmp) {
    st_tensor *t = st_find(&m->S, name);
    if (t && t->dtype == 3) {                    /* I8/U8: container colibri */
        char qs[300]; snprintf(qs, sizeof(qs), "%s.qs", name);
        st_read_raw(&m->S, name, q, 1);          /* int8 grezzi, gia' quantizzati */
        st_read_f32(&m->S, qs, scale, 1);        /* scale per riga */
        return;
    }
    st_read_f32(&m->S, name, tmp, 1);            /* pread + fadvise DONTNEED */
    quantize_rows(tmp, q, scale, O, I, m->quant_bits);
}

/* ---------- cache expert: ritorna i pesi quantizzati (q+scale) da cache o disco ---------- */
static void expert_get(Model *m, int layer, int eid, Slot **out) {
    LCache *lc = &m->cache[layer];
    for (int i = 0; i < lc->n; i++) if (lc->slots[i].eid == eid) {
        m->hits++; lc->slots[i].used = ++m->clock; *out = &lc->slots[i]; return;
    }
    m->miss++;
    Cfg *c = &m->c;
    int64_t ng = (int64_t)c->inter * c->hidden, nd = (int64_t)c->hidden * c->inter;
    Slot *s;
    if (lc->n < lc->cap) {
        s = &lc->slots[lc->n++];
        s->g = slot_alloc(ng); s->u = slot_alloc(ng); s->d = slot_alloc(nd);
        s->gs = slot_alloc(c->inter*sizeof(float)); s->us = slot_alloc(c->inter*sizeof(float)); s->ds = slot_alloc(c->hidden*sizeof(float));
    } else { int lru = 0; for (int i = 1; i < lc->n; i++) if (lc->slots[i].used < lc->slots[lru].used) lru = i; s = &lc->slots[lru]; }
    float *tmp = falloc(ng > nd ? ng : nd);
    char nm[256];
    snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.gate_proj.weight",layer,eid); load_expert_w(m,nm,s->g,s->gs,c->inter,c->hidden,tmp);
    snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.up_proj.weight",  layer,eid); load_expert_w(m,nm,s->u,s->us,c->inter,c->hidden,tmp);
    snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.down_proj.weight",layer,eid); load_expert_w(m,nm,s->d,s->ds,c->hidden,c->inter,tmp);
    free(tmp);
    s->eid = eid; s->used = ++m->clock;
    *out = s;
}

/* ---------- RoPE su un vettore di una testa (head_dim) a posizione assoluta pos ---------- */
static void rope_head(float *x, int pos, const Cfg *c) {
    int h = c->head_dim / 2;
    for (int j = 0; j < h; j++) {
        float inv = powf(c->theta, -2.0f * j / c->head_dim);
        float ang = pos * inv, cs = cosf(ang), sn = sinf(ang);
        float a = x[j], b = x[j+h];
        x[j]   = a*cs - b*sn;
        x[j+h] = b*cs + a*sn;
    }
}

/* GEMM densa: GPU tensor ops per il prefill (S>=8), CPU matmul altrimenti */
static void dense_mm(float *y, const float *x, const float *W, int S, int I_, int O_) {
#ifdef OM_METAL
    if (S >= 8 && omm_on() && omm_dense(W, O_, I_, x, S, y) == 0) return;
#endif
    matmul(y, x, W, S, I_, O_);
}

/* attenzione sui token nuovi x[S,hidden]; pos_base = posizione assoluta del primo token nuovo */
static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out) {
    Cfg *c = &m->c; int H = c->n_heads, hd = c->head_dim, D = c->hidden;
    float *q = falloc((int64_t)S*D), *k = falloc((int64_t)S*D), *vv = falloc((int64_t)S*D);
    dense_mm(q, x, l->q, S, D, D);
    dense_mm(k, x, l->k, S, D, D);
    dense_mm(vv, x, l->v, S, D, D);
    /* qk-norm sull'intero vettore hidden, poi RoPE per testa */
    for (int s = 0; s < S; s++) {
        rmsnorm_row(q + (int64_t)s*D, q + (int64_t)s*D, l->qn, D, c->eps);
        rmsnorm_row(k + (int64_t)s*D, k + (int64_t)s*D, l->kn, D, c->eps);
        int pos = pos_base + s;
        for (int hh = 0; hh < H; hh++) { rope_head(q + (int64_t)s*D + hh*hd, pos, c); rope_head(k + (int64_t)s*D + hh*hd, pos, c); }
    }
    /* scrive k,v nella kv-cache alle posizioni pos_base..pos_base+S-1 */
    for (int s = 0; s < S; s++) for (int hh = 0; hh < H; hh++) {
        int t = pos_base + s;
        memcpy(m->K[layer] + ((int64_t)hh*m->max_t + t)*hd, k + (int64_t)s*D + hh*hd, hd*sizeof(float));
        memcpy(m->V[layer] + ((int64_t)hh*m->max_t + t)*hd, vv + (int64_t)s*D + hh*hd, hd*sizeof(float));
    }
    int Tk = pos_base + S;             /* numero di key totali disponibili */
    float scale = 1.f / sqrtf((float)hd);
    float *ctx = falloc((int64_t)S*D);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int hh = 0; hh < H; hh++) {
        for (int s = 0; s < S; s++) {
            int qpos = pos_base + s;
            const float *qv = q + (int64_t)s*D + hh*hd;
            float sc[4096];
            for (int t = 0; t <= qpos; t++) {          /* causale: t <= qpos */
                const float *kv = m->K[layer] + ((int64_t)hh*m->max_t + t)*hd;
                float acc = 0; for (int dd = 0; dd < hd; dd++) acc += qv[dd]*kv[dd];
                sc[t] = acc * scale;
            }
            softmax_row(sc, qpos+1);
            float *cx = ctx + (int64_t)s*D + hh*hd;
            for (int dd = 0; dd < hd; dd++) cx[dd] = 0;
            for (int t = 0; t <= qpos; t++) {
                const float *vrow = m->V[layer] + ((int64_t)hh*m->max_t + t)*hd;
                float a = sc[t];
                for (int dd = 0; dd < hd; dd++) cx[dd] += a * vrow[dd];
            }
        }
    }
    (void)Tk;
    dense_mm(out, ctx, l->o, S, D, D);
    free(q); free(k); free(vv); free(ctx);
}

/* MoE sui token x[S,hidden] -> out[S,hidden] */
static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out) {
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts, K = c->topk, I = c->inter;
    float *logits = falloc((int64_t)S*E);
    matmul(logits, x, l->gate, S, D, E);
    memset(out, 0, (int64_t)S*D*sizeof(float));
    /* fase 1: routing per tutte le posizioni */
    int *sidx = malloc((size_t)S*K*sizeof(int)); float *sval = malloc((size_t)S*K*sizeof(float));
    for (int s = 0; s < S; s++) {
        float *pr = logits + (int64_t)s*E;
        softmax_row(pr, E);
        int *idx = sidx + (size_t)s*K; float *val = sval + (size_t)s*K;
        for (int kk = 0; kk < K; kk++) {
            int best = -1; float bv = -1e30f;
            for (int e = 0; e < E; e++) {
                int taken = 0; for (int j = 0; j < kk; j++) if (idx[j]==e){taken=1;break;}
                if (!taken && pr[e] > bv) { bv = pr[e]; best = e; }
            }
            idx[kk] = best; val[kk] = bv;
        }
        if (c->norm_topk) { float sm=0; for(int kk=0;kk<K;kk++) sm+=val[kk]; for(int kk=0;kk<K;kk++) val[kk]/=sm; }
    }
#ifdef OM_METAL
    /* fase 2-GPU (prefill S>=8): batch-union per expert, SwiGLU su tensor ops */
    if (S >= 8 && omm_on()) {
        int *epos = malloc((size_t)S*K*sizeof(int)); float *ew = malloc((size_t)S*K*sizeof(float));
        omm_job jobs[64]; int nj = 0, cur = 0;
        for (int e = 0; e < E; e++) {
            int m0 = cur;
            for (int s = 0; s < S; s++) for (int kk = 0; kk < K; kk++)
                if (sidx[(size_t)s*K+kk] == e) { epos[cur] = s; ew[cur] = sval[(size_t)s*K+kk]; cur++; }
            if (cur == m0) continue;
            Slot *sl; expert_get(m, layer, e, &sl);
            jobs[nj++] = (omm_job){sl->g, sl->u, sl->d, sl->gs, sl->us, sl->ds,
                                   cur - m0, epos + m0, ew + m0};
        }
        int rc = omm_moe(jobs, nj, x, S, D, I, out);
        free(epos); free(ew);
        if (rc == 0) { free(logits); free(sidx); free(sval); return; }
        /* rc!=0: fall through alla via CPU */
    }
#endif
    /* fase 2-CPU */
    float *g = falloc(I), *u = falloc(I), *hh = falloc(D);
    for (int s = 0; s < S; s++) {
        const float *xs = x + (int64_t)s*D;
        for (int kk = 0; kk < K; kk++) {
            Slot *e; expert_get(m, layer, sidx[(size_t)s*K+kk], &e);
            matmul_q(g, xs, e->g, e->gs, D, I);     /* gate_proj [I,D] */
            matmul_q(u, xs, e->u, e->us, D, I);     /* up_proj   [I,D] */
            for (int i = 0; i < I; i++) { float gv = g[i]; g[i] = (gv / (1.f + expf(-gv))) * u[i]; }
            matmul_q(hh, g, e->d, e->ds, I, D);     /* down_proj [D,I] */
            float w = sval[(size_t)s*K+kk];
            float *os = out + (int64_t)s*D;
            for (int d = 0; d < D; d++) os[d] += w * hh[d];
        }
    }
    free(g); free(u); free(hh);
    free(logits); free(sidx); free(sval);
}

/* un passo: token nuovi ids[S] a posizione pos_base. Ritorna logits dell'ultimo token (malloc'd). */
static float *step(Model *m, const int *ids, int S, int pos_base) {
    Cfg *c = &m->c; int D = c->hidden;
    float *x = falloc((int64_t)S*D);
    for (int s = 0; s < S; s++) memcpy(x + (int64_t)s*D, m->embed + (int64_t)ids[s]*D, D*sizeof(float));
    float *nrm = falloc((int64_t)S*D), *tmp = falloc((int64_t)S*D);
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->in_ln, D, c->eps);
        attention(m, l, i, nrm, S, pos_base, tmp);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->post_ln, D, c->eps);
        moe(m, l, i, nrm, S, tmp);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
    }
    m->kv_len = pos_base + S;
    /* solo l'ultimo token -> logits */
    float *last = falloc(D);
    rmsnorm_row(last, x + (int64_t)(S-1)*D, m->final_norm, D, c->eps);
    float *logit = falloc(c->vocab);
    matmul(logit, last, m->lm_head, 1, D, c->vocab);
    free(x); free(nrm); free(tmp); free(last);
    return logit;
}

/* generazione greedy. prompt[np] -> riempie out[np+n_new] */
static void generate(Model *m, const int *prompt, int np, int n_new, int *out) {
    Cfg *c = &m->c;
    m->max_t = np + n_new;
    m->K = calloc(c->n_layers, sizeof(float*)); m->V = calloc(c->n_layers, sizeof(float*));
    for (int i = 0; i < c->n_layers; i++) {
        m->K[i] = falloc((int64_t)c->n_heads * m->max_t * c->head_dim);
        m->V[i] = falloc((int64_t)c->n_heads * m->max_t * c->head_dim);
    }
    for (int i = 0; i < np; i++) out[i] = prompt[i];
    double tp0 = now_s();
    float *logit = step(m, prompt, np, 0);          /* PREFILL */
    double tp = now_s() - tp0;
    if (np > 1) fprintf(stderr, "prefill: %d tok in %.2fs (%.1f tok/s)\n", np, tp, np/tp);
    int len = np;
    for (int s = 0; s < n_new; s++) {
        int best = 0; float bv = logit[0];
        for (int i = 1; i < c->vocab; i++) if (logit[i] > bv) { bv = logit[i]; best = i; }
        free(logit);
        out[len++] = best;
        if (s == n_new - 1) break;
        int one = best;
        logit = step(m, &one, 1, len - 1);          /* DECODE */
    }
}

/* ---------- serve mode (PROTOCOL.md v1: JSON lines su stdin/stdout) ---------- */
static int g_eos[8]; static int g_n_eos = 0;
static void load_eos(const char *snap) {
    /* generation_config.json se c'e', altrimenti config.json */
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
    }
}
static int is_eos(int t){ for(int i=0;i<g_n_eos;i++) if(g_eos[i]==t) return 1; return 0; }

static int serve(const char *snap) {
    const char *qc = getenv("QCACHE");
    int cap = qc ? atoi(qc) : 16; if (cap < 1) cap = 16;
    Model m; model_init(&m, snap, cap, 8);   /* container colibri: expert gia' int8 */
    load_eos(snap);
    Cfg *c = &m.c;
    /* KV allocato una volta per la sessione serve (in run mode lo fa generate) */
    m.max_t = 8192;
    m.K = calloc(c->n_layers, sizeof(float*)); m.V = calloc(c->n_layers, sizeof(float*));
    for (int i = 0; i < c->n_layers; i++) {
        m.K[i] = falloc((int64_t)c->n_heads * m.max_t * c->head_dim);
        m.V[i] = falloc((int64_t)c->n_heads * m.max_t * c->head_dim);
    }
    char nb[1024]; snprintf(nb,sizeof(nb),"%s",snap);       /* basename, slash finali tolti */
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
        if (js && js->boolean) {                             /* stop tra un token e l'altro: gia' fermi */
            printf("{\"id\":%ld,\"done\":true,\"n_out\":0,\"prefill_s\":0.0,\"decode_s\":0.0,\"hit\":0.0}\n",rid);
            fflush(stdout); continue;
        }
        int n = jn? (int)jn->num : 0;
        int nids = (jids && jids->t==J_ARR) ? jids->len : 0;
        int reset = jr && (jr->t==J_BOOL ? jr->boolean : jr->num!=0);
        if (reset) m.kv_len = 0;    /* attention pura: pos limita le letture del KV */
        if (m.kv_len + nids + n >= m.max_t) {
            printf("{\"id\":%ld,\"error\":\"context full\"}\n",rid); fflush(stdout); continue;
        }
        float *logit=NULL;
        double tp0=now_s();
        if (nids) {                                          /* PREFILL in un colpo (step batcha) */
            int *ids = malloc(nids*sizeof(int));
            for (int i = 0; i < nids; i++) ids[i]=(int)jids->kids[i]->num;
            logit = step(&m, ids, nids, m.kv_len);
            free(ids);
        }
        double prefill=now_s()-tp0, td0=now_s();
        int n_out=0;
        for (int i = 0; i < n && logit; i++) {
            int best=0; float bv=logit[0];
            for (int v = 1; v < c->vocab; v++) if (logit[v]>bv){bv=logit[v];best=v;}
            free(logit); logit=NULL;
            printf("{\"id\":%ld,\"tok\":%d}\n",rid,best); fflush(stdout);
            n_out++;
            int one = best;
            logit = step(&m, &one, 1, m.kv_len);             /* anche i generati avanzano lo stato */
            if (is_eos(best)) break;
        }
        free(logit);
        double tot = m.hits + m.miss;
        double hit = tot ? (double)m.hits/tot : 0.0;
        printf("{\"id\":%ld,\"done\":true,\"n_out\":%d,\"prefill_s\":%.3f,\"decode_s\":%.3f,\"hit\":%.3f}\n",
               rid, n_out, prefill, now_s()-td0, hit);
        fflush(stdout);
    }
    return 0;
}

/* ---------- lettura ref.json ---------- */
static int *read_int_array(jval *o, const char *key, int *n_out) {
    jval *a = json_get(o, key);
    int *r = malloc(a->len * sizeof(int));
    for (int i = 0; i < a->len; i++) r[i] = (int)a->kids[i]->num;
    *n_out = a->len; return r;
}

int main(int argc, char **argv) {
    const char *snap = getenv("SNAP");
    if (!snap) { fprintf(stderr, "set SNAP=<snapshot directory>\n"); return 1; }
    if (argc>1 && !strcmp(argv[1],"--serve")) return serve(snap);
    int cap  = argc > 1 ? atoi(argv[1]) : 16;
    int bits = argc > 2 ? atoi(argv[2]) : 8;
    if (bits < 2 || bits > 8) {   /* expert storage is int8_t: bits>8 truncates in quantize_rows (#134). f32 mode is not implemented here — int8 is already token-exact vs the oracle. */
        fprintf(stderr, "quant_bits must be 2..8 (got %d); OLMoE experts are int8-backed, no f32 mode\n", bits);
        return 1;
    }
    const char *refpath = argc > 3 ? argv[3] : "ref.json";

    FILE *f = fopen(refpath, "rb"); if(!f){perror(refpath);return 1;}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf=malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    char *arena=NULL; jval *ref = json_parse(buf, &arena);
    int np, nfull; int *prompt = read_int_array(ref,"prompt_ids",&np); int *full = read_int_array(ref,"full_ids",&nfull);
    int n_new = nfull - np;

    printf("== Streaming C engine, cache = %d experts/layer, experts @ %d-bit ==\n", cap, bits);
    Model m; model_init(&m, snap, cap, bits);
    printf("resident weights loaded in %.1fs | RSS after load: %.2f GB\n", m.dense_load_s, rss_gb());

    int *out = malloc((np + n_new) * sizeof(int));
    double t = now_s();
    generate(&m, prompt, np, n_new, out);
    double dt = now_s() - t;

    int match = 0;
    printf("\nReference: ");  for (int i=np;i<nfull;i++) printf("%d ", full[i]);
    printf("\nC engine : ");  for (int i=np;i<nfull;i++) { printf("%d ", out[i]); if (out[i]==full[i]) match++; }
    printf("\nMatching tokens: %d/%d\n", match, n_new);
    double tot = m.hits + m.miss;
    printf("\nPEAK RSS: %.2f GB\n", rss_gb());
    printf("Expert cache hit rate: %.1f%%  (hit=%llu miss=%llu)\n", tot?100.0*m.hits/tot:0.0,
           (unsigned long long)m.hits, (unsigned long long)m.miss);
    printf("Speed: %.2f tok/s (%.1fs for %d tokens)\n", n_new/dt, dt, n_new);
    free(buf); free(arena);
    return 0;
}
