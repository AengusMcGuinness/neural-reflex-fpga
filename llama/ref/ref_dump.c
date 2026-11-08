// Reference dumper built on karpathy/llama2.c run.c itself (compiled exactly like `make run`).
// Produces everything the FPGA kernel and its testbench need to be checked against the CPU:
//   <out>.prompt   int32 prompt tokens (BOS included), via run.c encode()
//   <out>.rope     float [seq_len][head_size/2][2] cos/sin, run.c's RoPE expression
//   <out>.argmax   int32 sample_argmax(logits) at every position
//   <out>.logits   float logits at every position (steps x vocab)
// and prints the generated text (must equal `./run ckpt -z tok -t 0 -n steps -i prompt`).
//
// build: gcc -O3 -DTESTING -o ref_dump ref_dump.c -lm   (with run.c on the include path)
#ifndef TESTING
#define TESTING
#endif
#include "run.c"

static void write_or_die(const char *path, const void *buf, size_t sz, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(buf, sz, n, f) != n) { fprintf(stderr, "write %s failed\n", path); exit(1); }
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s checkpoint tokenizer prompt steps outprefix\n", argv[0]);
        return 1;
    }
    char *ckpt = argv[1], *tokp = argv[2], *prompt = argv[3], *out = argv[5];
    int steps = atoi(argv[4]);

    Transformer t;
    build_transformer(&t, ckpt);
    Config *p = &t.config;
    if (steps <= 0 || steps > p->seq_len) steps = p->seq_len;
    Tokenizer tok;
    build_tokenizer(&tok, tokp, p->vocab_size);

    int n_prompt = 0;
    int *ptoks = malloc((strlen(prompt) + 3) * sizeof(int));
    encode(&tok, prompt, 1, 0, ptoks, &n_prompt);

    char path[4096];
    snprintf(path, sizeof path, "%s.prompt", out);
    write_or_die(path, ptoks, sizeof(int), n_prompt);

    // RoPE table: identical expression to forward(); index by head_dim/2.
    int head_size = p->dim / p->n_heads;
    int half = head_size / 2;
    float *rope = malloc(sizeof(float) * p->seq_len * half * 2);
    for (int pos = 0; pos < p->seq_len; pos++) {
        for (int head_dim = 0; head_dim < head_size; head_dim += 2) {
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            rope[(pos * half + head_dim / 2) * 2 + 0] = fcr;
            rope[(pos * half + head_dim / 2) * 2 + 1] = fci;
        }
    }
    snprintf(path, sizeof path, "%s.rope", out);
    write_or_die(path, rope, sizeof(float), (size_t)p->seq_len * half * 2);

    // generate() from run.c, greedy, recording argmax/logits at every position
    int *amax = malloc(sizeof(int) * steps);
    float *lg = malloc(sizeof(float) * (size_t)steps * p->vocab_size);
    int token = ptoks[0], pos = 0, next;
    while (pos < steps) {
        float *logits = forward(&t, token, pos);
        amax[pos] = sample_argmax(logits, p->vocab_size);
        memcpy(lg + (size_t)pos * p->vocab_size, logits, sizeof(float) * p->vocab_size);
        next = (pos < n_prompt - 1) ? ptoks[pos + 1] : amax[pos];
        pos++;
        if (next == 1) break;
        safe_printf(decode(&tok, token, next));
        token = next;
    }
    printf("\n");
    snprintf(path, sizeof path, "%s.argmax", out);
    write_or_die(path, amax, sizeof(int), pos);
    snprintf(path, sizeof path, "%s.logits", out);
    write_or_die(path, lg, sizeof(float), (size_t)pos * p->vocab_size);
    fprintf(stderr, "positions=%d prompt_tokens=%d dim=%d hidden=%d layers=%d heads=%d kv_heads=%d "
                    "vocab=%d seq=%d\n", pos, n_prompt, p->dim, p->hidden_dim, p->n_layers,
            p->n_heads, p->n_kv_heads, p->vocab_size, p->seq_len);
    return 0;
}
