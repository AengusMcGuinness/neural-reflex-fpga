// C-sim / co-sim testbench: runs run.c's greedy generate() loop on llama_forward and
// compares against ref_dump output (argmax at every position + logits).
// Memory layout is the same one the board uses (see llama_layout()).
// usage: tb <checkpoint.bin> <refprefix> [max_positions]
#include "llama_kernel.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

template <typename T> static std::vector<T> slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<T> v(sz / sizeof(T));
    if (fread(v.data(), sizeof(T), v.size(), f) != v.size()) { fprintf(stderr, "short read %s\n", path); exit(2); }
    fclose(f);
    return v;
}

static int align16(int x) { return (x + 15) & ~15; }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s ckpt refprefix [max_positions]\n", argv[0]); return 2; }
    std::string ref = argv[2];
    int max_pos = argc > 3 ? atoi(argv[3]) : 1 << 30;

    std::vector<float> file = slurp<float>(argv[1]); // whole checkpoint, header included
    const int *hdr = reinterpret_cast<const int *>(file.data());
    int dim = hdr[0], hidden = hdr[1], layers = hdr[2], heads = hdr[3], kvh = hdr[4];
    int vocab = hdr[5] < 0 ? -hdr[5] : hdr[5], seq = hdr[6], shared = hdr[5] > 0;
    int head_size = dim / heads, kv_dim = dim * kvh / heads, kv_mul = heads / kvh;
    std::vector<int> prompt = slurp<int>((ref + ".prompt").c_str());
    std::vector<float> rope = slurp<float>((ref + ".rope").c_str());
    std::vector<int> amax = slurp<int>((ref + ".argmax").c_str());
    std::vector<float> rlog = slurp<float>((ref + ".logits").c_str());

    // layout (float offsets): checkpoint @0 (weights @7), kv cache, rope table, logits
    int w_off = 7;
    int kv_off = align16((int)file.size());
    int rope_off = kv_off + 2 * layers * seq * kv_dim;
    int logits_off = align16(rope_off + seq * head_size);
    std::vector<float> mem(logits_off + vocab, 0.0f);
    printf("mem words=%zu (kv_off=%d rope_off=%d logits_off=%d)\n", mem.size(), kv_off, rope_off, logits_off);
    memcpy(mem.data(), file.data(), file.size() * sizeof(float));
    memcpy(mem.data() + rope_off, rope.data(), rope.size() * sizeof(float));

    int n = (int)amax.size() < max_pos ? (int)amax.size() : max_pos;
    int token = prompt[0], mismatches = 0;
    double worst = 0.0;
    long exact = 0, total = 0;
    for (int pos = 0; pos < n; pos++) {
        int hw = llama_forward(mem.data(), w_off, kv_off, rope_off, logits_off, dim, hidden, layers,
                               heads, kvh, vocab, seq, shared, head_size, kv_dim, kv_mul, token, pos, 1);
        const float *r = &rlog[(size_t)pos * vocab];
        const float *lg = &mem[logits_off];
        double d = 0.0;
        for (int i = 0; i < vocab; i++) {
            double e = std::fabs((double)lg[i] - (double)r[i]);
            if (e > d) d = e;
            exact += lg[i] == r[i];
        }
        total += vocab;
        if (d > worst) worst = d;
        if (hw != amax[pos]) {
            mismatches++;
            printf("MISMATCH pos %d: fpga %d ref %d (max |dlogit| %.3g)\n", pos, hw, amax[pos], d);
        }
        int next = pos < (int)prompt.size() - 1 ? prompt[pos + 1] : hw;
        if (next == 1) break;
        token = next;
    }
    printf("positions=%d argmax_mismatches=%d max_abs_logit_diff=%.6g bit_exact_logits=%ld/%ld\n",
           n, mismatches, worst, exact, total);
    return mismatches == 0 ? 0 : 1;
}
