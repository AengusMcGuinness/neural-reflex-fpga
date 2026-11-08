// Testbench / golden-output generator for the FP4 accelerator.
//  1. quality: teacher-force fp32 llama2.c's own token sequence through fp4_forward and count
//     positions where the FP4 argmax equals the fp32 argmax (generated positions only);
//  2. golden : run the on-chip generate loop (llama_fp4) for each prompt and write the tokens
//     to <outdir>/<case>.tokens - the board must reproduce these exactly.
// usage: tb_fp4 <refdir> <outdir> [steps]
//   refdir holds ref_dump outputs: s260K.{prompt,argmax} and tok/c260_<n>.{prompt,argmax}
#include "llama_fp4.h"
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <string>
#include <vector>

static std::vector<int> ints(const std::string &p) {
    std::vector<int> v;
    FILE *f = fopen(p.c_str(), "rb");
    if (!f) return v;
    int x;
    while (fread(&x, 4, 1, f) == 1) v.push_back(x);
    fclose(f);
    return v;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s refdir outdir [steps]\n", argv[0]);
        return 2;
    }
    std::string ref = argv[1], out = argv[2];
    int steps = argc > 3 ? atoi(argv[3]) : FP_SEQ;
    if (argc > 4 && std::string(argv[4]) == "quick") {
        // RTL co-simulation: one free-running generation of `steps` positions, checked against
        // the golden file in <out>/s260K.tokens (made by a full run of this testbench).
        std::vector<int> g = ints(out + "/s260K.tokens");
        if (g.size() < 3) {
            printf("quick mode needs %s/s260K.tokens\n", out.c_str());
            return 2;
        }
        int np = g[0], ng = g[1];
        static int prompt[FP_PARAMS], toks[FP_RING];
        int nout = 0;
        for (int i = 0; i < np; i++) prompt[i] = g[2 + i];
        llama_fp4(prompt, np, steps, toks, &nout, 0, 0, 0, 0);
        int bad = nout != (steps < ng ? steps : ng);
        for (int i = 0; i < nout && i < ng; i++)
            if (toks[i] != g[2 + np + i]) {
                printf("QUICK MISMATCH pos %d: %d vs golden %d\n", i, toks[i], g[2 + np + i]);
                bad = 1;
            }
        printf("QUICK %s: %d positions vs golden\n", bad ? "FAIL" : "PASS", nout);
        return bad;
    }
    if (argc > 4 && std::string(argv[4]).rfind("stream", 0) == 0) {
        // mode 1 (on-chip endless stream) must equal the host algorithm (host/fp4_host.py
        // `stream`) driving mode 0 run by run. Openers = the golden prompts without BOS.
        // argv[4] = "stream" (context 48, window 256, 1500 new tokens) or
        //           "streamquick" (context 4, window 12, 30 new tokens, for RTL co-simulation)
        bool quickst = std::string(argv[4]) == "streamquick";
        int C = quickst ? 4 : 48, W = quickst ? 12 : 256, N = quickst ? 30 : 1500;
        std::vector<std::vector<int>> op;
        const char *names[] = {"s260K", "c260_1", "c260_2", "c260_3", "c260_5", "c260_6", "c260_7"};
        for (const char *nm : names) {
            std::vector<int> g = ints(out + "/" + nm + ".tokens");
            if (g.size() < 3) {
                printf("stream mode needs %s/%s.tokens\n", out.c_str(), nm);
                return 2;
            }
            op.push_back(std::vector<int>(g.begin() + 3, g.begin() + 2 + g[0])); // drop BOS
        }
        static int table[FP_PARAMS], ring[FP_RING];
        int t = 0;
        table[t++] = (int)op.size();
        for (auto &o : op) {
            table[t++] = (int)o.size();
            for (int x : o) table[t++] = x;
        }
        int produced = 0;
        llama_fp4(table, 0, 0, ring, &produced, 1, C, W, N);
        // reference: the host algorithm, one mode-0 run per chunk
        std::vector<int> want;
        auto emit = [&](int tok, int flag) {
            int idx = (int)want.size();
            want.push_back(((idx & 0x7fff) << 16) | (flag << 12) | tok);
        };
        int k = 0, story_len = 0, emitted = 0;
        std::vector<int> hist(op[0]);
        for (size_t j = 0; j < hist.size(); j++) emit(hist[j], j == 0);
        while (emitted < N) {
            static int pr[FP_PARAMS], tk[FP_RING];
            int m = (int)hist.size() < C - 1 ? (int)hist.size() : C - 1;
            pr[0] = 1;
            for (int j = 0; j < m; j++) pr[1 + j] = hist[hist.size() - m + j];
            int np = 1 + m, n = 0;
            llama_fp4(pr, np, W, tk, &n, 0, 0, 0, 0);
            bool ended = false;
            for (int pos = np - 1; pos < n; pos++) {
                if (tk[pos] == 1) {
                    ended = true;
                    break;
                }
                emit(tk[pos], 0);
                hist.push_back(tk[pos]);
                emitted++;
            }
            story_len += n - (np - 1);
            if (!ended && hist.size() >= 64) {
                std::vector<int> tail(hist.end() - 64, hist.end());
                std::sort(tail.begin(), tail.end());
                int distinct = (int)(std::unique(tail.begin(), tail.end()) - tail.begin());
                if (distinct <= 12) ended = true;
            }
            if (!ended && story_len > 800) ended = true;
            if (ended) {
                story_len = 0;
                k = (k + 1) % (int)op.size();
                hist = op[k];
                for (size_t j = 0; j < hist.size(); j++) emit(hist[j], j == 0);
            }
        }
        int bad = produced != (int)want.size();
        int stories = 0;
        for (int i = 0; i < (int)want.size(); i++) {
            stories += (want[i] >> 12) & 1;
            if (i >= produced - FP_RING && ring[i & (FP_RING - 1)] != want[i]) {
                if (!bad) printf("STREAM MISMATCH entry %d: 0x%08x vs host algorithm 0x%08x\n", i, ring[i & (FP_RING - 1)], want[i]);
                bad = 1;
            }
        }
        printf("STREAM %s: %d ring entries (%d new tokens requested, %d stories) vs host algorithm\n",
               bad ? "FAIL" : "PASS", produced, N, stories);
        return bad;
    }
    std::vector<std::string> cases = {"s260K"};
    for (int i = 1; i <= 7; i++) cases.push_back("tok/c260_" + std::to_string(i));
    long agree = 0, total = 0;
    int bad = 0;
    for (const std::string &c : cases) {
        std::vector<int> pr = ints(ref + "/" + c + ".prompt"), am = ints(ref + "/" + c + ".argmax");
        if (pr.empty() || am.empty()) {
            printf("missing reference %s\n", c.c_str());
            bad = 1;
            continue;
        }
        // 1. teacher forcing on the fp32 sequence
        int a_case = 0, t_case = 0, n = (int)am.size() < FP_SEQ ? (int)am.size() : FP_SEQ;
        int tok = pr[0];
        for (int pos = 0; pos < n; pos++) {
            int a = fp4_forward(tok, pos);
            if (pos >= (int)pr.size() - 1) {
                t_case++;
                a_case += a == am[pos];
            }
            int next = pos < (int)pr.size() - 1 ? pr[pos + 1] : am[pos];
            if (next == 1) break;
            tok = next;
        }
        agree += a_case;
        total += t_case;
        // 2. free-running generation on the accelerator's own loop
        static int prompt[FP_PARAMS], toks[FP_RING];
        int nout = 0;
        int np = (int)pr.size() < FP_MAXPROMPT ? (int)pr.size() : FP_MAXPROMPT;
        for (int i = 0; i < np; i++) prompt[i] = pr[i];
        llama_fp4(prompt, np, steps, toks, &nout, 0, 0, 0, 0);
        std::string name = c.substr(c.find_last_of('/') + 1);
        FILE *f = fopen((out + "/" + name + ".tokens").c_str(), "wb");
        if (!f || nout < 1) {
            printf("cannot write golden for %s (n_out %d)\n", c.c_str(), nout);
            bad = 1;
        } else {
            fwrite(prompt, 4, np, f);  // header: n_prompt, prompt tokens, then argmax per pos
            fclose(f);
            f = fopen((out + "/" + name + ".tokens").c_str(), "wb");
            int hdr[2] = {np, nout};
            fwrite(hdr, 4, 2, f);
            fwrite(prompt, 4, np, f);
            fwrite(toks, 4, nout, f);
            fclose(f);
        }
        printf("%-10s teacher-forced agreement with fp32 llama2.c: %3d/%3d   generated %d positions\n",
               name.c_str(), a_case, t_case, nout);
    }
    printf("TOTAL agreement %ld/%ld (%.1f%%)\n", agree, total, total ? 100.0 * agree / total : 0.0);
    return bad;
}
