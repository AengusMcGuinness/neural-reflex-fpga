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
        int prompt[FP_MAXPROMPT] = {0}, toks[FP_SEQ] = {0}, nout = 0;
        for (int i = 0; i < np; i++) prompt[i] = g[2 + i];
        llama_fp4(prompt, np, steps, toks, &nout);
        int bad = nout != (steps < ng ? steps : ng);
        for (int i = 0; i < nout && i < ng; i++)
            if (toks[i] != g[2 + np + i]) {
                printf("QUICK MISMATCH pos %d: %d vs golden %d\n", i, toks[i], g[2 + np + i]);
                bad = 1;
            }
        printf("QUICK %s: %d positions vs golden\n", bad ? "FAIL" : "PASS", nout);
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
        int prompt[FP_MAXPROMPT] = {0}, toks[FP_SEQ] = {0}, nout = 0;
        int np = (int)pr.size() < FP_MAXPROMPT ? (int)pr.size() : FP_MAXPROMPT;
        for (int i = 0; i < np; i++) prompt[i] = pr[i];
        llama_fp4(prompt, np, steps, toks, &nout);
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
