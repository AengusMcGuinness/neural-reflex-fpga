#!/usr/bin/env python3
"""Checks the Python tokenizer port against run.c itself (ref_dump output from Linux):
  * encode(prompt) must equal the C prompt tokens, for every case and both tokenizers;
  * replaying run.c generate() with the C argmax tokens through decode()+safe() must
    reproduce the exact bytes run.c printed.
usage: test_tokenizer.py <llama dir>   (expects models/ and ref_out/ inside it)
"""
import glob
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llama_host import Tokenizer  # noqa: E402


def ints(path):
    b = open(path, "rb").read()
    return list(struct.unpack(f"<{len(b) // 4}i", b))


def replay(tok, prompt, amax):
    out = bytearray()
    token = prompt[0]
    for pos in range(len(amax)):
        nxt = prompt[pos + 1] if pos < len(prompt) - 1 else amax[pos]
        if nxt == 1:
            break
        out += Tokenizer.safe(tok.decode(token, nxt))
        token = nxt
    return bytes(out) + b"\n"


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..")
    toks = {
        "c15": Tokenizer(os.path.join(root, "models/tokenizer.bin"), 32000),
        "c260": Tokenizer(os.path.join(root, "models/tok512.bin"), 512),
    }
    cases = sorted(glob.glob(os.path.join(root, "ref_out/tok/case_*.prompt.txt")))
    fails = checked = 0
    for c in cases:
        i = c.split("case_")[1].split(".")[0]
        text = open(c, "rb").read()
        for name, tok in toks.items():
            pre = os.path.join(root, f"ref_out/tok/{name}_{i}")
            want = ints(pre + ".prompt")
            got = tok.encode(text, bos=True, eos=False)
            checked += 1
            if got != want:
                fails += 1
                print(f"ENCODE FAIL {name} case {i}: {got} != {want}")
            txt = replay(tok, want, ints(pre + ".argmax"))
            ref = open(pre + ".txt", "rb").read()
            checked += 1
            if txt != ref:
                fails += 1
                print(f"DECODE FAIL {name} case {i}:\n  py {txt!r}\n  c  {ref!r}")
    # the 200-token generations used for the hardware checks
    for name, model in (("s15M", "c15"), ("s260K", "c260")):
        pre = os.path.join(root, f"ref_out/{name}")
        checked += 1
        if replay(toks[model], ints(pre + ".prompt"), ints(pre + ".argmax")) != open(pre + ".txt", "rb").read():
            fails += 1
            print(f"DECODE FAIL {name}")
    print(f"tokenizer checks: {checked - fails}/{checked} passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
