# llama2.c on the Opal Kelly XEM7310 (Artix-7 xc7a75t)

A port of the FP4 stories260K accelerator from `llama/` (RealDigital Urbana, Spartan-7 xc7s50)
to the Opal Kelly XEM7310-A75. Same model, same numerics: the board reproduces the same golden
tokens bit for bit. What changed:

| | Urbana (`llama/`) | XEM7310 (`xem7310/`) |
|---|---|---|
| FPGA | xc7s50, 32.6k LUTs, 75 BRAM36, 120 DSP | xc7a75t, 47.2k LUTs, 105 BRAM36, 180 DSP |
| matmul | 64 FP4 MACs per clock | **128** FP4 MACs per clock (both ROM lanes at once) |
| attention | 2 passes of 4 heads over the KV cache | **1 pass of 8 heads** |
| host link | UART, 2 Mbaud, one round trip per command | **FrontPanel USB**, one round trip per run |
| kernel clock | 125 MHz from the 100 MHz oscillator | **133.3 MHz** from the 200 MHz oscillator |
| cycles for 8 positions (RTL co-simulation) | 96,918 | **75,490** |
| cycles per token, 256 positions | 13.9k | **10.0k** |
| tok/s, 256 positions, measured on chip | 8,981 | **13,333** |
| `stream`, new tokens/s end to end (USB included) | about 3,900 | **9,053** |

Measured on the board (XEM7310-A75 on USB 2 high speed, 2026-11-08), "Once upon a time", 256
positions. All 8 golden prompts match the C++ golden model token for token.

| | tok/s |
|---|---|
| XEM7310 FP4 accelerator, 133.3 MHz | **13,333** |
| XEM7310 FP4 accelerator, 125 MHz build | 12,500 |
| Urbana FP4 accelerator, 125 MHz (`llama/`) | 8,981 |
| MacBook M3 Pro, llama2.c `clang -O3` | 8,225 |
| MacBook M3 Pro, llama2.c `-O3 -ffast-math -mcpu=native` | 15,937 |

So the port is 1.48x the Urbana and 1.62x the Mac's plain `-O3` build. It is still 0.84x the
Mac's `-ffast-math` build. Streaming is 2.3x faster end to end than on the Urbana, because a run
costs one USB write and one read instead of many UART round trips.

Resources (post-route, 133.3 MHz): 40,058 LUTs (85%), 82 BRAM36 tiles (78%), 152 DSPs (84%).
Timing is met with 0.046 ns of setup slack at 133.3 MHz. The 125 MHz build has 0.208 ns; rebuild
with `FP4_CLK_DIV=8.0` if you want the margin.

## Run it

You need the XEM7310 on USB and the CS1410 FrontPanel SDK on the Mac (`~/cs1410_lab/lib/macOS`,
the same binding `fpga/check_board.py` uses). Nothing here needs Vivado; the tested bitstream is
in `bitstreams/`.

```bash
export DYLD_LIBRARY_PATH="$HOME/cs1410_lab/lib/macOS"
cd xem7310
python3 host/fp4_ok_host.py --bit bitstreams/llama_xem7310_top.bit check
```

`--bit` configures the FPGA over USB; a power cycle loses it. Leave `--bit` off on later
commands while the board stays powered.

```bash
python3 host/fp4_ok_host.py run -i "Once upon a time" -n 256   # one run, prints tok/s
python3 host/fp4_ok_host.py stream                              # endless stories, Ctrl-C to stop
python3 host/fp4_ok_host.py check                               # all 8 golden prompts, exact match
```

LEDs (active low): 0 = clock locked, 1 = kernel busy, 2 = commands arriving, 3 = results leaving,
4 = a FIFO overflowed (reset by reopening the host, which pulses WireIn 0x00).

## How it works

- **Kernel** (`fp4/llama_fp4.cpp`): the Urbana kernel with two changes. The matmul engine reads
  ROM lane A (even chunks) and lane B (odd chunks) in the same cycle, so it does two 64-weight
  chunks per clock; the W2 matmul's 3-chunk rows are paired over 3 cycles. Attention scores all
  8 heads in one pass over the cache. Integer sums are exact and every float operation is the same
  single IEEE op in the same order, so the outputs are bit-identical. The ROM lane is a template
  parameter so HLS sees one read per ROM per cycle (otherwise it schedules the loop at II=2).
- **Board top** (`rtl/llama_xem7310_top.sv`): FrontPanel endpoints on okClk (100.8 MHz), the
  kernel on an MMCM clock from the 200 MHz oscillator, and two `xpm_fifo_async` FIFOs between them.
- **Command processor** (`rtl/fp_cmd_proc.sv`): executes the host's command words (burst writes,
  burst reads, status reads, and WAIT, which polls `ap_done` on the FPGA). A run is one PipeIn
  write (prompt, n_prompt, steps, ap_start, WAIT, reads) and one PipeOut read, so the host does
  not poll the kernel over USB. The busy-cycle counter runs from the ap_start write until WAIT
  sees ap_done.

| endpoint | use |
|---|---|
| WireIn 0x00 | bit 0: reset |
| WireOut 0x20 | words waiting in the response FIFO |
| WireOut 0x21 | [0] locked, [1] busy, [2] command overflow, [3] response overflow, [31:16] 0xF4F4 |
| PipeIn 0x80 | command words: `[31:28]` op (0 NOP, 1 WRITE, 2 READ, 3 WAIT), `[27]` status block, `[26:16]` count-1, `[15:0]` byte address; WRITE is followed by its data words |
| PipeOut 0xA0 | READ results, in order |

Status block (READ with bit 27): +0x0/+0x4 busy cycles low/high, +0x8 kernel clock in Hz,
+0xC magic `0xF4F4_0002`.

## Rebuild (needs `ssh cs1410vnc`)

Sources mirror to `~/xem7310/src` on the server. Golden tokens and HLS use the same flow as
`llama/README.md`; the HLS part is `xc7a75tfgg484-1`.

```bash
# on the Mac
rsync -a xem7310/ cs1410vnc:xem7310/src/
```

On the server:

```bash
cd ~/xem7310
source /software/xilinx-tools-2022/Vitis_HLS/2022.2/settings64.sh
export PATH=$HOME/bin:$PATH          # the zip stand-in that cosim needs
# 1. bit-exactness in C (writes gold/, compare with src/fp4/golden/)
g++ -O2 -ffp-contract=off -I/software/xilinx-tools-2022/Vitis_HLS/2022.2/include -Isrc/fp4 \
    -o tb_fast src/fp4/tb_fp4.cpp src/fp4/llama_fp4.cpp
mkdir -p gold && ./tb_fast ~/llama/ref_out gold 256
for f in src/fp4/golden/*.tokens; do cmp $f gold/$(basename $f); done
# 2. HLS, RTL co-simulation (8 positions vs golden), IP export
FP4_STEPS="csynth cosim syn" FP4_REF=$HOME/llama/ref_out FP4_OUT=$PWD/gold vitis_hls -f src/fp4/run_hls.tcl
# 3. command-processor testbench
source /software/xilinx-tools-2022/Vivado/2022.2/settings64.sh
mkdir -p tbsim && cd tbsim && xvlog -sv ../src/rtl/fp_cmd_proc.sv ../src/rtl/tb_fp_cmd_proc.sv \
    && xelab tb_fp_cmd_proc -s tb && xsim tb -R && cd ..
# 4. bitstream (refuses to write one unless timing is met); FP4_CLK_DIV divides the 1 GHz VCO
FP4_CLK_DIV=7.5 vivado -mode batch -source src/build.tcl -tclargs fp4_hls/sol1/impl/ip out
```

`build.tcl` reads the FrontPanel HDL from the CS1410 copy on the server
(`~/cs1410_lab/lab2/lab2_server/OK_library`); pass another directory as the third argument.
The vendor files are not in this repo.

## Files

| path | what |
|---|---|
| `fp4/llama_fp4.cpp`, `fp4/llama_fp4.h` | kernel (HLS source = golden model), 128-lane matmul, 8-head attention |
| `fp4/model_rom.h`, `fp4/tok512.bin` | weights ROM image and tokenizer, unchanged from `llama/fp4` |
| `fp4/tb_fp4.cpp`, `fp4/golden/` | testbench and the golden tokens the board must reproduce (unchanged) |
| `fp4/run_hls.tcl` | HLS flow for xc7a75t |
| `rtl/llama_xem7310_top.sv` | board top: FrontPanel, MMCM, clock-crossing FIFOs, LEDs |
| `rtl/fp_cmd_proc.sv`, `rtl/tb_fp_cmd_proc.sv` | command processor and its testbench |
| `constraints/llama_xem7310.xdc` | FrontPanel pins (Opal Kelly), 200 MHz clock and LEDs (CS1410 XDC) |
| `build.tcl` | bitstream build |
| `host/fp4_ok_host.py` | run / check / stream over FrontPanel |
| `bitstreams/` | the tested bitstream with its timing and utilization reports |
