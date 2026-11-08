# llama2.c on the Urbana FPGA (Spartan-7 xc7s50)

Two accelerators for Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) on the RealDigital
Urbana board (MIT 6.205):

| design | model | where the weights live | status |
|---|---|---|---|
| **FP4** (`fp4/`, `rtl/llama_fp4_top.sv`) | stories260K | on chip (block RAM), FP4 | **runs on the board**: bit-exact vs its golden model; 8,981 tok/s per run, endless on-chip stream at 6,900–7,600 tok/s end to end |
| fp32 (`hls/`, `rtl/llama_top.sv`) | stories260K / stories15M | DDR3 | built, tests pass, not yet run on the board |

## Run the FP4 accelerator (the one that works)

You need the board on USB (FT2232H, shows up as "Xilinx JTAG+Serial"), Homebrew and Python 3.
Nothing here needs Vivado. The tested bitstream is in `bitstreams/`.

```bash
brew install openfpgaloader
cd llama
python3 -m venv host/.venv && host/.venv/bin/pip install pyserial==3.5
```

1. Check the board is visible:
   ```bash
   openFPGALoader --scan-usb
   openFPGALoader -c digilent --detect
   ```
   `--detect` should print `model xc7s50`.

2. Load the bitstream. This writes the FPGA's configuration memory only; a power cycle brings
   back whatever is in the board's flash.
   ```bash
   openFPGALoader -c digilent --fpga-part xc7s50csga324 bitstreams/llama_fp4_top.bit
   ```
   LED 0 lights when the clock locks.

3. Find the UART. It's the second of the two ports (the one ending in `1`):
   ```bash
   ls /dev/cu.usbserial-*
   ```

4. Generate text. Prompts are tokenized with the bundled `fp4/tok512.bin`:
   ```bash
   host/.venv/bin/python host/fp4_host.py --port /dev/cu.usbserial-XXXX1 run -i "Once upon a time" -n 256
   ```
   `-n` is the number of positions, 1 to 256 (the on-chip KV cache size). The reported tok/s
   comes from a cycle counter on the FPGA, so the UART is not in the timing. Compare with the
   Mac numbers at `-n 256`: shorter runs are faster because attention covers fewer positions.

5. Generate forever, on chip (Ctrl-C to stop; it then prints the speed):
   ```bash
   host/.venv/bin/python host/fp4_host.py --port /dev/cu.usbserial-XXXX1 stream
   ```
   The FPGA runs the endless loop itself (kernel mode 1). Each run is BOS plus the last 47
   story tokens, 256 positions. A story ends at end-of-text, when the last 64 tokens hold 12 or
   fewer distinct tokens (greedy decoding stuck in a loop), or after 800 positions; the next
   story starts from a rotating list of openers. Tokens go into a 2,048-entry ring buffer
   with sequence tags, and the Mac drains it 512 entries per USB round trip.
   - `-q` measures without printing.
   - `-c 16` / `-c 8` carry less context, which is faster.
   - `--host` runs the old host-driven loop, one round trip per run.
   - `streamcheck` requires the on-chip stream to equal the host-driven one, token for token.

   Measured on the board (2026-11-08):

   | | new tokens/s end to end | FPGA |
   |---|---|---|
   | on-chip stream, context 48 (default) | **6,919** | 7,176 |
   | on-chip stream, context 8 | 7,612 | 8,111 |
   | host-driven stream (`--host`) | 3,618 | 7,185 |

   The board draws more current when it streams continuously. On a plain Mac USB port it has
   dropped off USB twice during long streams. A powered USB hub (or the board's external power
   input) avoids that. If it happens, replug and reload the bitstream.

6. Check the board against the golden model. All 8 prompts must match token for token:
   ```bash
   host/.venv/bin/python host/fp4_host.py --port /dev/cu.usbserial-XXXX1 check
   ```

Measured on the board (2026-11-08), 256 positions, "Once upon a time":

| | tok/s |
|---|---|
| Urbana FP4 accelerator (125 MHz) | **8,981** |
| MacBook M3 Pro, llama2.c `clang -O3` (`make run`) | 8,225 |
| MacBook M3 Pro, llama2.c `-O3 -ffast-math -mcpu=native` | 15,937 |

### How the FP4 design works

Everything happens on the FPGA: embedding, 5 layers, classifier, argmax, and the generate loop
itself. The Mac writes the prompt tokens once, starts the kernel, and reads all tokens back.

- **Weights:** FP4 E2M1 in blocks of 16, each with an 8-bit scale (`fp4/convert.py`). RMSNorm
  gains are folded into the weights. The classifier and the token embedding share one table.
- **Matmul:** 64 FP4×int8 multiply-adds per clock in exact integer arithmetic. Activations
  are quantized to int8 with power-of-two scales.
- **Attention:** int8 K/V cache on chip for 256 positions. Integer scores, softmax from a
  2^(i/256) table, integer weighted sum of V, 4 heads per pass.
- **Numerics:** `fp4/llama_fp4.cpp` is both the HLS source and the golden model. Every operation
  is exact integer arithmetic or a single IEEE float op, so the board reproduces it bit for bit.
  Against fp32 llama2.c it picks the same next token 85.3% of the time (teacher-forced).

### Rebuild the FP4 bitstream (needs `ssh cs1410vnc`)

Sources mirror to `~/llama/src` on the server.

```bash
# on the Mac: regenerate the ROM image from the llama2.c checkpoint
host/.venv/bin/pip install numpy==2.1.3
host/.venv/bin/python fp4/convert.py models/stories260K.bin fp4/model_rom.h
```

On the server, after `source /software/xilinx-tools-2022/Vitis_HLS/2022.2/settings64.sh`:

1. Golden tokens and quality, using the fast native build:
   ```bash
   g++ -O2 -ffp-contract=off -I/software/xilinx-tools-2022/Vitis_HLS/2022.2/include -Isrc/fp4 \
       -o tb_fast src/fp4/tb_fp4.cpp src/fp4/llama_fp4.cpp
   ./tb_fast ref_out golden 256
   ```

2. HLS synthesis and IP export (`syn`), then RTL co-simulation (8 positions vs golden):
   ```bash
   FP4_STEPS=syn FP4_REF=x FP4_OUT=x vitis_hls -f src/fp4/run_hls.tcl
   FP4_STEPS="csynth cosim" FP4_REF=$PWD/ref_out FP4_OUT=$PWD/golden vitis_hls -f src/fp4/run_hls.tcl
   ```
   Co-simulation calls `zip`, which cs1410vnc lacks; without it Vitis reports FAIL even when
   the RTL is correct. A Python stand-in lives in `~/bin/zip` on the server.

3. Bitstream. The script refuses to write one unless timing is met:
   ```bash
   source /software/xilinx-tools-2022/Vivado/2022.2/settings64.sh
   FP4_CTRL_AW=11 vivado -mode batch -source src/build_fp4.tcl -tclargs <fp4_hls/sol1/impl/ip> out
   ```

## The fp32 DDR3 design

`hls/llama_kernel.cpp` is llama2.c's forward pass in fp32. It matches `run.c`'s summation order;
in C simulation it gives the same argmax at all 200 positions on stories260K and stories15M.
Weights stream from the board's 256 MB DDR3 through Dandi's UberDDR3 controller (`dandi/`, 6.205
lab06) and `rtl/axi_wb_ddr.sv`. The host loop is `host/llama_host.py`. On this board it is
DDR3-bandwidth-bound (about 1.3 GB/s), so it cannot beat a laptop. It is built
(`build.tcl`) and its testbenches pass (`rtl/tb_*.sv`), but it hasn't been run on the board.

## Files

| path | what |
|---|---|
| `fp4/llama_fp4.cpp`, `fp4/llama_fp4.h` | FP4 kernel (HLS source = golden model) |
| `fp4/convert.py` → `fp4/model_rom.h` | checkpoint → on-chip ROM image |
| `fp4/tb_fp4.cpp`, `fp4/golden/` | testbench, golden tokens the board must reproduce |
| `fp4/run_hls.tcl`, `build_fp4.tcl` | HLS flow, bitstream build |
| `rtl/llama_fp4_top.sv` | board top: 125 MHz MMCM, UART bridge, cycle counter, LEDs |
| `rtl/uart_axi_bridge.sv` | UART ↔ AXI bridge (`W`/`R`/`S` commands), with testbench |
| `constraints/*.xdc` | verbatim lines from the 6.205 `top_level.xdc` (`dandi/top_level.xdc`) |
| `host/fp4_host.py` | run / check the FP4 accelerator |
| `host/llama_host.py`, `host/test_tokenizer.py`, `host/fake_board.py` | fp32 host, tokenizer port (checked against `run.c`), board stand-in |
| `ref/ref_dump.c` | dumps `run.c`'s tokens/logits/RoPE table as references |
| `bitstreams/` | the tested FP4 bitstream + its timing and utilization reports |
