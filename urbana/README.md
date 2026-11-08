# EMG decoder on the Urbana (Spartan-7)

The ridge decoder is trained in Python, folded and quantized to fixed point, and computed on the FPGA (`f·W′ + b′`, then clamp).

## Flow
```
python3 python/make_dummy_model.py            # stand-in until real data: build/model.npz, build/calib.npz
python3 python/export_weights.py --map grip:0:1:2048:3000
python3 -m pytest python/test_golden.py -q
```
- `build/model.npz`: `mean`, `scale`, `W (N_FEAT, N_OUT)`, `b`, optional `target_names`. The real training script must write these same keys.
- `build/calib.npz`: held-out raw features `X`, optional targets `Y`. These set the fixed-point ranges and are used to measure the error.
- `--map name:in_lo:in_hi:out_lo:out_hi` folds a linear map to servo ticks into that output's weights and clamps it to `[out_lo, out_hi]`.

## Outputs
| file | contents |
|---|---|
| `data/weights.mem` | int16 `w_q[j][i]`, output-major (all features of output 0, then output 1, ...) |
| `data/bias.mem` | 48-bit `b_q[j]` |
| `data/cfg.mem` | 16-bit `shift[j], lo[j], hi[j]` for each output |
| `build/quant_params.json` | all of the above plus `feat_exp`, which the Mac uses to quantize features before sending them |

`python/fixed_point.py` is the bit-exact golden model, and the RTL must match `predict_int()`.
