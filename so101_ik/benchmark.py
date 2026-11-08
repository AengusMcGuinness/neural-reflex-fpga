"""Benchmark: DNN-initialised vs random-initialised Newton-Raphson IK (SO-101).

Replicates ``benchmark_ik2.py`` from the ES 259 project:

* sample ``q_true`` uniformly inside ``JOINT_LIMITS``, ``T_target = FK(q_true)``
* ``q_seed = clip(q_true + N(0, 0.5), limits)``  (the "current robot state")
* **dnn**:   Newton from the DNN prediction ``solve(T_target, q_seed)``
* **random**: best of 5 Newton runs from uniform-random starts inside the limits
* **seed**:  (extra, honest baseline) Newton started directly from the noisy seed,
             i.e. what you get WITHOUT the DNN but with the same prior knowledge.

Reports convergence %, iteration statistics, DNN vs random speedup, DNN-only
position/rotation error and per-stage wall-clock (DNN forward, Newton, total).

Usage::

    python -m so101_ik.benchmark --model_dir DIR --n 1000 [--out DIR/benchmark_summary.json --task se3]
    python -m so101_ik.benchmark --no_model --n 1000 --out benchmark_baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from so101_ik.fk import JOINT_LIMITS, N_JOINTS, build_so101_model, forward_kinematics
from so101_ik.newton import ik_newton_refine

_LO = np.array([lo for lo, _ in JOINT_LIMITS], dtype=np.float64)
_HI = np.array([hi for _, hi in JOINT_LIMITS], dtype=np.float64)

N_RANDOM_RESTARTS = 5


def _stats(results: list[tuple], total: int) -> dict:
    """results: list of (iters, pos_err_mm, rot_err_deg, converged[, ms])."""
    conv = sum(1 for r in results if r[3])
    iters_conv = [r[0] for r in results if r[3]]
    stats = {
        "converged": int(conv),
        "failed": int(total - conv),
        "convergence_percent": float(100.0 * conv / max(total, 1)),
    }
    if iters_conv:
        pct = np.percentile(iters_conv, [25, 50, 75, 90])
        stats.update(
            {
                "avg_iterations": float(np.mean(iters_conv)),
                "median_iterations": int(np.median(iters_conv)),
                "min_iterations": int(min(iters_conv)),
                "max_iterations": int(max(iters_conv)),
                "p25_iterations": int(pct[0]),
                "p50_iterations": int(pct[1]),
                "p75_iterations": int(pct[2]),
                "p90_iterations": int(pct[3]),
            }
        )
    pos_all = [r[1] for r in results]
    stats["final_pos_err_mm_median"] = float(np.median(pos_all)) if pos_all else None
    if len(results) and len(results[0]) > 4:
        ms = [r[4] for r in results]
        stats["mean_ms"] = float(np.mean(ms))
        stats["median_ms"] = float(np.median(ms))
    return stats


def _best_of_random(T_target, robot, rng, task, max_iter, eps_v_mm, eps_w_rad):
    best = (max_iter, 999.0, 999.0, False)
    t0 = time.perf_counter()
    for _ in range(N_RANDOM_RESTARTS):
        q_rand = rng.uniform(_LO, _HI)
        _, it, pe, re, conv = ik_newton_refine(
            T_target, q_rand, robot, task=task, max_iter=max_iter, eps_v_mm=eps_v_mm, eps_w_rad=eps_w_rad
        )
        if conv and (not best[3] or it < best[0]):
            best = (it, pe, re, conv)
        elif not best[3] and pe < best[1]:
            best = (it, pe, re, conv)
    ms = (time.perf_counter() - t0) * 1e3
    return (int(best[0]), float(best[1]), float(best[2]), bool(best[3]), ms)


def run_benchmark(
    model_dir: str | None,
    num_samples: int = 1000,
    seed: int = 42,
    device: str = "cpu",
    task: str = "se3",
    seed_noise: float = 0.5,
    max_iter: int = 200,
    eps_v_mm: float = 1.0,
    eps_w_rad: float = 0.01,
    tool_offset_m=(0.0, 0.0, 0.0),
    verbose: bool = True,
):
    rng = np.random.default_rng(seed)
    robot = build_so101_model(tool_offset_m=tool_offset_m)

    solver = None
    if model_dir is not None:
        from so101_ik.infer import IKSolver

        solver = IKSolver(model_dir, device=device, tool_offset_m=tool_offset_m)
        # warm-up so the first timed forward pass is representative
        solver.solve(np.eye(4), np.zeros(N_JOINTS))

    dnn_results, rand_results, seed_results = [], [], []
    dnn_only_pos, dnn_only_rot = [], []
    t_dnn_fwd, t_dnn_newton, t_dnn_total = [], [], []

    t_start = time.time()
    for i in range(num_samples):
        q_true = rng.uniform(_LO, _HI)
        T_target = forward_kinematics(q_true, robot)
        q_seed = np.clip(q_true + rng.normal(0.0, seed_noise, N_JOINTS), _LO, _HI)

        if solver is not None:
            t0 = time.perf_counter()
            q_dnn = np.clip(solver.solve(T_target, q_seed), _LO, _HI)
            t1 = time.perf_counter()
            _, it, pe, re, conv = ik_newton_refine(
                T_target, q_dnn, robot, task=task, max_iter=max_iter, eps_v_mm=eps_v_mm, eps_w_rad=eps_w_rad
            )
            t2 = time.perf_counter()
            t_dnn_fwd.append((t1 - t0) * 1e3)
            t_dnn_newton.append((t2 - t1) * 1e3)
            t_dnn_total.append((t2 - t0) * 1e3)
            dnn_results.append((int(it), float(pe), float(re), bool(conv), (t2 - t0) * 1e3))

            T_dnn = forward_kinematics(q_dnn, robot)
            dnn_only_pos.append(float(np.linalg.norm(T_target[:3, 3] - T_dnn[:3, 3])) * 1e3)
            cos_a = np.clip((np.trace(T_target[:3, :3].T @ T_dnn[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
            dnn_only_rot.append(float(np.degrees(np.arccos(cos_a))))

        t0 = time.perf_counter()
        _, it_s, pe_s, re_s, conv_s = ik_newton_refine(
            T_target, q_seed, robot, task=task, max_iter=max_iter, eps_v_mm=eps_v_mm, eps_w_rad=eps_w_rad
        )
        seed_results.append((int(it_s), float(pe_s), float(re_s), bool(conv_s), (time.perf_counter() - t0) * 1e3))

        rand_results.append(_best_of_random(T_target, robot, rng, task, max_iter, eps_v_mm, eps_w_rad))

        if verbose and (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{num_samples}", flush=True)

    elapsed = time.time() - t_start
    rand_stats = _stats(rand_results, num_samples)
    seed_stats = _stats(seed_results, num_samples)

    summary = {
        "robot": "SO-101 (so101_new_calib, 5 kinematic joints)",
        "model_dir": model_dir,
        "num_samples": int(num_samples),
        "seed": int(seed),
        "device": device,
        "task": task,
        "seed_noise_rad": float(seed_noise),
        "max_iterations": int(max_iter),
        "random_restarts": N_RANDOM_RESTARTS,
        "tolerance_position_mm": float(eps_v_mm),
        "tolerance_rotation_deg": float(np.degrees(eps_w_rad)),
        "elapsed_seconds": float(elapsed),
        "random": rand_stats,
        "seed_init_baseline": seed_stats,
    }
    if solver is not None:
        dnn_stats = _stats(dnn_results, num_samples)
        summary["dnn"] = dnn_stats
        summary["dnn_only_error"] = {
            "pos_err_mm_mean": float(np.mean(dnn_only_pos)),
            "pos_err_mm_median": float(np.median(dnn_only_pos)),
            "rot_err_deg_mean": float(np.mean(dnn_only_rot)),
            "rot_err_deg_median": float(np.median(dnn_only_rot)),
        }
        summary["timing_ms"] = {
            "dnn_forward_mean": float(np.mean(t_dnn_fwd)),
            "newton_polish_mean": float(np.mean(t_dnn_newton)),
            "total_mean": float(np.mean(t_dnn_total)),
            "total_median": float(np.median(t_dnn_total)),
            "random_best_of_5_mean": float(rand_stats.get("mean_ms", float("nan"))),
            "seed_init_mean": float(seed_stats.get("mean_ms", float("nan"))),
        }
        if dnn_stats.get("avg_iterations") and rand_stats.get("avg_iterations"):
            summary["speedup_x"] = float(rand_stats["avg_iterations"] / dnn_stats["avg_iterations"])
            summary["saved_iterations"] = float(rand_stats["avg_iterations"] - dnn_stats["avg_iterations"])
        if dnn_stats.get("avg_iterations") and seed_stats.get("avg_iterations"):
            summary["speedup_vs_seed_init_x"] = float(seed_stats["avg_iterations"] / dnn_stats["avg_iterations"])
    return summary, dnn_results, rand_results, seed_results


def _print_block(title: str, st: dict, total: int) -> None:
    print(f"  {title}:")
    print(f"    Convergence rate:    {st['converged']}/{total} ({st['convergence_percent']:.1f}%)")
    if "avg_iterations" in st:
        print(f"    Avg iterations:      {st['avg_iterations']:.1f}")
        print(f"    Median iterations:   {st['median_iterations']}")
        print(f"    Min / Max:           {st['min_iterations']} / {st['max_iterations']}")
        print(
            f"    Percentiles:         P25={st['p25_iterations']} P50={st['p50_iterations']} "
            f"P75={st['p75_iterations']} P90={st['p90_iterations']}"
        )
    if "mean_ms" in st:
        print(f"    Mean wall time:      {st['mean_ms']:.2f} ms")
    print()


def print_summary(summary: dict, dnn_results, rand_results, seed_results) -> None:
    n = summary["num_samples"]
    print()
    print("=" * 70)
    print(f"  SO-101 IK SOLVER BENCHMARK ({n} random targets, task={summary['task']})")
    print("  Newton-Raphson + Damped Least Squares + Step Clamping")
    print(f"  Model dir: {summary['model_dir']} | Device: {summary['device']}")
    print(
        f"  Tolerance: {summary['tolerance_position_mm']} mm position, "
        f"{summary['tolerance_rotation_deg']:.2f} deg rotation"
    )
    print(f"  Max iterations: {summary['max_iterations']} | Random uses best-of-{N_RANDOM_RESTARTS} restarts")
    print(f"  Time: {summary['elapsed_seconds']:.1f}s")
    print("=" * 70)
    print()
    if "dnn" in summary:
        _print_block("DNN-Initialized (1 DNN call + Newton polish)", summary["dnn"], n)
    _print_block(f"Random-Initialized (best of {N_RANDOM_RESTARTS} restarts)", summary["random"], n)
    _print_block("Seed-Initialized baseline (Newton from the noisy seed, NO DNN)", summary["seed_init_baseline"], n)

    if "dnn" in summary:
        e = summary["dnn_only_error"]
        print("  DNN-only accuracy (before Newton):")
        print(f"    Position error:      mean={e['pos_err_mm_mean']:.2f} mm  median={e['pos_err_mm_median']:.2f} mm")
        print(f"    Rotation error:      mean={e['rot_err_deg_mean']:.2f} deg  median={e['rot_err_deg_median']:.2f} deg")
        t = summary["timing_ms"]
        print("  Timing per solve (ms):")
        print(
            f"    DNN forward={t['dnn_forward_mean']:.3f}  Newton={t['newton_polish_mean']:.3f}  "
            f"total={t['total_mean']:.3f} (median {t['total_median']:.3f})"
        )
        print(f"    Random best-of-{N_RANDOM_RESTARTS}={t['random_best_of_5_mean']:.3f}  seed-init={t['seed_init_mean']:.3f}")
        print()
        if "speedup_x" in summary:
            print(f"  SPEEDUP: {summary['speedup_x']:.2f}x fewer iters (DNN vs Random)")
            print(f"  SAVED:   ~{summary['saved_iterations']:.1f} iterations per solve")
        if "speedup_vs_seed_init_x" in summary:
            print(f"  vs seed-init: {summary['speedup_vs_seed_init_x']:.2f}x fewer iters")
        print(
            f"  RELIABILITY: DNN {summary['dnn']['converged']} vs Random {summary['random']['converged']} "
            f"vs Seed-init {summary['seed_init_baseline']['converged']} converged out of {n}"
        )
        print()
        print("-" * 70)
        print("  First 30 individual results:")
        print(f"  {'#':>3}  {'DNN it':>7} {'DNN':>4} {'pos_mm':>7}  {'Rnd it':>7} {'Rnd':>4} {'pos_mm':>7}  {'Seed it':>7} {'Seed':>4}")
        for i in range(min(30, n)):
            d, r, s = dnn_results[i], rand_results[i], seed_results[i]
            print(
                f"  {i + 1:>3}  {d[0]:>7}  {'YES' if d[3] else 'no':>4} {d[1]:>7.2f}  "
                f"{r[0]:>7}  {'YES' if r[3] else 'no':>4} {r[1]:>7.2f}  "
                f"{s[0]:>7}  {'YES' if s[3] else 'no':>4}"
            )
    else:
        print(f"  RELIABILITY: Random {summary['random']['converged']} vs Seed-init "
              f"{summary['seed_init_baseline']['converged']} converged out of {n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DNN-initialised IK vs random restarts (SO-101)")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--no_model", action="store_true", help="run only the random-init and seed-init baselines")
    parser.add_argument("--n", "--num_samples", dest="n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--task", type=str, default="se3", choices=["se3", "pen"])
    parser.add_argument("--seed_noise", type=float, default=0.5)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--eps_v_mm", type=float, default=1.0)
    parser.add_argument("--eps_w_rad", type=float, default=0.01)
    parser.add_argument("--tool_offset_m", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--out", type=str, default=None, help="JSON path (default: <model_dir>/benchmark_summary.json)")
    args = parser.parse_args()

    model_dir = None if args.no_model else args.model_dir
    if model_dir is None and not args.no_model:
        parser.error("--model_dir is required unless --no_model is given")

    summary, dnn_results, rand_results, seed_results = run_benchmark(
        model_dir=model_dir,
        num_samples=args.n,
        seed=args.seed,
        device=args.device,
        task=args.task,
        seed_noise=args.seed_noise,
        max_iter=args.max_iter,
        eps_v_mm=args.eps_v_mm,
        eps_w_rad=args.eps_w_rad,
        tool_offset_m=tuple(args.tool_offset_m),
    )
    print_summary(summary, dnn_results, rand_results, seed_results)

    out = args.out
    if out is None:
        out = os.path.join(model_dir, "benchmark_summary.json") if model_dir else "benchmark_summary.json"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    def _ex(res, i):
        return {
            "iterations": res[i][0],
            "position_error_mm": res[i][1],
            "rotation_error_deg": res[i][2],
            "converged": res[i][3],
        }

    examples = []
    for i in range(min(30, args.n)):
        row = {"index": i + 1, "random": _ex(rand_results, i), "seed_init": _ex(seed_results, i)}
        if dnn_results:
            row["dnn"] = _ex(dnn_results, i)
        examples.append(row)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "first_30_examples": examples}, f, indent=2)
    print(f"\nSaved benchmark JSON -> {out}")


if __name__ == "__main__":
    main()
