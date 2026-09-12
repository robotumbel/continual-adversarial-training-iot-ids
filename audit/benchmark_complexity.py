"""
benchmark_complexity.py — parameter counts, training cost, and inference
latency per backbone, for the complexity table a reviewer asked for.

Reports, per backbone:
  * total / trainable parameter count
  * teacher + Fisher storage that CAT adds on top of the backbone
  * training wall-clock per epoch, with and without the PGD inner loop
    (their ratio is the PGD training multiplier)
  * peak memory during training
  * inference latency (ms/sample) and throughput

Runs on whatever device is available; pass --device to force one. Feature
counts default to the three benchmarks' real widths.

Usage:
  python benchmark_complexity.py --out complexity.csv
  python benchmark_complexity.py --device cpu --n-features 39
"""
from __future__ import annotations
import argparse, csv, os, time

import torch
import torch.nn as nn

from run_paper3_exp1 import build_model
from evaluation import count_model_params, measure_inference_time

BACKBONES = ["dnn", "lstm", "transformer", "rdt",
             "aam_trans_no_gates", "aam_trans"]
SEQ_LEN = 50


def _logits(out):
    if isinstance(out, dict):
        return out["logits"]
    return out[0] if isinstance(out, tuple) else out


def train_epoch_cost(model, x, y, device, pgd_steps, batch=64):
    """One epoch of training over `x`, optionally with a PGD inner loop.
    Returns seconds elapsed. pgd_steps=0 gives the no-adversarial cost."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(x), batch):
        xb, yb = x[i:i + batch].to(device), y[i:i + batch].to(device)
        if pgd_steps:
            delta = torch.zeros_like(xb).uniform_(-0.1, 0.1).requires_grad_(True)
            for _ in range(pgd_steps):
                logits = _logits(model(xb + delta))
                g = torch.autograd.grad(lossf(logits, yb), delta)[0]
                delta = (delta + 0.02 * g.sign()).clamp(-0.1, 0.1).detach()
                delta.requires_grad_(True)
            xb = (xb + delta).detach()
        logits = _logits(model(xb))
        opt.zero_grad(); lossf(logits, yb).backward(); opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


CSV_FIELDS = ["backbone", "params_total", "params_trainable",
              "cat_extra_state_MB", "epoch_s_with_pgd", "epoch_s_no_pgd",
              "pgd_multiplier", "peak_train_mem_MB",
              "latency_ms_per_sample", "throughput_samples_sec", "device"]


def write_rows(path, rows):
    """Rewrite the CSV from scratch; called after each backbone."""
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--n-features", type=int, default=39,
                    help="39 for CICIoT2023/CICIoMT2024, 158 for CICIoV2024.")
    ap.add_argument("--n-classes", type=int, default=6)
    ap.add_argument("--n-samples", type=int, default=2048,
                    help="Rows used for the timing epoch.")
    ap.add_argument("--pgd-steps", type=int, default=7)
    ap.add_argument("--out", default="complexity.csv")
    args = ap.parse_args()

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    name = (torch.cuda.get_device_name(0) if device.type == "cuda"
            else "CPU")
    print(f"device: {device} ({name}) | n_features={args.n_features} | "
          f"n_classes={args.n_classes}\n")

    x = torch.randn(args.n_samples, SEQ_LEN, args.n_features)
    y = torch.randint(0, args.n_classes, (args.n_samples,))

    # Resume: keep whatever a previous attempt already measured, so a
    # crash or an out-of-memory kill on one backbone does not cost the
    # others. Delete the file to force a full re-measurement.
    rows, done = [], set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            rows = [{k: v for k, v in r.items() if k in CSV_FIELDS}
                    for r in csv.DictReader(f)]
        rows = [r for r in rows if r.get("backbone")]
        done = {r["backbone"] for r in rows}
        if done:
            print(f"resuming: {len(done)} backbone(s) already measured "
                  f"in {args.out}\n")

    for bb in BACKBONES:
        if bb in done:
            print(f"{bb:22s} already measured, skipping")
            continue
        model = build_model(bb, args.n_features, args.n_classes,
                            SEQ_LEN).to(device)
        p = count_model_params(model)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t_adv = train_epoch_cost(model, x, y, device, args.pgd_steps)
        t_std = train_epoch_cost(model, x, y, device, 0)
        peak = (torch.cuda.max_memory_allocated() / 2**20
                if device.type == "cuda" else float("nan"))

        lat = measure_inference_time(model, x, device)

        # CAT's extra state: a frozen teacher (one model copy) and a diagonal
        # Fisher vector (one float per parameter). Both are O(|theta|).
        extra_mb = 2 * p["total_params"] * 4 / 2**20

        row = {
            "backbone": bb,
            "params_total": p["total_params"],
            "params_trainable": p["trainable_params"],
            "cat_extra_state_MB": round(extra_mb, 2),
            "epoch_s_with_pgd": round(t_adv, 2),
            "epoch_s_no_pgd": round(t_std, 2),
            "pgd_multiplier": round(t_adv / t_std, 2) if t_std else None,
            "peak_train_mem_MB": round(peak, 1) if peak == peak else "",
            "latency_ms_per_sample": round(lat["latency_ms_per_sample"], 4),
            "throughput_samples_sec": round(lat["throughput_samples_sec"], 1),
            "device": name,
        }
        rows.append(row)
        print(f"{bb:22s} params={row['params_total']:>8,d} "
              f"epoch(adv)={row['epoch_s_with_pgd']:>6.2f}s "
              f"x{row['pgd_multiplier']} "
              f"lat={row['latency_ms_per_sample']:.3f} ms/sample")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Persist after every backbone, not at the end, so an
        # interrupted run keeps what it has already measured.
        write_rows(args.out, rows)

    print(f"\nwrote {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
