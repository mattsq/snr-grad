"""Experiment runner for systematic SNR parameter sweeps.

Runs benchmark.py, benchmark_spectral.py, and benchmark_hard.py with parameterized
configs, logs per-run summaries, and writes a machine-readable sweep manifest.

Usage:
  python experiment_runner.py --stage screening --trials 50 --out-dir runs/screening
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class TrialConfig:
    benchmark: str
    seed: int
    lr: float
    weight_decay: float
    rho: float
    lambda_pop: float
    gate: str
    alpha: str
    n_steps: int
    batch_size: int

    def config_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def log_uniform(lo: float, hi: float, rng: random.Random) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def screening_space(trials: int, seeds: Iterable[int], rng: random.Random) -> List[TrialConfig]:
    benchmarks = ["benchmark.py", "benchmark_spectral.py", "benchmark_hard.py"]
    gates = ["snr", "soft", "hard"]
    rhos = [0.9, 0.95, 0.99, 0.995, 0.999]
    alphas = ["online", "finite", "0.1", "0.3", "1.0", "3.0"]
    out: List[TrialConfig] = []
    for _ in range(trials):
        for benchmark in benchmarks:
            for seed in seeds:
                out.append(
                    TrialConfig(
                        benchmark=benchmark,
                        seed=seed,
                        lr=log_uniform(1e-4, 1e-2, rng),
                        weight_decay=log_uniform(1e-6, 1e-1, rng),
                        rho=rng.choice(rhos),
                        lambda_pop=log_uniform(1e-3, 1e2, rng),
                        gate=rng.choice(gates),
                        alpha=rng.choice(alphas),
                        n_steps=5000 if benchmark != "benchmark_hard.py" else 8000,
                        batch_size=rng.choice([32, 64, 128]),
                    )
                )
    return out


def interaction_space(seeds: Iterable[int]) -> List[TrialConfig]:
    benchmarks = ["benchmark.py", "benchmark_spectral.py", "benchmark_hard.py"]
    lrs = [3e-4, 1e-3, 3e-3]
    lambdas = [0.1, 1.0, 10.0]
    rhos = [0.95, 0.99, 0.995]
    out: List[TrialConfig] = []
    for benchmark, lr, lam, rho, seed in itertools.product(benchmarks, lrs, lambdas, rhos, seeds):
        out.append(
            TrialConfig(
                benchmark=benchmark,
                seed=seed,
                lr=lr,
                weight_decay=0.0,
                rho=rho,
                lambda_pop=lam,
                gate="snr",
                alpha="finite" if benchmark == "benchmark.py" else "online",
                n_steps=5000 if benchmark != "benchmark_hard.py" else 8000,
                batch_size=64,
            )
        )
    return out


def run_trial(cfg: TrialConfig, out_dir: Path) -> Dict[str, str]:
    trial_dir = out_dir / cfg.benchmark.replace(".py", "") / cfg.config_id()
    trial_dir.mkdir(parents=True, exist_ok=True)
    cfg_json = trial_dir / "config.json"
    cfg_json.write_text(json.dumps(asdict(cfg), indent=2))

    env = {
        **dict(**__import__("os").environ),
        "SNR_SWEEP_CONFIG": str(cfg_json),
        "SNR_SWEEP_OUT": str(trial_dir / "metrics.csv"),
    }
    cmd = [sys.executable, cfg.benchmark, "--sweep-config", str(cfg_json), "--sweep-out", str(trial_dir / "metrics.csv")]
    proc = subprocess.run(cmd, cwd=Path(__file__).parent, env=env, capture_output=True, text=True)

    (trial_dir / "stdout.log").write_text(proc.stdout)
    (trial_dir / "stderr.log").write_text(proc.stderr)
    return {
        "config_id": cfg.config_id(),
        "benchmark": cfg.benchmark,
        "seed": str(cfg.seed),
        "returncode": str(proc.returncode),
        "metrics": str(trial_dir / "metrics.csv"),
    }


def write_manifest(rows: List[Dict[str, str]], out_dir: Path) -> None:
    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["screening", "interaction"], default="screening")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--out-dir", type=Path, default=Path("runs/snr_sweeps"))
    ap.add_argument("--rng-seed", type=int, default=123)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x]
    rng = random.Random(args.rng_seed)

    if args.stage == "screening":
        trials = screening_space(args.trials, seeds, rng)
    else:
        trials = interaction_space(seeds)

    rows: List[Dict[str, str]] = []
    for i, cfg in enumerate(trials, start=1):
        print(f"[{i}/{len(trials)}] {cfg.benchmark} seed={cfg.seed} id={cfg.config_id()}")
        rows.append(run_trial(cfg, args.out_dir))

    write_manifest(rows, args.out_dir)
    print(f"Done. Manifest: {args.out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
