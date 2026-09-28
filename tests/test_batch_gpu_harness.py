"""CPU smoke test: the accelerator harness and its analysis run end to end."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
TINY = ["--device", "cpu", "--dtype", "fp32", "--synthetic", "4000", "--seq-len", "16",
        "--width", "16", "--depth", "1", "--heads", "2", "--micro-batch", "2",
        "--sizes", "4", "8", "--reference-batch", "4", "--initial-batch", "8",
        "--valid-windows", "8", "--measure-every", "1", "--controller-warmup", "1",
        "--controller-dwell", "1"]


def _run(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True, capture_output=True, text=True)


@pytest.mark.parametrize("optimizer", ["adamw", "snr_muon"])
def test_harness_modes_and_analysis(tmp_path, optimizer):
    pytest.importorskip("numpy")
    common = [*TINY, "--optimizer", optimizer]
    _run("benchmark_batch_gpu.py", "throughput", *common, "--throughput-steps", "2",
         "--throughput-warmup", "1", "--output", str(tmp_path / "tp.json"))
    assert set(json.loads((tmp_path / "tp.json").read_text())["seconds"]) == {"4", "8"}
    _run("benchmark_batch_gpu.py", "rollout", *common, "--seeds", "0", "1", "--time-budget", "0.3",
         "--eval-every", "0.1", "--step-times", str(tmp_path / "tp.json"),
         "--output", str(tmp_path / "roll.jsonl"))
    _run("benchmark_batch_gpu.py", "validate", *common, "--seeds", "0", "--time-budget", "0.3",
         "--checkpoints", "0.5", "--repetitions", "1", "--continuation-steps", "2",
         "--probe-batch", "8", "--output", str(tmp_path / "val.jsonl"))
    _run("analyze_batch_gpu.py", str(tmp_path / "roll.jsonl"), str(tmp_path / "val.jsonl"),
         "--tuning-seeds", "0", "--summary", str(tmp_path / "summary.json"))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert "controller:aware" in summary["rollout"][optimizer]["final"]
    assert summary["validate"][optimizer]["checkpoints"]
