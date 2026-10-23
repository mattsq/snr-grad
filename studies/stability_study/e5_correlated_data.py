"""
E5: correlated examples break the Sigma / B scaling -- how badly, and in which direction?

PLRF checkpoint (alpha=0.9, SGD, 0.5 x edge, step 750). A training batch of B=64
rows contains 64/r distinct examples, each repeated r times. The true variance of
the batch-mean gradient is r * tr(Sigma) / 64. A K=8 probe (microbatches of 8)
estimates it as trace_sigma / 64 under the independence assumption, with two
ways of splitting the batch:

  grouped      duplicates land in the same microbatch (split by example id);
  interleaved  copies of an example are spread across microbatches.

Also reported: the grouped probe with the correct unit count (microbatch_size =
8 / r distinct examples), which recovers per-example Sigma.

Run: uv run python studies/stability_study/e5_correlated_data.py
"""

import torch

from common import bp, load_edges, save
from snr_grad import StabilityProbe


def main():
    torch.set_num_threads(1)
    edges = load_edges()
    cfg = bp.Config()
    prob = bp.PLRF(cfg, 0.9)
    lr = 0.5 * edges[("sgd", 0.9)]
    _, snaps = bp.train(prob, "sgd", lr, 1, 751, checkpoints=(750,))
    theta = snaps[750][0][0]
    sigma = prob.sigma(theta)
    tr_sigma = float(sigma.diagonal().sum())
    B, K, b = 64, 8, 8
    gen = torch.Generator().manual_seed(0)
    reps = 300
    out = []
    p = torch.zeros(prob.D, dtype=bp.DTYPE)
    for r in (1, 2, 4, 8):
        n_distinct = B // r
        est = {"grouped": [], "interleaved": [], "grouped_correct_units": []}
        for _ in range(reps):
            # Per-example gradients of the distinct examples.
            u = torch.randn(n_distinct, prob.D + 1, generator=gen, dtype=bp.DTYPE) @ prob.F.t()
            z, y = u[:, : prob.D], u[:, prob.D]
            g_ex = z * ((z @ theta) - y)[:, None]
            rows = g_ex.repeat_interleave(r, dim=0)            # grouped order: copies adjacent
            inter = g_ex.repeat(r, 1)                          # interleaved: copies spread
            for key, data, units in (("grouped", rows, b), ("interleaved", inter, b),
                                     ("grouped_correct_units", rows, max(1, b // r))):
                micro = data.reshape(K, b, prob.D).mean(1)
                m = StabilityProbe([{p: micro[k]} for k in range(K)], units).measure()
                est[key].append(m.trace_sigma / (B if key != "grouped_correct_units" else n_distinct))
        truth = r * tr_sigma / B
        row = {"r": r, "true_batch_mean_variance": truth}
        for key, vals in est.items():
            row[f"{key}_ratio"] = float(torch.tensor(vals).mean()) / truth
        out.append(row)
        print(f"r={r}: estimated / true batch-mean variance  grouped={row['grouped_ratio']:.3f}  "
              f"interleaved={row['interleaved_ratio']:.3f}  grouped+correct units="
              f"{row['grouped_correct_units_ratio']:.3f}")
    path = save("e5_correlated_data.json", {"rows": out, "tr_sigma": tr_sigma, "reps": reps})
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
