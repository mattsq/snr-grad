"""
E4 (proposal Stage 2): optimizer timescales and SNR gating vs loss excursions.

All Adam-family variants share one learning-rate grid per spectrum (fractions of
the measured AdamW edge), so differences are not artefacts of per-variant edges.
For each variant: excursion events per run, fraction of runs with events, median
max excursion, onset step of the first event, events in the first vs second half
(does gating *suppress* or merely *delay* spikes?), final and second-half mean excess loss, and steps to
reach a common target (2x the best final excess of any variant) (is "fewer spikes" just "slower learning"?).

Run: uv run python studies/stability_study/e4_timescales.py [--quick]
"""

import argparse
import time

import torch
import torch.nn as nn

from common import bp, load_edges, save
from snr_grad import SNRAdamW, detect_excursions

VARIANTS = {
    "adamw_b2=0.95": lambda ps, lr: torch.optim.AdamW(ps, lr=lr, betas=(0.9, 0.95), weight_decay=0.0),
    "adamw_b2=0.99": lambda ps, lr: torch.optim.AdamW(ps, lr=lr, betas=(0.9, 0.99), weight_decay=0.0),
    "adamw_b2=0.999": lambda ps, lr: torch.optim.AdamW(ps, lr=lr, betas=(0.9, 0.999), weight_decay=0.0),
    "snr_rho=0.9": lambda ps, lr: SNRAdamW(ps, lr=lr, gate="snr", rho=0.9),
    "snr_rho=0.99": lambda ps, lr: SNRAdamW(ps, lr=lr, gate="snr", rho=0.99),
    "snr_rho=0.999": lambda ps, lr: SNRAdamW(ps, lr=lr, gate="snr", rho=0.999),
    "snr_lam=0.25": lambda ps, lr: SNRAdamW(ps, lr=lr, gate="snr", lambda_pop=0.25),
    "snr_lam=4": lambda ps, lr: SNRAdamW(ps, lr=lr, gate="snr", lambda_pop=4.0),
    "soft_lam=1": lambda ps, lr: SNRAdamW(ps, lr=lr, gate="soft", lambda_pop=1.0),
    "soft_lam=4": lambda ps, lr: SNRAdamW(ps, lr=lr, gate="soft", lambda_pop=4.0),
    "snr_b2=0.95": lambda ps, lr: SNRAdamW(ps, lr=lr, betas=(0.9, 0.95), gate="snr"),
}


def train(prob, factory, lr, seeds, steps, data_seed=1000):
    theta = nn.Parameter(torch.zeros(seeds, prob.D, dtype=bp.DTYPE))
    opt = factory([theta], lr)
    gen = torch.Generator().manual_seed(data_seed)
    losses = torch.empty(seeds, steps + 1, dtype=bp.DTYPE)
    with torch.no_grad():
        losses[:, 0] = prob.loss(theta)
    dead = torch.zeros(seeds, dtype=torch.bool)
    l0 = float(losses[0, 0])
    for t in range(steps):
        with torch.no_grad():
            g = prob.minibatch_grad(theta.detach(), gen)
            g[dead] = 0.0
            theta.grad = g
        opt.step()
        with torch.no_grad():
            lt = prob.loss(theta)
            dead |= ~torch.isfinite(lt) | (lt > 1e6 * l0)
            theta.data[dead] = 0.0
            lt[dead] = float("inf")
            losses[:, t + 1] = lt
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    edges = load_edges()
    cfg = bp.Config()
    alphas, mults, seeds, steps = (0.5, 0.9), (0.3, 0.6, 1.0), 8, 4000
    variants = VARIANTS
    if args.quick:
        alphas, mults, seeds, steps = (0.9,), (0.6,), 2, 600
        variants = {k: VARIANTS[k] for k in ("adamw_b2=0.999", "snr_rho=0.99")}
    t0 = time.perf_counter()
    rows = []
    for a in alphas:
        prob = bp.PLRF(cfg, a)
        for mult in mults:
            lr = mult * edges[("adamw", a)]
            curves = {}
            for key, factory in variants.items():
                curves[key] = train(prob, factory, lr, seeds, steps)
            # Strict common target: 2x the best final median excess among all variants.
            best = min(float((c[:, -1] - prob.loss_star).median()) for c in curves.values())
            target = 2.0 * best + prob.loss_star
            for key, losses in curves.items():
                reps = [detect_excursions(losses[s], window=cfg.window, k_mad=cfg.k_mad,
                                          rel_floor=cfg.rel_floor) for s in range(seeds)]
                half = steps // 2
                first = [r.event_steps[0] for r in reps if r.event_steps]
                hit = []
                for s in range(seeds):
                    idx = torch.nonzero(losses[s] <= target)
                    hit.append(int(idx[0]) if idx.numel() else None)
                reached = [h for h in hit if h is not None]
                row = {
                    "alpha": a, "lr_mult_of_adamw_edge": mult, "lr": lr, "variant": key,
                    "events_per_run": sum(r.num_events for r in reps) / seeds,
                    "frac_runs_with_events": sum(r.spiked for r in reps) / seeds,
                    "divergence_fraction": sum(r.diverged for r in reps) / seeds,
                    "max_excursion_median": float(torch.tensor([r.max_excursion for r in reps]).median()),
                    "first_event_median": float(torch.tensor(first, dtype=torch.float64).median()) if first else None,
                    "events_first_half": sum(sum(e < half for e in r.event_steps) for r in reps) / seeds,
                    "events_second_half": sum(sum(e >= half for e in r.event_steps) for r in reps) / seeds,
                    "final_excess_median": float((losses[:, -1] - prob.loss_star).median()),
                    "second_half_mean_excess_median": float(
                        (losses[:, half:] - prob.loss_star).clamp_max(1e6).mean(1).median()),
                    "target_excess": target - prob.loss_star,
                    "frac_reached_target": len(reached) / seeds,
                    "steps_to_target_median": float(torch.tensor(reached, dtype=torch.float64).median()) if reached else None,
                }
                rows.append(row)
                print(f"a={a} mult={mult} {key:<16} ev/run={row['events_per_run']:6.2f} "
                      f"maxexc={row['max_excursion_median']:8.3f} h1/h2={row['events_first_half']:5.1f}/"
                      f"{row['events_second_half']:5.1f} final={row['final_excess_median']:.3g} "
                      f"reach={row['frac_reached_target']:.2f}@{row['steps_to_target_median']} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    path = save("e4_timescales" + ("_quick" if args.quick else "") + ".json",
                {"rows": rows, "seconds": time.perf_counter() - t0})
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
