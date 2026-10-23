"""Figures for the stability study (reads results/*.json, writes results/*.png)."""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "results")


def load(name):
    path = os.path.join(R, name)
    return json.load(open(path)) if os.path.exists(path) else None


def e2():
    d = load("e2_estimator_calibration.json")
    if not d:
        return
    rows = d["rows"]
    Ks = sorted({r["K"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for noise, ls in (("plrf", "-"), ("gaussian", "--")):
        def mean(key, K):
            v = [r[key] for r in rows if r["K"] == K and r["noise"] == noise and r[key] is not None]
            return sum(v) / len(v) if v else float("nan")
        axes[0].plot(Ks, [mean("R_rel_rmse", K) for K in Ks], ls, marker="o", label=f"{noise}")
        axes[1].plot(Ks, [mean("cover_2se", K) for K in Ks], ls, marker="o", label=f"+/-2 SE ({noise})")
        axes[1].plot(Ks, [mean("cover_log_t", K) for K in Ks], ls, marker="s", label=f"log-t CI ({noise})")
        axes[2].plot(Ks, [mean("D_rel_bias_dof4", K) for K in Ks], ls, marker="o", label=f"{noise}")
    axes[1].axhline(0.95, color="k", lw=0.8)
    axes[2].axhline(0.0, color="k", lw=0.8)
    for ax, t in zip(axes, ("relative RMSE of R", "interval coverage", "relative bias of D_eff")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("K microbatches")
        ax.set_title(t, fontsize=10)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "e2_estimator_calibration.png"), dpi=120)
    plt.close(fig)


def e4():
    d = load("e4_timescales.json")
    if not d:
        return
    rows = d["rows"]
    alphas = sorted({r["alpha"] for r in rows})
    fig, axes = plt.subplots(1, len(alphas), figsize=(6 * len(alphas), 4.2), squeeze=False)
    for ax, a in zip(axes[0], alphas):
        variants = sorted({r["variant"] for r in rows})
        cmap = plt.get_cmap("tab20")
        for i, v in enumerate(variants):
            rs = [r for r in rows if r["alpha"] == a and r["variant"] == v]
            style = "-" if v.startswith("adamw") else "--" if v.startswith("snr") else ":"
            ax.plot([r["second_half_mean_excess_median"] for r in rs],
                    [r["events_per_run"] + 0.05 for r in rs], style, marker="o", ms=4, lw=1.0,
                    color=cmap(i % 20), label=v)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("second-half mean excess loss (median over seeds)")
        ax.set_ylabel("excursion events per run (+0.05)")
        ax.set_title(f"alpha_spectrum={a}; points = lr 0.3/0.6/1.0 x AdamW edge", fontsize=9)
    axes[0][0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "e4_timescales.png"), dpi=120)
    plt.close(fig)


def bars(name, scores, title, key="mean", out=None):
    items = [(k, v[key]) for k, v in scores.items() if isinstance(v.get(key), float) and not math.isnan(v[key])]
    items.sort(key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(items) + 1))
    ax.barh([k for k, _ in items], [v for _, v in items])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(R, out or name), dpi=120)
    plt.close(fig)


def e1():
    d = load("e1_decoupling.json")
    if not d:
        return
    bars("e1_vary_B.png", d["scores_vary_B"], "E1: mean within-stratum Spearman, varying batch size")
    bars("e1_vary_beta.png", d["scores_vary_beta"], "E1: mean within-stratum Spearman, varying target exponent")


def e3():
    d = load("e3_simulator.json")
    if not d:
        return
    sc = {k: v for k, v in d["scores"].items() if k.endswith("@200")}
    bars("e3_rank.png", sc, "E3: Spearman(sim severity, real severity), horizon 200",
         key="spearman_severity")
    bars("e3_calibration.png", sc, "E3: event-fraction bias (sim - real), horizon 200",
         key="event_fraction_bias")


def e6():
    d = load("e6_network_scale.json")
    if not d:
        return
    summ = d["summary"]
    widths = sorted(summ, key=int)
    keys = [k[:-9] for k in summ[widths[0]] if k.endswith("_spearman")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for k in keys:
        axes[0].plot([int(w) for w in widths], [summ[w][f"{k}_spearman"] for w in widths], marker="o", label=k)
        axes[1].plot([int(w) for w in widths],
                     [summ[w][f"{k}_seconds"] / summ[w]["real_seconds"] for w in widths], marker="o", label=k)
    axes[0].set_ylabel("Spearman with real severity")
    axes[1].set_ylabel("simulator seconds / real-continuation seconds")
    axes[1].axhline(1.0, color="k", lw=0.8)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("hidden width")
        ax.legend(fontsize=7)
    axes[1].set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(R, "e6_network_scale.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    for f in (e1, e2, e3, e4, e6):
        f()
    print("figures written to", R)
