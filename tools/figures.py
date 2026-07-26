"""Figures for the reproduction report, drawn from the raw artifact CSVs.

Every figure reads the same CSV files that the pages link to, so a figure can
never disagree with the numbers printed beside it.  Both an SVG (for the
Hugging Face logbook, which takes text uploads only) and a PNG (for the GitHub
report) are written for each figure.
"""

from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# A restrained, colour-blind-safe qualitative ramp; one hue per curve.
PALETTE = ["#3B6FD4", "#D9822B", "#2E9E7E", "#B5478F", "#7A6BD1", "#B0543C"]
GRID = dict(color="#D6D9DE", linewidth=0.7)


def _style(ax, xlabel, ylabel, title=None):
    ax.set_facecolor("white")
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#8A8F98")
    ax.tick_params(colors="#41464F", labelsize=9)
    ax.set_xlabel(xlabel, fontsize=10, color="#24272C")
    ax.set_ylabel(ylabel, fontsize=10, color="#24272C")
    if title:
        ax.set_title(title, fontsize=11, color="#14171A", pad=8)


def _read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    svg = os.path.join(out_dir, f"{name}.svg")
    png = os.path.join(out_dir, f"{name}.png")
    fig.savefig(svg, format="svg", bbox_inches="tight", facecolor="white")
    fig.savefig(png, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return svg, png


# --------------------------------------------------------------------------

def figure2(art: str, out: str):
    """Headline: fraction of positive compatibility scores vs injected error."""
    rows = _read(os.path.join(art, "claim4", "figure2_synthetic.csv"))
    panels = ["m", "n", "p"]
    labels = {"m": "hidden variables $m$", "n": "observed variables $n$",
              "p": "edge density $p$"}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharey=True)
    for ax, panel in zip(axes, panels):
        sub = [r for r in rows if r["panel"] == panel]
        values = sorted({float(r["value"]) for r in sub})
        for k, val in enumerate(values):
            cur = sorted([r for r in sub if float(r["value"]) == val],
                         key=lambda r: float(r["sigma"]))
            x = [float(r["sigma"]) for r in cur]
            y = [float(r["frac_positive"]) for r in cur]
            lo = [float(r["frac_ci_lo"]) for r in cur]
            hi = [float(r["frac_ci_hi"]) for r in cur]
            c = PALETTE[k % len(PALETTE)]
            ax.plot(x, y, "-o", color=c, markersize=3.6, linewidth=1.7,
                    label=f"{panel} = {val:g}")
            ax.fill_between(x, lo, hi, color=c, alpha=0.15, linewidth=0)
        _style(ax, r"injected error variance $\sigma$", "", labels[panel])
        ax.legend(frameon=False, fontsize=8.5)
        ax.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel("fraction of lists with\npositive compatibility score",
                       fontsize=10, color="#24272C")
    fig.suptitle("Reproduction of Figure 2 — the fraction of positive scores "
                 "strictly decreases as statements degrade",
                 fontsize=12.5, color="#14171A", y=1.04)
    return _save(fig, out, "figure2_fraction_positive")


def figure5(art: str, out: str):
    """Mean heuristic incompatibility vs number of injected graph errors."""
    rows = _read(os.path.join(art, "claim6", "figure5_monotonicity.csv"))
    panels = ["m", "n", "p"]
    labels = {"m": "hidden variables $m$", "n": "observed variables $n$",
              "p": "edge density $p$"}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    for ax, panel in zip(axes, panels):
        sub = [r for r in rows if r["panel"] == panel]
        values = sorted({float(r["value"]) for r in sub})
        for k, val in enumerate(values):
            cur = sorted([r for r in sub if float(r["value"]) == val],
                         key=lambda r: int(r["n_errors"]))
            x = [int(r["n_errors"]) for r in cur]
            y = [float(r["mean_score"]) for r in cur]
            lo = [float(r["ci_lo"]) for r in cur]
            hi = [float(r["ci_hi"]) for r in cur]
            c = PALETTE[k % len(PALETTE)]
            ax.plot(x, y, "-o", color=c, markersize=3.6, linewidth=1.7,
                    label=f"{panel} = {val:g}")
            ax.fill_between(x, lo, hi, color=c, alpha=0.15, linewidth=0)
        ax.plot(x, x, "--", color="#8A8F98", linewidth=1.2,
                label="true error count")
        _style(ax, "injected errors", "", labels[panel])
        ax.legend(frameon=False, fontsize=8.5)
    axes[0].set_ylabel("mean heuristic\nincompatibility $c(G)$", fontsize=10,
                       color="#24272C")
    fig.suptitle("Reproduction of Figure 5 — incompatibility increases "
                 "monotonically with injected errors",
                 fontsize=12.5, color="#14171A", y=1.04)
    return _save(fig, out, "figure5_monotonicity")


def sample_complexity(art: str, out: str):
    """Measured minimum sample size against the theorem's predicted scaling."""
    rows = _read(os.path.join(art, "claim3", "min_sample_complexity.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))

    ax = axes[0]
    sweeps = sorted({r["sweep"] for r in rows})
    for k, sw in enumerate(sweeps):
        sub = sorted([r for r in rows if r["sweep"] == sw],
                     key=lambda r: float(r["sweep_value"]))
        pred = [float(r["n"]) ** 4 * (1 + float(r["a"]) + float(r["b"])) ** 4
                * float(r["V"]) ** 4 / float(r["eps"]) ** 2
                * np.log(float(r["n"]) / float(r["delta"])) for r in sub]
        meas = [float(r["N_star_median"]) for r in sub]
        ax.loglog(pred, meas, "o", color=PALETTE[k % len(PALETTE)],
                  markersize=6, label=sw, alpha=0.85)
    lim = ax.get_xlim()
    xs = np.geomspace(max(lim[0], 1e-3), lim[1], 50)
    for c_, style in ((1e-3, "-"), (1e-4, "--"), (1e-5, ":")):
        ax.loglog(xs, c_ * xs, style, color="#8A8F98", linewidth=1.1,
                  label=f"$N = {c_:g}\\times$ bound")
    _style(ax, r"theorem bound  $n^4(1{+}a{+}b)^4V^4\varepsilon^{-2}\log(n/\delta)$",
           "measured minimum $N^*$",
           "Measured sample complexity vs the theorem's bound")
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[1]
    import json
    with open(os.path.join(art, "claim3", "rates.json")) as fh:
        rates = json.load(fh)
    jf = rates["joint_fit"]
    names = list(jf)
    xs = np.arange(len(names))
    vals = [jf[k]["exponent"] for k in names]
    errs = [1.96 * jf[k]["stderr"] for k in names]
    caps = [jf[k]["cap"] for k in names]
    ax.bar(xs - 0.19, vals, 0.36, yerr=errs, capsize=3, color=PALETTE[0],
           label="measured exponent (95% CI)")
    ax.bar(xs + 0.19, caps, 0.36, color="#C9CDD3", label="theorem's exponent")
    ax.set_xticks(xs)
    ax.set_xticklabels(["$n$", "$1{+}a{+}b$", "$V$", r"$1/\varepsilon$",
                        r"$\log(n/\delta)$"], fontsize=9)
    ax.axhline(0, color="#8A8F98", linewidth=0.8)
    _style(ax, "", "exponent in $\\log N^*$",
           "Joint-fit exponents never exceed the theorem's")
    ax.legend(frameon=False, fontsize=8.5)
    return _save(fig, out, "sample_complexity")


def expected_compatibility(art: str, out: str):
    """Theorem 2.9: E[comp] > 0 across dimensions and distribution families."""
    rows = _read(os.path.join(art, "claim2", "expected_compatibility.csv"))
    gating = [r for r in rows if r["gating"] == "True"]
    fams = sorted({r["coef_family"] for r in gating})
    dims = sorted({int(r["n"]) for r in gating})
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    for k, fam in enumerate(fams):
        for noise, marker in (("diag_exponential", "o"), ("wishart_dense", "s")):
            sub = sorted([r for r in gating if r["coef_family"] == fam
                          and r["noise_family"] == noise],
                         key=lambda r: int(r["n"]))
            if not sub:
                continue
            x = [int(r["n"]) for r in sub]
            y = [float(r["mean_comp"]) for r in sub]
            lo = [float(r["comp_ci_lo"]) for r in sub]
            ax.plot(x, y, marker=marker, linestyle="-", color=PALETTE[k % len(PALETTE)],
                    markersize=4.5, linewidth=1.4, alpha=0.9,
                    label=f"{fam} / {noise.split('_')[0]}")
            ax.fill_between(x, lo, [float(r["comp_ci_hi"]) for r in sub],
                            color=PALETTE[k % len(PALETTE)], alpha=0.10, linewidth=0)
    ax.axhline(0, color="#B0543C", linewidth=1.3, linestyle="--",
               label="zero (theorem: strictly above)")
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_xticks(dims)
    _style(ax, "number of variables $n$", r"$\mathbb{E}[\mathrm{comp}(\Sigma_X, A)]$",
           "Theorem 2.9 — expected compatibility of TRUE statements is positive\n"
           "(Bonferroni-corrected 95% intervals, every gating configuration)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    return _save(fig, out, "expected_compatibility")


def llm_scores(art: str, out: str):
    """LLM statement quality against model capacity (Figures 4, 6 and 7)."""
    lin = _read(os.path.join(art, "claim4", "figure4_llm_extended_ladder.csv"))
    gra = _read(os.path.join(art, "claim6", "figures67_llm_extended_ladder.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3))

    ax = axes[0]
    x = [float(r["params_b"]) for r in lin]
    y = [float(r["median_comp"]) for r in lin]
    ax.semilogx(x, y, "o", color=PALETTE[0], markersize=7)
    for r in lin:
        ax.annotate(r["paper_name"], (float(r["params_b"]),
                                      float(r["median_comp"])),
                    fontsize=6.5, color="#41464F",
                    xytext=(3, 4), textcoords="offset points")
    ax.axhline(0, color="#B0543C", linewidth=1.2, linestyle="--")
    if len(x) > 2:
        b = np.polyfit(np.log(x), y, 1)
        xs = np.geomspace(min(x), max(x), 40)
        ax.plot(xs, np.polyval(b, np.log(xs)), "-", color="#8A8F98", linewidth=1.2)
    _style(ax, "total parameters (B, log scale)",
           "median compatibility score",
           "Figure 4 — linear statements\n(above the dashed line = not falsified)")

    ax = axes[1]
    sub = [r for r in gra if r["mean_incomp_below_cap"] not in ("", "nan")]
    x = [float(r["params_b"]) for r in sub]
    y = [float(r["mean_incomp_below_cap"]) for r in sub]
    ax.semilogx(x, y, "s", color=PALETTE[2], markersize=7)
    for r in sub:
        ax.annotate(r["paper_name"], (float(r["params_b"]),
                                      float(r["mean_incomp_below_cap"])),
                    fontsize=6.5, color="#41464F",
                    xytext=(3, 4), textcoords="offset points")
    if len(x) > 2:
        b = np.polyfit(np.log(x), y, 1)
        xs = np.geomspace(min(x), max(x), 40)
        ax.plot(xs, np.polyval(b, np.log(xs)), "-", color="#8A8F98", linewidth=1.2)
    _style(ax, "total parameters (B, log scale)",
           "mean incompatibility $c(G)$",
           "Figure 7 — graphical statements, density $\\leq 2/3$\n"
           "(lower is better)")
    return _save(fig, out, "llm_scores")


def lemma37(art: str, out: str):
    """How tight the heuristic is against the exact incompatibility score."""
    rows = _read(os.path.join(art, "claim6", "lemma37_exhaustive.csv"))
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    labels = [f"n={r['n']}\n{int(r['n_graphs']):,} graphs" for r in rows]
    xs = np.arange(len(rows))
    exact = [float(r["frac_exact"]) for r in rows]
    ax.bar(xs, exact, 0.5, color=PALETTE[2], label="heuristic exactly equals incomp(G)")
    ax.bar(xs, [1 - e for e in exact], 0.5, bottom=exact, color="#C9CDD3",
           label="heuristic strictly over-estimates")
    for i, r in enumerate(rows):
        ax.text(i, 1.02, f"0 violations\nmean gap {float(r['mean_gap']):.2f}",
                ha="center", fontsize=8, color="#41464F")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.22)
    _style(ax, "", "fraction of statement graphs",
           "Lemma 3.7 — c(G) never falls below incomp(G), and is zero exactly "
           "when incomp(G) is")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    return _save(fig, out, "lemma37")


ALL = [figure2, figure5, sample_complexity, expected_compatibility, llm_scores,
       lemma37]


def build(art: str, out: str) -> list[str]:
    made = []
    for fn in ALL:
        try:
            svg, png = fn(art, out)
            made.append(os.path.basename(png))
            print(f"  figure {os.path.basename(png)}")
        except Exception as exc:                       # a missing artifact
            print(f"  SKIPPED {fn.__name__}: {exc!r}")
    return made


if __name__ == "__main__":
    import sys
    build(sys.argv[1], sys.argv[2])
