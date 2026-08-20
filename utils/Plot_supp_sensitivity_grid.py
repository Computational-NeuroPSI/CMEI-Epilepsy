"""
Render the supplementary robustness figure from sensitivity_results.csv.

Layout: rows = metric, columns = IMP fraction. Each subpanel is a
(parameter x perturbation level) heatmap, cell values = mean over seeds
(annotated as mean, with +/- SD underneath when N_SEEDS > 1).

The center column (level 0%) is the Figure-3 reference and is identical across
the three parameter rows within a panel -- a built-in sanity check.

Reads the CSV written by supp_sensitivity_analysis.py. By default it plots the
IMP population.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

CSV = "../simulations/sensitivity_results.csv"
#CSV = "../simulations/sensitivity_results_tiny.csv"
POP = "IMP"                       # population to display
OUT_PNG = "supp_sensitivity_grid.png"
OUT_PDF = "supp_sensitivity_grid.pdf"
OUT_SVG = "supp_sensitivity_grid.svg"

# display order / labels
FRACTIONS = ["10p", "50p", "100p"]
FRACTION_LABELS = {"10p": "10% IMP", "50p": "50% IMP", "100p": "100% IMP"}
PARAMS = ["C", "tau_w", "p_conn"]
PARAM_LABELS = {"C": r"$C$", "tau_w": r"$\tau_w$", "p_conn": r"$p_{\mathrm{conn}}$"}
LEVELS = [-0.20, -0.10, 0.0, 0.10, 0.20]
LEVEL_LABELS = [f"{int(l*100):+d}%" if l != 0 else "0%" for l in LEVELS]

METRICS = [
    ("mean_sttc", "Mean STTC", "viridis"),
    ("pct_ictal", "% time ictal", "magma"),
    ("mean_rate", "Mean rate (Hz)", "cividis"),
    ("cv_rate",   "Rate CV", "plasma"),
]


def cell_arrays(df, metric):
    """
    For a given (fraction, metric): return mean and sd arrays of shape
    (len(PARAMS), len(LEVELS)) averaged over seeds.
    """
    mean = np.full((len(PARAMS), len(LEVELS)), np.nan)
    sd = np.full((len(PARAMS), len(LEVELS)), np.nan)
    for i, param in enumerate(PARAMS):
        for j, lvl in enumerate(LEVELS):
            sel = df[(df.perturb_param == param) &
                     (np.isclose(df.level, lvl))]
            if len(sel):
                mean[i, j] = sel[metric].mean()
                sd[i, j] = sel[metric].std(ddof=0)
    return mean, sd


def main():
    df = pd.read_csv(CSV)
    df = df[df["pop"] == POP]
    n_seeds = df["seed"].nunique()

    nrows, ncols = len(METRICS), len(FRACTIONS)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.4 * ncols, 2.5 * nrows),
                             squeeze=False)

    for r, (metric, mlabel, cmap) in enumerate(METRICS):
        # shared color scale per metric row (across fractions)
        allvals = []
        for frac in FRACTIONS:
            sub = df[df.imp_fraction == frac]
            m, _ = cell_arrays(sub, metric)
            allvals.append(m)
        allvals = np.concatenate([a.ravel() for a in allvals])
        allvals = allvals[np.isfinite(allvals)]
        vmin = float(np.nanmin(allvals)) if allvals.size else 0.0
        vmax = float(np.nanmax(allvals)) if allvals.size else 1.0
        if vmin == vmax:
            vmax = vmin + 1e-6

        im = None
        for c, frac in enumerate(FRACTIONS):
            ax = axes[r][c]
            sub = df[df.imp_fraction == frac]
            mean, sd = cell_arrays(sub, metric)

            im = ax.imshow(mean, aspect="auto", cmap=cmap,
                           vmin=vmin, vmax=vmax, origin="upper")

            # annotate each cell
            for i in range(len(PARAMS)):
                for j in range(len(LEVELS)):
                    if not np.isfinite(mean[i, j]):
                        continue
                    txt = f"{mean[i, j]:.2f}"
                    if n_seeds > 1 and np.isfinite(sd[i, j]):
                        txt += f"\n±{sd[i, j]:.2f}"
                    # contrast-aware text color
                    norm = (mean[i, j] - vmin) / (vmax - vmin)
                    color = "white" if norm < 0.5 else "black"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=7, color=color)

            ax.set_xticks(range(len(LEVELS)))
            ax.set_xticklabels(LEVEL_LABELS, fontsize=7)
            ax.set_yticks(range(len(PARAMS)))
            ax.set_yticklabels([PARAM_LABELS[p] for p in PARAMS], fontsize=9)

            if r == 0:
                ax.set_title(FRACTION_LABELS[frac], fontsize=10)
            if c == 0:
                ax.set_ylabel(mlabel, fontsize=9)
            if r == nrows - 1:
                ax.set_xlabel("perturbation", fontsize=8)

        # one colorbar per metric row
        cbar = fig.colorbar(im, ax=list(axes[r]), fraction=0.025, pad=0.02)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Parameter sensitivity of the reference network "
        f"($Z_0=-50$ mV, Tau ratio $=1$) — {POP} population"
        + (f", mean ± SD over {n_seeds} seeds" if n_seeds > 1 else ""),
        fontsize=11, y=0.995,
    )
    fig.savefig('../simulations/figures/'+OUT_PNG, dpi=200, bbox_inches="tight")
    #fig.savefig(OUT_PDF, bbox_inches="tight")
    #fig.savefig(OUT_SVG, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
