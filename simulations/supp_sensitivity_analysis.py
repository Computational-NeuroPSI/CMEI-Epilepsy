"""
Supplementary robustness / sensitivity analysis.

Re-runs the reference network conditions of Figure 3 (Z0 = -50 mV, Tau ratio = 1)
while perturbing three otherwise-fixed parameters one at a time:

    - C      : membrane capacitance (membrane time constant), scaled globally
    - tau_w  : adaptation time constant, scaled globally
    - p_conn : RECURRENT connection probability (external input connectivity held fixed)

for three network compositions (IMP 10% / 50% / 100%).

For each condition it stores summary metrics computed over the stimulation window (mean STTC, % time ictal, mean rate, and the temporal CV of the population rate). 
Output is a CSV that can be plotted using plot_supp_sensitivity_grid.py.
"""

import os
import sys
import csv

import numpy as np
import brian2 as b2

# --- make the utils/ package importable (same as before) ------
cwd = os.getcwd()
parent_dir = os.path.abspath(os.path.join(cwd, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.Brian_function_helper import setting_simulation_Brian_taus
from utils.Network_helper import get_network_config
from utils.NN_simulation_analysis_helper import (
    compute_population_rate,
    extract_all_regimes,
    compute_sliding_sttc_dardel,
)

# =============================================================================
# CONFIG 
# =============================================================================

OUT_CSV = "sensitivity_results_tiny.csv"
DATA_FOLDER = "../AdEx_models"
NETWORK_CONFIG_FOLDER = "network_config_files"

# Baseline neuron models: Z0 = -50 mV, Tau ratio = 1 (the Figure-3 reference).
MODELS = {
    "FS":  "FS_cr_tau_r_1",
    "RS":  "RS_cr_tau_r_1",
    "IMP": "IMP_cr_Z0_-50_tau_r_1",
}

# Network compositions -> config file (gives FS/RS/IMP counts and baseline conn_prob).
IMP_FRACTIONS = {
    "10p":  "network_config_file_sim_10p.json",
    "50p":  "network_config_file_sim_50p.json",
    "100p": "network_config_file_sim_100p.json",
}

# --- Stimulation protocol -----------------------------------------------------
### This is just a tiny run ###
SIM_DURATION = 6.0   # s
P_START      = 1.0    # s  (stimulation onset)
P_END        = 4.0   # s  (stimulation offset)
DT           = 0.1 * b2.ms
BIN_SIZE     = 0.1    # s
### But for the manuscript we used: 
#SIM_DURATION = 65.0   # s
#P_START      = 5.0    # s
#P_END        = 45.0   # s
#DT           = 0.1 * b2.ms
#BIN_SIZE     = 0.1    # s

# --- Drive (Poisson) ----------------------------------------------------------
# TODO: set to the production Figure-3 values. Defaults mirror the v0 test script.
INPUT_FS  = 1.0   # Hz
INPUT_RS  = 1.0   # Hz
INPUT_IMP = 1.0   # Hz
# background "all" rate is read from the config file (background_input.freq)

# --- Quantal conductances (as in NN_FS_RS_IMP_sim_taus_80p_v0.py) -------------
QE = 1.5 * b2.nS
QI = 5.0 * b2.nS

# --- Perturbation grid --------------------------------------------------------
# Same relative levels for every parameter so the heatmap grid is rectangular.
LEVELS = [-0.20, -0.10, 0.0, 0.10, 0.20]     # fraction change from default
PARAMS = ["C", "tau_w", "p_conn"]            # perturbed one at a time

# Number of independent network realizations per condition (mean +/- SD).
# Main cost multiplier. Start at 1 for a smoke test, then raise to 3-5.
N_SEEDS = 3
BASE_SEED = 12345

# Ictal / interictal thresholds (as before).
ICTAL_THRESH    = {"FS": 100, "RS": 100, "IMP": 125}
INTERICTAL_THRESH = {"FS": 10, "RS": 10, "IMP": 10}
MIN_DURATION = 0.5   # s  (500 ms)

# STTC parameters (as before; dt_corr = 2 ms matches the manuscript).
STTC_WINDOW = 0.2      # s (200 ms sliding window)
STTC_DT     = 0.002    # s (+/- 2 ms correlation window)
STTC_PAIR_FRACTION = 0.1   # number of sampled pairs = 10% of population size

# Populations to report in the CSV (IMP is primary; FS kept for completeness).
REPORT_POPS = ["IMP", "FS"]

# =============================================================================
# Helpers
# =============================================================================

def make_timed_array(times, dt, rate, p_start, p_end):
    """Poisson rate = `rate` inside [p_start, p_end], else 0."""
    arr = b2.zeros(len(times)) * b2.Hz
    arr[(times >= p_start) & (times < p_end)] = rate
    return b2.TimedArray(arr, dt=dt)


def connect_input(P, G, conn_prob):
    """External excitatory Poisson drive onto group G (as in the repo)."""
    S = b2.Synapses(P, G, on_pre="GsynE_post+=Qe")
    S.connect(p=conn_prob)
    return S


def build_recurrent_synapses(pops, conn_prob, seed=None):
    """
    Reproduce network_creation's recurrent connectivity

    pops : dict like {'FS': G_inh, 'RS': G_exc, 'IMP': G_imp}; keys may be absent.
    FS is inhibitory (GsynI += Qi); RS and IMP are excitatory (GsynE += Qe).
    Within-population sets exclude self-connections (i != j), as in the repo.
    """
    if seed is not None:
        b2.seed(seed)   # control connectivity, matching network_creation

    syn = []
    order = [k for k in ("FS", "RS", "IMP") if k in pops]
    for src in order:
        for tgt in order:
            on_pre = "GsynI_post+=Qi" if src == "FS" else "GsynE_post+=Qe"
            S = b2.Synapses(pops[src], pops[tgt], on_pre=on_pre,
                            name=f"S_{src}_{tgt}")
            if src == tgt:
                S.connect("i!=j", p=conn_prob)
            else:
                S.connect(p=conn_prob)
            syn.append(S)
    return syn


def pct_time_in_intervals(intervals, win_start, win_end):
    """Percentage of [win_start, win_end] covered by `intervals` (start, dur, rate)."""
    win = max(win_end - win_start, 1e-12)
    covered = 0.0
    for start, dur, _ in intervals:
        lo = max(start, win_start)
        hi = min(start + dur, win_end)
        if hi > lo:
            covered += (hi - lo)
    return 100.0 * covered / win


def window_stats(values, time_bins, win_start, win_end):
    """mean, std, cv of a time series over [win_start, win_end]."""
    time_bins = np.asarray(time_bins)
    values = np.asarray(values)
    n = min(len(values), len(time_bins))
    tb, v = time_bins[:n], values[:n]
    mask = (tb >= win_start) & (tb < win_end)
    seg = v[mask]
    if seg.size == 0:
        return 0.0, 0.0, 0.0
    m = float(np.mean(seg))
    s = float(np.std(seg))
    cv = s / m if m > 1e-12 else 0.0
    return m, s, cv


# =============================================================================
# Core: run one condition, return summary metrics
# =============================================================================

def run_condition(imp_key, config_file, perturb_param, factor, seed):
    """
    Build and run one network under a single-parameter perturbation.
    `factor` multiplies the default value of `perturb_param` ('C', 'tau_w', 'p_conn').
    Returns a dict of per-population summary metrics.
    """
    b2.start_scope()
    np.random.seed(seed)   # deterministic STTC pair sampling + Poisson given seed

    sim_duration = SIM_DURATION * b2.second
    times = b2.arange(0, sim_duration, DT)
    bin_edges = np.arange(0, SIM_DURATION + BIN_SIZE, BIN_SIZE)
    time_bins = bin_edges[:-1] + BIN_SIZE / 2.0

    cfg = get_network_config(
        json_file_name=os.path.join(NETWORK_CONFIG_FOLDER, config_file)
    )
    comp = cfg["network_composition"]
    N = {"FS": comp["FS_neuron"], "RS": comp["RS_neuron"], "IMP": comp["IMP_neuron"]}
    conn_prob = comp["conn_prob"]
    N_external_exc = cfg["external_input"]["N_external_exc"]
    rate_bg = cfg["background_input"]["freq"] * b2.Hz

    # --- apply the connectivity perturbation (recurrent only) -----------------
    if perturb_param == "p_conn":
        conn_prob = float(np.clip(conn_prob * factor, 1e-6, 1.0))
    input_conn_prob = comp["conn_prob"]   # external drive connectivity: unperturbed

    # --- build the (non-empty) populations ------------------------------------
    pops, Ncell = {}, {}
    for pop in ("FS", "RS", "IMP"):
        if N[pop] <= 0:
            continue
        G = setting_simulation_Brian_taus(
            idx=0, N_cell=N[pop], neuron_model=MODELS[pop],
            json_file_name=os.path.join(DATA_FOLDER, MODELS[pop] + ".json"),
            curr_inj=None,
        )
        # apply intrinsic perturbations globally (Cm / tau_w are per-neuron attrs)
        if perturb_param == "C":
            G.Cm = G.Cm * factor
        elif perturb_param == "tau_w":
            G.tau_w = G.tau_w * factor
        pops[pop] = G
        Ncell[pop] = N[pop]

    # --- recurrent connectivity (tolerant of empty RS) ------------------------
    syn = build_recurrent_synapses(pops, conn_prob=conn_prob, seed=seed)

    # --- external Poisson drive (structure copied from the repo) --------------
    input_rate = {"FS": INPUT_FS * b2.Hz, "RS": INPUT_RS * b2.Hz, "IMP": INPUT_IMP * b2.Hz}
    ta_all = make_timed_array(times, DT, rate_bg, P_START * b2.second, P_END * b2.second)

    poisson_groups, input_syn = [], []
    namespace = {"ta_all": ta_all, "Qe": QE, "Qi": QI}

    P_all = b2.PoissonGroup(N_external_exc, rates="ta_all(t)")
    poisson_groups.append(P_all)
    for pop in pops:
        input_syn.append(connect_input(P_all, pops[pop], input_conn_prob))

    for pop in pops:
        ta = make_timed_array(times, DT, input_rate[pop],
                              P_START * b2.second, P_END * b2.second)
        key = f"ta_{pop}"
        namespace[key] = ta
        Pg = b2.PoissonGroup(N_external_exc, rates=f"{key}(t)")
        poisson_groups.append(Pg)
        input_syn.append(connect_input(Pg, pops[pop], input_conn_prob))

    # --- monitors -------------------------------------------------------------
    mon = {pop: b2.SpikeMonitor(pops[pop]) for pop in pops}

    # --- assemble an EXPLICIT network -----------------------------------------
    net = b2.Network()
    net.add(*pops.values())         # neuron groups
    net.add(*syn)                   # recurrent synapses
    net.add(*poisson_groups)        # external Poisson drive
    net.add(*input_syn)             # input synapses
    net.add(*mon.values())          # spike monitors

    # --- run ------------------------------------------------------------------
    net.run(sim_duration, namespace=namespace)

    # --- analysis -------------------------------------------------------------
    rate = {}
    for pop in ("FS", "RS", "IMP"):
        if pop in mon:
            rate[pop] = compute_population_rate(mon[pop].t, bin_edges, Ncell[pop])
        else:
            rate[pop] = np.zeros(len(time_bins))

    regimes = extract_all_regimes(
        time_bins, rate["FS"], rate["RS"], rate["IMP"],
        ICTAL_THRESH, INTERICTAL_THRESH, min_duration=MIN_DURATION,
    )

    metrics = {}
    for pop in REPORT_POPS:
        if pop not in mon:
            metrics[pop] = dict(mean_sttc=np.nan, pct_ictal=np.nan,
                                mean_rate=np.nan, cv_rate=np.nan)
            continue
        n_pairs = max(int(Ncell[pop] * STTC_PAIR_FRACTION), 1)
        sync = compute_sliding_sttc_dardel(
            mon[pop].t, mon[pop].i, time_bins, STTC_WINDOW,
            n_pairs=n_pairs, dt_corr=STTC_DT,
        )
        mean_sttc, _, _ = window_stats(sync, time_bins, P_START, P_END)
        mean_rate, _, cv_rate = window_stats(rate[pop], time_bins, P_START, P_END)
        pct_ictal = pct_time_in_intervals(regimes[f"ictal_{pop}"], P_START, P_END)
        metrics[pop] = dict(mean_sttc=mean_sttc, pct_ictal=pct_ictal,
                            mean_rate=mean_rate, cv_rate=cv_rate)
    return metrics


# =============================================================================
# Build the condition list and run the sweep
# =============================================================================

def build_conditions():
    """
    Yield (imp_key, config_file, param, factor, level, seed).
    The baseline (factor 1.0) is run once per (fraction, seed) and labelled under
    every parameter with level 0.0 so each parameter's heatmap has a center column.
    """
    conditions = []
    for imp_key, cfg_file in IMP_FRACTIONS.items():
        for s in range(N_SEEDS):
            seed = BASE_SEED + s
            # baseline once
            conditions.append((imp_key, cfg_file, "_baseline", 1.0, 0.0, seed))
            # perturbations (skip level 0.0, it's the baseline)
            for param in PARAMS:
                for level in LEVELS:
                    if abs(level) < 1e-12:
                        continue
                    conditions.append((imp_key, cfg_file, param, 1.0 + level, level, seed))
    return conditions


def main():
    conditions = build_conditions()
    print(f"Total conditions to run: {len(conditions)}")

    rows = []
    for k, (imp_key, cfg_file, param, factor, level, seed) in enumerate(conditions):
        print(f"[{k+1}/{len(conditions)}] {imp_key} | {param} "
              f"| factor={factor:.2f} | seed={seed}")
        metrics = run_condition(imp_key, cfg_file, param, factor, seed)

        # expand a baseline row into all three parameter columns (center column)
        param_labels = PARAMS if param == "_baseline" else [param]
        for plabel in param_labels:
            for pop in REPORT_POPS:
                m = metrics[pop]
                rows.append(dict(
                    imp_fraction=imp_key, perturb_param=plabel,
                    level=level, factor=factor, seed=seed, pop=pop,
                    mean_sttc=m["mean_sttc"], pct_ictal=m["pct_ictal"],
                    mean_rate=m["mean_rate"], cv_rate=m["cv_rate"],
                ))

    fieldnames = ["imp_fraction", "perturb_param", "level", "factor", "seed",
                  "pop", "mean_sttc", "pct_ictal", "mean_rate", "cv_rate"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    # Serial by default (robust), but parallelized on Dardel
    main()
