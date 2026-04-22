import h5py
import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt

color = {
    'FS':  '#cb181d',
    'RS':  '#238b45',
    'IMP': '#2171b5'
}

#--------------------------------------------------
def load_spike_h5(fname):
    """
    Load spikes and metadata from a single HDF5 file.
    
    Returns a dictionary with:
    - 'metadata': dict of scalar values
    - 'spikes': {
          'FS': {'i': array, 't': array},
          'RS': {'i': array, 't': array},
          'IMP': {'i': array, 't': array}
      }
    """
    data = {}
    with h5py.File(fname, "r") as f:
        meta = {}
        for k in f["metadata"].keys():
            meta[k] = f["metadata"][k][()]
        data["metadata"] = meta
        
        # population spike arrays
        spikes = {}
        for pop in ("FS", "RS", "IMP"):
            grp = f"/spikes/{pop}"
            i = f[grp + "/i"][()]
            t = f[grp + "/t"][()]
            # indices i are neuron ids (0..N-1) relative to that population
            spikes[pop] = {"i": i.astype(int), "t": t.astype(float)}
        data["spikes"] = spikes    
    return data

#--------------------------------------------------
def load_ictal_interictal_synchro_h5(fname):
    """Load spikes and metadata from a single HDF5 file following the described structure."""
    with h5py.File(fname, "r") as f:

        meta = {}
        for k in f["metadata"].keys():
            meta[k] = f["metadata"][k][()]
        
        # population ictal, interictal and synchro values
        ictal = {}
        interictal= {}
        synchro = {}

        for pop in ("FS", "RS", "IMP"):

            dat_ictal = f"/ictal/ictal_{pop}"
            ictal[pop] = f[dat_ictal][()]
            dat_interictal = f"/interictal/interictal_{pop}"
            interictal[pop] = f[dat_interictal][()]
            dat_synchro = f"/synchro/synchro_{pop}"
            synchro[pop] = f[dat_synchro][()]
            
    return meta, ictal, interictal, synchro

#--------------------------------------------------
def compute_population_rate(spike_times, bin_edges, N_neurons):
    """
    Compute population firing rate (Hz)
    """
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    dt = np.mean(np.diff(bin_edges))
    rate = counts / (N_neurons * dt)
    return rate
    
#--------------------------------------------------
def extract_regime_intervals(time_bins,
                             rate,
                             threshold,
                             condition='above',
                             min_duration=0.0):
    """
    Extract contiguous intervals where rate satisfies a condition.

    Parameters
    ----------
    time_bins : array
    rate : array
    threshold : float
    condition : 'above' or 'below'
    min_duration : float (seconds)

    Returns
    -------
    intervals : list of tuples
        (start_time, duration, mean_rate)
    """

    time_bins = np.asarray(time_bins)
    rate = np.asarray(rate)

    dt = np.mean(np.diff(time_bins))

    if condition == 'above':
        mask = rate > threshold
    elif condition == 'below':
        mask = rate < threshold
    else:
        raise ValueError("condition must be 'above' or 'below'")

    intervals = []
    in_interval = False
    start_idx = 0

    for i, val in enumerate(mask):
        if val and not in_interval:
            in_interval = True
            start_idx = i

        elif not val and in_interval:
            end_idx = i
            in_interval = False

            duration = (end_idx - start_idx) * dt

            if duration >= min_duration:
                start_time = time_bins[start_idx]
                mean_rate = np.mean(rate[start_idx:end_idx])
                intervals.append((start_time, duration, mean_rate))

    # Handle case where interval goes until the end
    if in_interval:
        end_idx = len(rate)
        duration = (end_idx - start_idx) * dt

        if duration >= min_duration:
            start_time = time_bins[start_idx]
            mean_rate = np.mean(rate[start_idx:end_idx])
            intervals.append((start_time, duration, mean_rate))

    return intervals

#--------------------------------------------------
def extract_all_regimes(time_bins,
                        mean_rate_FS,
                        mean_rate_RS,
                        mean_rate_IMP,
                        ictal_thresholds,
                        interictal_thresholds,
                        min_duration=0.0):

    rates = {
        'FS': mean_rate_FS,
        'RS': mean_rate_RS,
        'IMP': mean_rate_IMP
    }

    results = {}

    for pop in rates:
        rate = rates[pop]

        # ICTAL
        results[f'ictal_{pop}'] = extract_regime_intervals(
            time_bins,
            rate,
            threshold=ictal_thresholds[pop],
            condition='above',
            min_duration=min_duration
        )

        # INTERICTAL
        results[f'interictal_{pop}'] = extract_regime_intervals(
            time_bins,
            rate,
            threshold=interictal_thresholds[pop],
            condition='below',
            min_duration=min_duration
        )

    return results

    
#--------------------------------------------------
def split_intervals(intervals, split_time):
    pre, post = [], []
    for start, duration, rate in intervals:
        if start < split_time:
            pre.append((start, duration, rate))
        else:
            post.append((start, duration, rate))
    return pre, post

def durations(intervals):
    return [d for _, d, _ in intervals]

def duration_rate(intervals):
    d = [x[1] for x in intervals]
    r = [x[2] for x in intervals]
    return d, r

#--------------------------------------------------
def compute_synchrony_index(rate):
    """
    Variance-to-mean ratio of population rate
    No explicit phase 
    Detects coincident spiking/bursts
    """
    mean_r = np.mean(rate)
    var_r = np.var(rate)

    if mean_r == 0:
        return 0

    return var_r / mean_r

def compute_synchrony_over_time(rate, window_size, dt):
    window_bins = int(window_size / dt)
    half_win = window_bins // 2
    
    sync = np.zeros(len(rate)) # Pre-allocate array of SAME length
    
    # Slide the window but keep the index centered
    for i in range(len(rate)):
        # Define start/end, clipping at the boundaries of the data
        start = max(0, i - half_win)
        end = min(len(rate), i + half_win + 1)
        
        segment = rate[start:end]
        sync[i] = compute_synchrony_index(segment)
        
    return sync

#--------------------------------------------------
def compute_interval_synchrony(intervals, time_bins, sync):
    """
    Average synchrony within each interval
    """
    values = []

    for start, duration, _ in intervals:
        mask = (time_bins[:len(sync)] >= start) & \
               (time_bins[:len(sync)] < start + duration)

        segment = sync[mask]

        if len(segment) > 0:
            values.append((start, duration, np.mean(segment)))

    return values

def calculate_sttc(spike_times_1, spike_times_2, rec_duration, dt=0.005):
    """Calculates the Spike Time Tiling Coefficient."""
    if len(spike_times_1) == 0 or len(spike_times_2) == 0:
        return 0.0

    def get_time_tiled(spikes, dt, duration):
        if len(spikes) == 0: return 0
        # Create start/end for each spike and merge
        intervals = np.stack([spikes - dt, spikes + dt], axis=1)
        intervals = np.clip(intervals, 0, duration)
        # Fast merging of intervals
        intervals = intervals[intervals[:, 0].argsort()]
        if len(intervals) == 0: return 0
        merged = [intervals[0]]
        for curr in intervals[1:]:
            prev = merged[-1]
            if curr[0] <= prev[1]:
                merged[-1] = (prev[0], max(prev[1], curr[1]))
            else:
                merged.append(curr)
        tiled_time = sum(m[1] - m[0] for m in merged)
        return tiled_time / duration

    def get_p(spikes_a, spikes_b, dt):
        if len(spikes_a) == 0 or len(spikes_b) == 0: return 0
        # Find if any spike in B is within dt of spikes in A
        idx = np.searchsorted(spikes_b, spikes_a)
        # Check closest spike to the left and right
        left_dist = np.full(len(spikes_a), np.inf)
        right_dist = np.full(len(spikes_a), np.inf)
        valid_l = idx > 0
        valid_r = idx < len(spikes_b)
        left_dist[valid_l] = spikes_a[valid_l] - spikes_b[idx[valid_l]-1]
        right_dist[valid_r] = spikes_b[idx[valid_r]] - spikes_a[valid_r]
        return np.mean(np.minimum(left_dist, right_dist) <= dt)

    TA = get_time_tiled(spike_times_1, dt, rec_duration)
    TB = get_time_tiled(spike_times_2, dt, rec_duration)
    PA = get_p(spike_times_1, spike_times_2, dt)
    PB = get_p(spike_times_2, spike_times_1, dt)

    # Standard STTC formula
    num1, den1 = PA - TB, 1 - PA * TB
    num2, den2 = PB - TA, 1 - PB * TA
    
    # Handle edge cases where denominator is zero
    term1 = num1/den1 if abs(den1) > 1e-10 else 0
    term2 = num2/den2 if abs(den2) > 1e-10 else 0
    
    return 0.5 * (term1 + term2)

def compute_sliding_sttc(spike_data_pop, time_bins, window_size_sec, n_pairs=40, dt_corr=0.005):
    """Computes mean STTC across sampled pairs in a sliding window."""
    times, indices = spike_data_pop['t'], spike_data_pop['i']
    uids = np.unique(indices)
    sync_series = np.zeros(len(time_bins))
    
    if len(uids) < 2: return sync_series
    
    # Pre-sample pairs to keep it consistent
    pairs = [np.random.choice(uids, 2, replace=False) for _ in range(n_pairs)]
    half_win = window_size_sec / 2

    for i, t in enumerate(time_bins):
        t_start, t_end = t - half_win, t + half_win
        mask = (times >= t_start) & (times < t_end)
        w_t, w_i = times[mask], indices[mask]
        
        scores = []
        for id1, id2 in pairs:
            s1, s2 = w_t[w_i == id1], w_t[w_i == id2]
            if len(s1) > 2 and len(s2) > 2: # Min spikes for correlation
                scores.append(calculate_sttc(s1, s2, window_size_sec, dt=dt_corr))
        sync_series[i] = np.mean(scores) if scores else 0
    return sync_series

def compute_sliding_sttc_dardel(spike_data_pop_t, spike_data_pop_i, time_bins, window_size_sec, n_pairs=40, dt_corr=0.005):
    """Computes mean STTC across sampled pairs in a sliding window."""
    times, indices = spike_data_pop_t / b2.second, spike_data_pop_i
    uids = np.unique(indices)
    sync_series = np.zeros(len(time_bins))
    
    if len(uids) < 2: return sync_series
    
    # Pre-sample pairs to keep it consistent
    pairs = [np.random.choice(uids, 2, replace=False) for _ in range(n_pairs)]
    half_win = window_size_sec / 2

    for i, t in enumerate(time_bins):
        t_start, t_end = t - half_win, t + half_win
        mask = (times >= t_start) & (times < t_end)
        w_t, w_i = times[mask], indices[mask]
        
        scores = []
        for id1, id2 in pairs:
            s1, s2 = w_t[w_i == id1], w_t[w_i == id2]
            if len(s1) > 2 and len(s2) > 2: # Min spikes for correlation
                scores.append(calculate_sttc(s1, s2, window_size_sec, dt=dt_corr))
        sync_series[i] = np.mean(scores) if scores else 0
    return sync_series

