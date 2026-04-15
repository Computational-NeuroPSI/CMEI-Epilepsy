import json
import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
from matplotlib.ticker import MaxNLocator
import pandas as pd

import brian2 as b2

from utils.NN_simulation_analysis_helper import *

color = {
    'FS':  '#cb181d',
    'RS':  '#238b45',
    'IMP': '#2171b5'
}

#--------------------------------------------------
def plotting_3_traces_per_population(pop1 = None, 
                                     pop2 = None, 
                                     pop3 = None,
                                     ext_input_0 = None,
                                     ext_input_1 = None,
                                     ext_input_2 = None,
                                     ext_input_3 = None):
    if pop3 != None:
        fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    else:
        fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(pop1.t/b2.second, pop1.v[0] / b2.mV, color='#67000d')
    ax[0].plot(pop1.t/b2.second, pop1.v[1] / b2.mV, color=color['FS'])
    ax[0].plot(pop1.t/b2.second, pop1.v[2] / b2.mV, color='#fb6a4a')
    ax[0].set_title('Selected FS traces')
    
    ax[1].plot(pop2.t/b2.second, pop2.v[0] / b2.mV, color='#00441b')
    ax[1].plot(pop2.t/b2.second, pop2.v[1] / b2.mV, color=color['RS'])
    ax[1].plot(pop2.t/b2.second, pop2.v[2] / b2.mV, color='#74c476')
    ax[1].set_title('Selected RS traces')

    if pop3 != None:
        ax[2].plot(pop3.t/b2.second, pop3.v[0] / b2.mV, color='#08306b')
        ax[2].plot(pop3.t/b2.second, pop3.v[1] / b2.mV, color=color['IMP'])
        ax[2].plot(pop3.t/b2.second, pop3.v[2] / b2.mV, color='#6baed6')
        ax[2].set_title('Selected IMP traces')

    if ext_input_0 != None:
        ax[-1].plot(ext_input_0.t / b2.second, ext_input_0.i, '.', color='k', alpha=0.3, markersize=1)
    ax[-1].plot(ext_input_1.t / b2.second, ext_input_1.i, '.', color=color['FS'], markersize=1)
    ax[-1].plot(ext_input_2.t / b2.second, ext_input_2.i, '.', color=color['RS'], markersize=1)
    if ext_input_3 != None:
        ax[-1].plot(ext_input_3.t / b2.second, ext_input_3.i, '.', color=color['IMP'], markersize=1)    
    ax[-1].set_ylabel('External Input Neuron index')
    ax[-1].set_title('External Poisson input spike raster')
    
    ax[-1].set_xlabel('Time (ms)')
    ax[1].set_ylabel('Membrane potential (mV)')
    
    plt.tight_layout()
    return fig

# -------------------- #                 
def get_pretty_voltage(volt, thresh):
    for i in range(len(volt) - 1):
        if volt[i] > thresh * b2.mV and volt[i+1] < volt[i]:
            volt[i] = 0 * b2.mV #forcing peak to be 0 mV
    return volt
    
#--------------------------------------------------
def network_raster_plot(pop1 = None, 
                        pop2 = None, 
                        pop3 = None,
                        N_pop1 = None,
                        N_pop2 = None,
                        N_pop3 = None,
                        marker = None, 
                        markersize = None,
                        x_lim = None):
    
    if markersize == None:
        m_size = 1
    else:
        m_size = markersize

    if marker == None:
        marker = '.'    
    
    fig = plt.figure(figsize=(10, 6))
    offset = 0


        
    if pop1 != None:
        # Plot FS interneurons
        plt.plot(pop1.t / b2.second, pop1.i + offset, marker=marker, linestyle='',  color=color['FS'], label='FS', markersize=m_size)
        offset += N_pop1

    if pop2 != None:    
        # Plot RS excitatory neurons
        plt.plot(pop2.t / b2.second, pop2.i + offset, marker=marker, linestyle='', color=color['RS'], label='RS', markersize=m_size)
        offset += N_pop2

    if pop3 != None:    
    # Plot impaired models
        plt.plot(pop3.t / b2.second, pop3.i + offset, marker=marker, linestyle='', color=color['IMP'], label='IMP', markersize=m_size)
    
    if x_lim != None:
        plt.xlim(x_lim)
    
    yticks = []

    offset = 0
    if N_pop1 is not None:
        yticks.append(offset)
        offset += N_pop1

    if N_pop2 is not None:
        yticks.append(offset)
        offset += N_pop2

    if N_pop3 is not None:
        yticks.append(offset)
        offset += N_pop3
        yticks.append(offset)
        
    ax = plt.gca()
    ax.set_yticks(yticks)

    plt.xlabel('Time (s)')
    plt.ylabel('Neuron index (with offsets)')
    plt.title('Network Raster Plot')
    
    # Force y-axis to use integer ticks
    #ax = plt.gca()
    #ax.yaxis.set_major_locator(MaxNLocator(integer=True))
     
    return fig

#--------------------------------------------------
def network_raster_plot_h5(pop1 = None, 
                        pop2 = None, 
                        pop3 = None,
                        N_pop1 = None,
                        N_pop2 = None,
                        N_pop3 = None,
                        markersize = None,
                        x_lim = None):
    
    if markersize == None:
        m_size = 1
    else:
        m_size = markersize
        
    fig = plt.figure(figsize=(10, 6))
    offset = 0

    if pop1 != None:
        # Plot FS interneurons
        plt.plot(pop1['t'], pop1['i'] + offset, ',', color=color['FS'], label='FS', markersize=m_size)
        offset += N_pop1

    if pop2 != None:    
        # Plot RS excitatory neurons
        plt.plot(pop2['t'], pop2['i'] + offset, ',', color=color['RS'], label='RS', markersize=m_size)
        offset += N_pop2

    if pop3 != None:    
    # Plot impaired models
        plt.plot(pop3['t'], pop3['i'] + offset, ',', color=color['IMP'], label='IMP', markersize=m_size)
    
    if x_lim != None:
        plt.xlim(x_lim)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron index (with offsets)')
    plt.title('Network Raster Plot')
    
    # Force y-axis to use integer ticks
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    return fig

#--------------------------------------------------
def plotting_pop_firing_rate(pop1 = None, 
                            pop2 = None, 
                            pop3 = None,
                            bin_window = None):

    fig = plt.figure(figsize=(10, 6))
    if bin_window == None:
        bin_window = 100.1*b2.ms
    
    plt.plot(pop1.t / b2.second, pop1.smooth_rate(window='flat', width=bin_window) / b2.Hz, label='avg FS freq', color=color['FS'])
    plt.plot(pop2.t / b2.second, pop2.smooth_rate(window='flat', width=bin_window) / b2.Hz, label='avg RS freq', color=color['RS'])
    if pop3 != None:
        plt.plot(pop3.t / b2.second, pop3.smooth_rate(window='flat', width=bin_window) / b2.Hz, label='avg IMP freq', color=color['IMP'])
    plt.xlabel('Time (s)')
    plt.ylabel('Firing rate (Hz)')
    plt.title(f'Smoothed population firing rate ({bin_window} s window)')
    plt.legend()
    return fig

#--------------------------------------------------
def plotting_pop_freq_and_std(sim_duration = None, 
                              pop1 = None, 
                              pop2 = None, 
                              pop3 = None,
                              N_pop1 = None,
                              N_pop2 = None,
                              N_pop3 = None,
                              bin_size = None,
                              ylim = None):

    # Parameters
    if bin_size == None:
        bin_size = 0.1 * b2.second
    bin_edges = np.arange(0, sim_duration + bin_size, bin_size)
    time_bins = bin_edges[:-1]
    
    # Create spike matrix: (n_neurons, n_time_bins)
    spike_matrix_FS = np.zeros((N_pop1, len(time_bins)))
    spike_matrix_RS = np.zeros((N_pop2, len(time_bins)))
    if pop3 != None:
        spike_matrix_IMP = np.zeros((N_pop3, len(time_bins)))
    
    # Fill the spike matrix
    for i, t in zip(pop1.i, pop1.t / b2.second):
        bin_idx = int(t // bin_size)
        if bin_idx < len(time_bins):
            spike_matrix_FS[i, bin_idx] += 1
    
    for i, t in zip(pop2.i, pop2.t / b2.second):
        bin_idx = int(t // bin_size)
        if bin_idx < len(time_bins):
            spike_matrix_RS[i, bin_idx] += 1

    if pop3 != None:
        for i, t in zip(pop3.i, pop3.t / b2.second):
            bin_idx = int(t // bin_size)
            if bin_idx < len(time_bins):
                spike_matrix_IMP[i, bin_idx] += 1
    
    # Convert to rate (Hz)
    spike_matrix_FS /= bin_size
    spike_matrix_RS /= bin_size
    if pop3 != None:
        spike_matrix_IMP /= bin_size
    
    
    # Compute mean and std
    mean_rate_FS = np.mean(spike_matrix_FS, axis=0)
    std_rate_FS = np.std(spike_matrix_FS, axis=0)
    mean_rate_RS = np.mean(spike_matrix_RS, axis=0)
    std_rate_RS = np.std(spike_matrix_RS, axis=0)
    if pop3 != None:
        mean_rate_IMP = np.mean(spike_matrix_IMP, axis=0)
        std_rate_IMP = np.std(spike_matrix_IMP, axis=0)
    
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(time_bins, mean_rate_FS, '.-', label='avg FS freq', color=color['FS'])
    plt.plot(time_bins, mean_rate_RS, '.-', label='avg RS freq', color=color['RS'])
    if pop3 != None:
        plt.plot(time_bins, mean_rate_IMP, '.-', label='avg IMP freq', color=color['IMP'])
    
    plt.fill_between(
        time_bins,
        (mean_rate_FS - std_rate_FS).clip(0 * b2.Hz, np.inf * b2.Hz),
        (mean_rate_FS + std_rate_FS).clip(0 * b2.Hz, np.inf * b2.Hz),
        color=color['FS'], alpha=0.3, label='± FS std'
    )

    plt.fill_between(
        time_bins,
        (mean_rate_RS - std_rate_RS).clip(0 * b2.Hz, np.inf * b2.Hz),
        (mean_rate_RS + std_rate_RS).clip(0 * b2.Hz, np.inf * b2.Hz),
        color=color['RS'], alpha=0.3, label='± RS std'
    )

    if pop3 is not None:
        plt.fill_between(
            time_bins,
            (mean_rate_IMP - std_rate_IMP).clip(0 * b2.Hz, np.inf * b2.Hz),
            (mean_rate_IMP + std_rate_IMP).clip(0 * b2.Hz, np.inf * b2.Hz),
            color=color['IMP'], alpha=0.3, label='± IMP std'
        )
        
    plt.xlabel('Time (s)')
    plt.ylabel(f'Firing rate (Hz, bin={int(bin_size*1000)} ms)')
    plt.title('Network Population firing rate ± std')
    plt.legend()
    plt.tight_layout()
    if ylim is not None:
        plt.ylim(ylim)   
    return fig

#--------------------------------------------------
def plotting_single_pop_freq_and_std(sim_duration = None,
                                     neuron_model = None, 
                                     pop = None, 
                                     N_pop = None,
                                     bin_size = None):

    # Parameters
    if bin_size == None:
        bin_size = 0.1  # seconds
    bin_edges = np.arange(0, sim_duration / b2.second + bin_size, bin_size)
    time_bins = bin_edges[:-1]
    
    # Create spike matrix: (n_neurons, n_time_bins)
    spike_matrix = np.zeros((N_pop, len(time_bins)))
    
    # Fill the spike matrix
    for i, t in zip(pop.i, pop.t / b2.second):
        bin_idx = int(t // bin_size)
        if bin_idx < len(time_bins):
            spike_matrix[i, bin_idx] += 1
      
    # Convert to rate (Hz)
    spike_matrix /= bin_size
        
    # Compute mean and std
    mean_rate = np.mean(spike_matrix, axis=0)
    std_rate = np.std(spike_matrix, axis=0)
    
    if neuron_model == 'FS':
        color = color['FS']
    if neuron_model == 'RS':
        color = color['RS']
    if neuron_model == 'IMP':
        color = color['IMP']
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(time_bins, mean_rate, label='avg freq', color = color)
    
    plt.fill_between(
        time_bins,
        #np.clip(mean_rate - std_rate, 0 * b2.Hz, None),    # to avoid negative firing rate
        (mean_rate - std_rate).clip(0 * b2.Hz, np.inf * b2.Hz),
        mean_rate + std_rate,
        color= color, alpha=0.3, label='± std'
    )
    
    
    plt.xlabel('Time (s)')
    plt.ylabel(f'Firing rate (Hz, bin={int(bin_size*1000)} ms)')
    plt.title(f'Network {neuron_model} Population firing rate ± std')
    plt.legend()
    plt.tight_layout()
    plt.ylim(-10, 275) 
    return fig

#--------------------------------------------------
def plotting_pop_freq_and_std_h5(sim_duration = None, 
                              pop1 = None, 
                              pop2 = None, 
                              pop3 = None,
                              N_pop1 = None,
                              N_pop2 = None,
                              N_pop3 = None,
                              bin_size = None):

    # Parameters
    if bin_size == None:
        bin_size = 0.1 * b2.second
    bin_edges = np.arange(0, sim_duration + bin_size, bin_size)
    time_bins = bin_edges[:-1]
    
    # Create spike matrix: (n_neurons, n_time_bins)
    spike_matrix_FS = np.zeros((N_pop1, len(time_bins)))
    spike_matrix_RS = np.zeros((N_pop2, len(time_bins)))
    if pop3 != None:
        spike_matrix_IMP = np.zeros((N_pop3, len(time_bins)))
    
    # Fill the spike matrix
    for i, t in zip(pop1['i'], pop1['t'] / b2.second):
        bin_idx = int(t // bin_size)
        if bin_idx < len(time_bins):
            spike_matrix_FS[i, bin_idx] += 1
    
    for i, t in zip(pop2['i'], pop2['t'] / b2.second):
        bin_idx = int(t // bin_size)
        if bin_idx < len(time_bins):
            spike_matrix_RS[i, bin_idx] += 1

    if pop3 != None:
        for i, t in zip(pop3['i'], pop3['t'] / b2.second):
            bin_idx = int(t // bin_size)
            if bin_idx < len(time_bins):
                spike_matrix_IMP[i, bin_idx] += 1
    
    # Convert to rate (Hz)
    spike_matrix_FS /= bin_size
    spike_matrix_RS /= bin_size
    if pop3 != None:
        spike_matrix_IMP /= bin_size
    
    
    # Compute mean and std
    mean_rate_FS = np.mean(spike_matrix_FS, axis=0)
    std_rate_FS = np.std(spike_matrix_FS, axis=0)
    mean_rate_RS = np.mean(spike_matrix_RS, axis=0)
    std_rate_RS = np.std(spike_matrix_RS, axis=0)
    if pop3 != None:
        mean_rate_IMP = np.mean(spike_matrix_IMP, axis=0)
        std_rate_IMP = np.std(spike_matrix_IMP, axis=0)
    
    
    fig = plt.figure(figsize=(8, 6))
    plt.plot(time_bins, mean_rate_FS, '.-', label='avg FS freq', color='#cb181d')
    plt.plot(time_bins, mean_rate_RS, '.-', label='avg RS freq', color='#238b45')
    if pop3 != None:
        plt.plot(time_bins, mean_rate_IMP, '.-', label='avg IMP freq', color='#2171b5')
    
    plt.fill_between(
        time_bins,
        #np.clip(mean_rate_FS - std_rate_FS, 0*b2.Hz, None),    # to avoid negative firing rate
        (mean_rate_FS - std_rate_FS).clip(0 * b2.Hz, np.inf * b2.Hz),
        mean_rate_FS + std_rate_FS,
        color='#cb181d', alpha=0.3, label='± FS std'
    )
    
    plt.fill_between(
        time_bins,
        #np.clip(mean_rate_RS - std_rate_RS, 0*b2.Hz, None),    # to avoid negative firing rate
        (mean_rate_RS - std_rate_RS).clip(0 * b2.Hz, np.inf * b2.Hz),
        mean_rate_RS + std_rate_RS,
        color='#238b45', alpha=0.3, label='± RS std'
    )

    if pop3 != None:
        plt.fill_between(
            time_bins,
            #np.clip(mean_rate_IMP - std_rate_IMP, 0*b2.Hz, None),  # to avoid negative firing rate
            (mean_rate_IMP - std_rate_IMP).clip(0 * b2.Hz, np.inf * b2.Hz),
            mean_rate_IMP + std_rate_IMP,
            color='#2171b5', alpha=0.3, label='± IMP std'
        )
    
    plt.xlabel('Time (s)')
    plt.ylabel(f'Firing rate (Hz, bin={int(bin_size*1000)} ms)')
    plt.title('Network Population firing rate ± std')
    plt.legend()
    plt.tight_layout()
   
    return fig, time_bins, mean_rate_FS, mean_rate_RS, mean_rate_IMP

#--------------------------------------------------
def plot_regime_timeline(results, split_time=42):

    pops = ['FS', 'RS', 'IMP']
    y_positions = {pop: i for i, pop in enumerate(pops)}

    fig, ax = plt.subplots(figsize=(7, 4))

    for pop in pops:
        y = y_positions[pop]

        # ICTAL (purple)
        for start, duration, rate in results[f'ictal_{pop}']:
            ax.broken_barh([(start, duration)], (y - 0.3, 0.25),
                           facecolors='#e7298a')

        # INTERICTAL (indigo)
        for start, duration, rate in results[f'interictal_{pop}']:
            ax.broken_barh([(start, duration)], (y + 0.05, 0.25),
                           facecolors='#bfd3e6')

    # Vertical split line
    ax.axvline(split_time, linestyle='--', color = 'k')

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(pops)

    ax.set_xlabel("Time (s)")
    ax.set_title("Ictal / Interictal Regimes")

    plt.tight_layout()
    return fig

#--------------------------------------------------
def plot_regime_timeline_h5(ictal, interictal, pops, split_time=42):

    y_positions = {pop: i for i, pop in enumerate(pops)}

    fig, ax = plt.subplots(figsize=(7, 4))

    for pop in pops:
        y = y_positions[pop]

        # ICTAL (purple)
        for start, duration, rate in ictal[f'{pop}']:
            ax.broken_barh([(start, duration)], (y - 0.3, 0.25),
                           facecolors='#e7298a')

        # INTERICTAL (indigo)
        for start, duration, rate in interictal[f'{pop}']:
            ax.broken_barh([(start, duration)], (y + 0.05, 0.25),
                           facecolors='#bfd3e6')

    # Vertical split line
    ax.axvline(split_time, linestyle='--', color = 'k')

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(pops)

    ax.set_xlabel("Time (s)")
    ax.set_title("Ictal / Interictal Regimes")

    plt.tight_layout()
    return fig

#--------------------------------------------------
def plot_ictal_regime_duration(results, pops, split_time=42):

    fig, ax = plt.subplots(figsize=(6, 4))

    dur = []
    labels = []

    for pop in pops:
        pre, post = split_intervals(results[f'ictal_{pop}'], split_time)

        dur.append(durations(pre))
        labels.append(f'{pop}-on')

        dur.append(durations(post))
        labels.append(f'{pop}-off')

    ax.boxplot(dur, medianprops={'color': '#e7298a', 'linewidth': 2.5})    
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("Duration (s)")
    ax.set_title("Ictal Duration (On vs Off)")
    plt.tight_layout()
    return fig

#--------------------------------------------------
def plot_ictal_regime_duration_h5(ictal, pops, split_time=42):

    fig, ax = plt.subplots(figsize=(6, 4))

    dur = []
    labels = []

    for pop in pops:
        pre, post = split_intervals(ictal[f'{pop}'], split_time)

        dur.append(durations(pre))
        labels.append(f'{pop}-on')

        dur.append(durations(post))
        labels.append(f'{pop}-off')

    ax.boxplot(dur, medianprops={'color': '#e7298a', 'linewidth': 2.5})    
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("Duration (s)")
    ax.set_title("Ictal Duration (On vs Off)")
    plt.tight_layout()
    return fig

#--------------------------------------------------
def plot_interictal_regime_duration(results, pops, split_time=42):
    
    fig, ax = plt.subplots(figsize=(6, 4))

    dur = []
    labels = []

    for pop in pops:
        pre, post = split_intervals(results[f'interictal_{pop}'], split_time)

        dur.append(durations(pre))
        labels.append(f'{pop}-on')

        dur.append(durations(post))
        labels.append(f'{pop}-off')

    ax.boxplot(dur, medianprops={'color': '#bfd3e6', 'linewidth': 2.5})    
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("Duration (s)")
    ax.set_title("Interictal Duration (On vs Off)")
    plt.tight_layout()
    return fig
    
#--------------------------------------------------    
def plot_interictal_regime_duration_h5(interictal, pops, split_time=42):
    
    fig, ax = plt.subplots(figsize=(6, 4))

    dur = []
    labels = []

    for pop in pops:
        pre, post = split_intervals(interictal[f'{pop}'], split_time)

        dur.append(durations(pre))
        labels.append(f'{pop}-on')

        dur.append(durations(post))
        labels.append(f'{pop}-off')

    ax.boxplot(dur, medianprops={'color': '#bfd3e6', 'linewidth': 2.5})    
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("Duration (s)")
    ax.set_title("Interictal Duration (On vs Off)")
    plt.tight_layout()
    return fig

#--------------------------------------------------
def plot_regime_summary(time_bins,
                        pops,
                        mean_rate_FS,
                        mean_rate_RS,
                        mean_rate_IMP,
                        results,
                        split_time=62):

    colors = {
        'ictal': '#e7298a',
        'interictal': '#bfd3e6'
    }

    fig = plt.figure(figsize=(8, 8))

    # =========================
    # PANEL A — Ictal / Interictal Regimes
    # =========================
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)

    y_positions = {pop: i for i, pop in enumerate(pops)}

    for pop in pops:
        y = y_positions[pop]

        for start, duration, _ in results[f'ictal_{pop}']:
            ax1.broken_barh([(start, duration)], (y - 0.3, 0.25),
                            facecolors=colors['ictal'])

        for start, duration, _ in results[f'interictal_{pop}']:
            ax1.broken_barh([(start, duration)], (y + 0.05, 0.25),
                            facecolors=colors['interictal'])

    ax1.axvline(split_time, linestyle='--', color = 'k')
    ax1.set_yticks(list(y_positions.values()))
    ax1.set_yticklabels(pops)
    ax1.set_xlabel("Time (s)")
    ax1.set_title("A. Ictal / Interictal Regimes")

    # =========================
    # PANEL B — Ictal Duration (Pre vs Post)
    # =========================
    ax2 = plt.subplot2grid((2, 2), (1, 0))

    dur = []
    labels = []

    for pop in pops:
        pre, post = split_intervals(results[f'ictal_{pop}'], split_time)

        dur.append(durations(pre))
        labels.append(f'{pop}-pre')

        dur.append(durations(post))
        labels.append(f'{pop}-post')

    ax2.boxplot(dur, medianprops={'color': '#e7298a', 'linewidth': 2.5})    
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_ylabel("Duration (s)")
    ax2.set_title("B. Ictal Duration (Pre vs Post)")

    # =========================
    # PANEL C — Duration vs Rate
    # =========================
    ax3 = plt.subplot2grid((2, 2), (1, 1))

    for pop in pops:
        pre, post = split_intervals(results[f'ictal_{pop}'], split_time)

        d_pre, r_pre = duration_rate(pre)
        d_post, r_post = duration_rate(post)

        ax3.scatter(d_pre, r_pre, marker = '<', color = color[pop], label=f'{pop}-pre', alpha=1)
        ax3.scatter(d_post, r_post, marker = '>', color = color[pop], label=f'{pop}-post', alpha=1)

    ax3.set_xlabel("Duration (s)")
    ax3.set_ylabel("Mean Rate (Hz)")
    ax3.set_title("C. Duration vs Rate")
    ax3.legend()

    plt.tight_layout()
    return fig

#--------------------------------------------------
def plot_synch_panels_h5(pops, ictal, interictal, sync_FS, sync_RS, sync_IMP, 
                          time_bins, split_time=62):
    fig = plt.figure(figsize=(10,7)) 
    rows = 3

    # --- ROW 1: STTC Synchrony A---
    ax0 = plt.subplot2grid((rows, 2), (0, 0), colspan=2)
    ax0.plot(time_bins, sync_FS, color=color['FS'], label='FS (STTC)')
    ax0.plot(time_bins, sync_RS, color=color['RS'], label='RS (STTC)')
    ax0.plot(time_bins, sync_IMP, color=color['IMP'], label='IMP (STTC)')
    ax0.axvline(split_time, color='black', linestyle='--')
    ax0.set_title("A. Synchrony over time (STTC)")
    ax0.set_ylabel("STTC Correlation Index")
    ax0.set_ylim(-0.1, 1.1) 
    ax0.legend()

    # --- DATA PREPARATION ---
    sync_ictal = {}
    sync_inter = {}
    for pop, sync in zip(pops, [sync_FS, sync_RS, sync_IMP]):
        sync_ictal[pop] = compute_interval_synchrony(ictal[f'{pop}'], time_bins, sync)
        sync_inter[pop] = compute_interval_synchrony(interictal[f'{pop}'], time_bins, sync)

    # --- ROW 1: ICTAL ANALYSES (B & C) ---
    axB = plt.subplot2grid((rows, 2), (1, 0)) 
    axC = plt.subplot2grid((rows, 2), (1, 1)) 
    
    sync_data_ictal = []
    labels_ictal = []
    for pop in pops:
        d = [x[1] for x in sync_ictal[pop]]
        s = [x[2] for x in sync_ictal[pop]]
        axB.scatter(d, s, color=color[pop], alpha=0.7)
        
        pre, post = split_intervals(sync_ictal[pop], split_time)
        sync_data_ictal.extend([[x[2] for x in pre], [x[2] for x in post]])
        labels_ictal.extend([f'{pop}\nPre', f'{pop}\nPost'])

    axB.set_title("B. ICTAL: Sync vs Duration")
    axB.set_xlabel("Duration (s)")
    axB.set_ylabel("Synchrony Index")
    axC.boxplot(sync_data_ictal, medianprops={'color': '#e7298a', 'linewidth': 2.5})
    axC.set_xticklabels(labels_ictal, rotation=45)
    axC.set_title("C. ICTAL: Pre vs Post")

    # --- ROW 3: INTERICTAL ANALYSES (D & E) ---
    axD = plt.subplot2grid((rows, 2), (2, 0))
    axE = plt.subplot2grid((rows, 2), (2, 1))
    
    sync_data_inter = []
    labels_inter = []
    for pop in pops:
        d = [x[1] for x in sync_inter[pop]]
        s = [x[2] for x in sync_inter[pop]]
        axD.scatter(d, s, color=color[pop], alpha=0.7)
        
        pre, post = split_intervals(sync_inter[pop], split_time)
        sync_data_inter.extend([[x[2] for x in pre], [x[2] for x in post]])
        labels_inter.extend([f'{pop}\nPre', f'{pop}\nPost'])

    axD.set_title("D. INTERICTAL: Sync vs Duration")
    axD.set_xlabel("Duration (s)")
    axD.set_ylabel("Synchrony Index")
    axE.boxplot(sync_data_inter, medianprops={'color': '#bfd3e6', 'linewidth': 2.5})
    axE.set_xticklabels(labels_inter, rotation=45)
    axE.set_title("E. INTERICTAL: Pre vs Post")

    plt.tight_layout()
    return fig





