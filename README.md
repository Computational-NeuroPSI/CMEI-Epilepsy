# CMEI-Epilepsy
CMEI-Epilepsy: Circadian Modulation of Excitation–Inhibition balance in epileptic networks

**This repository is associated with a paper currently under review, available on bioRxiv:**
[https://www.biorxiv.org/content/10.64898/2026.05.22.726242v1.article-info](https://www.biorxiv.org/content/10.64898/2026.05.22.726242v1.article-info)

## Overview
This repository provides the computational framework accompanying the manuscript. The model investigates how three interacting factors drive transitions between physiological and pathological (seizure-like) activity in spiking neural networks:
 
1. **Network composition** — the proportion of impaired neurons within the excitatory population
2. **Neuronal impairment** — altered intrinsic excitability in IMP neurons, controlled via the parameter `Z0`
3. **Synaptic kinetics** — circadian modulation of excitatory and inhibitory synaptic time constants (the `tau_ratio`), reflecting daily fluctuations in E/I balance
Networks consist of three neuronal populations: **Fast Spiking (FS)** inhibitory neurons, **Regular Spiking (RS)** excitatory neurons, and **Impaired (IMP)** excitatory neurons. Dynamics are characterised via mean firing rates and the Spike Time Tiling Coefficient (STTC), enabling joint assessment of excitability and synchrony across different parameter regimes.
 
The simulation scripts are set up with **80% IMP neurons as a working example**, but the network composition can be easily adjusted to explore different levels of pathological recruitment.

## Repository Structure
 
```
CMEI-Epilepsy/
│
├── AdEx_models/                  # Neuron model parameter files (JSON)
├── single_neuron_comparison/     # Single-neuron simulation notebooks
├── neural_network/               # Small network demo notebook
├── simulations/                  # Main simulation scripts and analysis notebooks
    └── network_config_files      # Json files for setting up the network containing different % of IMP neurons
│   └── testing_80p_v0/           # Pre-generated simulation output (HDF5 + figures)
├── utils/                        # Shared Python utility functions
└── paper_figures/                # Notebooks to reproduce manuscript figures
```
## Neuron Models — `AdEx_models/`
 
Each population is defined by a JSON file specifying parameters, initial conditions, and physical units. The models use the **Adaptive Exponential Integrate-and-Fire (AdEx)** formalism, with IMP neurons incorporating an additional slow variable `z` that captures ionic dysregulation.
 
## Single Neuron Comparison — `single_neuron_comparison/`
 
Two notebooks allow simulation and visualisation of isolated single neurons (FS, RS, IMP):
 
- **`_current_injection`** — stimulates neurons with current pulses and plots membrane voltage traces and firing rates.
- **`_synaptic_activation`** — drives neurons via a Poisson spike process and characterises the resulting synaptic responses.

## Small Network Demo — `neural_network/`
 
The notebook `small_neural_network_FS_RS_IMP_cr.ipynb` simulates a small network (10 neurons per population) over 30 seconds. It produces Raster plots and Population-averaged firing rate traces.
This serves as a quick sanity-check and visual introduction to network dynamics before running full-scale simulations.

## Simulations — `simulations/`
 
### Running a simulation
 
The main simulation script is:
 
```bash
python3 NN_FS_RS_IMP_sim_taus_v0.py
```
 
This sets up a network where **80% of the excitatory population are IMP neurons** and runs two simulations:
- `tau_ratio = 1.0` (baseline)
- `tau_ratio = 0.8` (circadian modulation)
Output is written to `testing_80p_v0/`, containing:
 
| File | Contents |
|------|----------|
| `spikes_FS_cr_tau_r_1_RS_cr_tau_r_1_IMP_cr_Z0_-50_tau_r_1.h5` | Spike data, tau ratio = 1.0 |
| `spikes_FS_cr_tau_r_1_RS_cr_tau_r_1_IMP_cr_Z0_-50_tau_r_08.h5` | Spike data, tau ratio = 0.8 |
| `ictal_interictal_synchro_IMP_cr_Z0_-50_tau_r_1.h5` | STTC + ictal/interictal classification, tau ratio = 1.0 |
| `ictal_interictal_synchro_IMP_cr_Z0_-50_tau_r_08.h5` | STTC + ictal/interictal classification, tau ratio = 0.8 |
 
> The `testing_80p_v0/` folder is pre-populated so that analysis notebooks can be run directly without re-running simulations.
 
### Analysis notebooks
 
- **`analysis_hdf5_file_80p_synch.ipynb`** — focused analysis of the STTC and ictal/interictal classification from the `ictal_interictal_synchro_` files.
- **`analysing_sim_folder.ipynb`** — comprehensive analysis using both HDF5 file types (spike data + synchrony), with full visualisation of firing rates, synchrony traces, and regime classification.
 
## Utilities — `utils/`
 
Python modules providing shared functions used across notebooks and simulation scripts, including spike train processing, STTC computation, ictal/interictal classification, and plotting routines.
 
## Manuscript Figures — `paper_figures/`
 
Notebooks to reproduce the figures in the associated manuscript are being added to this folder. Each notebook is self-contained and references the pre-generated data in `testing_xxx/` or can be rerun after new simulations.
 
## Citation
 
If you use this code or data in your work, please cite the associated preprint:
 
> *Circadian Modulation of Excitation–Inhibition Balance Shapes Seizure Dynamics in Epileptic Networks* — bioRxiv (2026).
> [https://www.biorxiv.org/content/10.64898/2026.05.22.726242v1.article-info](https://www.biorxiv.org/content/10.64898/2026.05.22.726242v1.article-info)

## Contact
For questions, issues, or interest in using or extending this code, feel free to open a GitHub issue or contact the corresponding author directly at [ilariac@kth.se] or [ilaria.carannante@gmail.com].
