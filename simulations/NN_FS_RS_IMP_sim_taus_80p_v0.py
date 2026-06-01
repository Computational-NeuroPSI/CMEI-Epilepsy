import sys
import os
import h5py

# Adding the project folder to sys.path
cwd = os.getcwd()
parent_dir = os.path.abspath(os.path.join(cwd, '..'))
sys.path.append(parent_dir)
# We do this so that we can directly import files in the utils folder

from utils.Brian_function_helper import *
from utils.Plots_helper import *
from utils.Network_helper import *
from utils.NN_simulation_analysis_helper import *

#sim_folder = '/cfs/klemming/projects/supr/snic2021-5-492/ilariac/circadian_rhythms/sim_10p_z0-80_tau_r_1'

sim_folder = 'testing_80p_v0'

# Connect external input
def connect_input(P, G, conn_prob):
    S = b2.Synapses(P, G, on_pre='GsynE_post+=Qe')
    S.connect(p=conn_prob)
    return S

# TimedArray helper --- 
def make_timed_array(times, dt,  rate, p_start = None, p_end = None):
    if p_start == None:
        p_start = 5 * b2.second
    if p_end == None:
        p_end = 15 * b2.second       
    
    arr = b2.zeros(len(times)) * b2.Hz
    arr[(times >= p_start) & (times < p_end)] = rate
    return b2.TimedArray(arr, dt=dt)

def run_single_sim(model_FS, model_RS, model_IMP):

    b2.start_scope()

    ### Defining duration of the simualtion
    sim_duration = 20 * b2.second
    dt = 0.1 * b2.ms  # time resolution
    times = b2.arange(0, sim_duration, dt)

    bin_size = 0.1 * b2.second
    bin_edges = np.arange(0, sim_duration + bin_size, bin_size)
    
    time_bins = time_bins = bin_edges[:-1] + (bin_size/b2.second / 2) 
    
    ### Setting up the neuron models
    data_folder = '../AdEx_models'
    idx = 0 # in this case only one model per cell type is provided
           
    ## Network config file
    network_config_folder = 'network_config_files/'
    network_file_name = 'network_config_file_sim_80p.json'
    
    network_config = get_network_config(json_file_name =  os.path.join(network_config_folder, network_file_name))
    
    
    ### Connecting the 3 populations
    conn_prob = network_config['network_composition']['conn_prob'] # connection probability
    # Quantal increment in synaptic conductances:
    Qe = 1.5 * b2.nS
    Qi = 5.0 * b2.nS
    
    N_external_exc = network_config['external_input']['N_external_exc'] # number of external neurons - number of incoming spikes
    rate_external_input_all = network_config['background_input']['freq'] * b2.Hz # firing rate of the external input
    
    print(f"Running condition: FS={model_FS}, RS={model_RS}, IMP={model_IMP}")


    #b2.start_scope()

    ###############################################################
    # 1. Build the three neuron populations
    ###############################################################
    neuron_model = model_FS
    N_cell_FS = network_config['network_composition']['FS_neuron']
    G_inh = setting_simulation_Brian_taus(
        idx=idx, N_cell=N_cell_FS,
        neuron_model=neuron_model,
        json_file_name=os.path.join(data_folder, model_FS + '.json'),
        curr_inj=None
    )

    neuron_model = model_RS
    N_cell_RS = network_config['network_composition']['RS_neuron']
    G_exc = setting_simulation_Brian_taus(
        idx=idx, N_cell=N_cell_RS,
        neuron_model=neuron_model,
        json_file_name=os.path.join(data_folder, model_RS + '.json'),
        curr_inj=None
    )

    neuron_model = model_IMP
    N_cell_IMP = network_config['network_composition']['IMP_neuron']
    G_imp = setting_simulation_Brian_taus(
        idx=idx, N_cell=N_cell_IMP,
        neuron_model=neuron_model,
        json_file_name=os.path.join(data_folder, model_IMP + '.json'),
        curr_inj=None
    )

    ###############################################################
    # 2. Connect populations
    ###############################################################
    S_11, S_12, S_13, S_21, S_22, S_23, S_31, S_32, S_33 = network_creation(
        conn_prob=conn_prob,
        pop_1=G_inh,
        pop_2=G_exc,
        pop_3=G_imp,
        Qe=Qe, Qi=Qi,
        seed=12345
    )

    ###############################################################
    # 3. Background inputs with condition-dependent rates
    ###############################################################
    rate_external_input_all = network_config['background_input']['freq'] * b2.Hz
    rate_external_input_inh = input_FS * b2.Hz
    rate_external_input_exc = input_RS * b2.Hz
    rate_external_input_imp = input_IMP * b2.Hz

    rate_timed_array_all = make_timed_array(times, dt, rate_external_input_all)
    rate_timed_array_inh = make_timed_array(times, dt, rate_external_input_inh)
    rate_timed_array_exc = make_timed_array(times, dt, rate_external_input_exc)
    rate_timed_array_imp = make_timed_array(times, dt, rate_external_input_imp)

    # Poisson Groups
    P_ed_all = b2.PoissonGroup(N_external_exc, rates='rate_timed_array_all(t)')
    P_ed_inh = b2.PoissonGroup(N_external_exc, rates='rate_timed_array_inh(t)')
    P_ed_exc = b2.PoissonGroup(N_external_exc, rates='rate_timed_array_exc(t)')
    P_ed_imp = b2.PoissonGroup(N_external_exc, rates='rate_timed_array_imp(t)')

    S_ed_background_inh = connect_input(P_ed_all, G_inh, conn_prob)
    S_ed_background_exc = connect_input(P_ed_all, G_exc, conn_prob)
    S_ed_background_imp = connect_input(P_ed_all, G_imp, conn_prob)
    S_ed_inh = connect_input(P_ed_inh, G_inh, conn_prob)
    S_ed_exc = connect_input(P_ed_exc, G_exc, conn_prob)
    S_ed_imp = connect_input(P_ed_imp, G_imp, conn_prob)

    ###############################################################
    # 4. Monitors
    ###############################################################
    mon_spike_FS  = b2.SpikeMonitor(G_inh)
    mon_spike_RS  = b2.SpikeMonitor(G_exc)
    mon_spike_IMP = b2.SpikeMonitor(G_imp)

    ###############################################################
    # 5. Run
    ###############################################################
    b2.run(
        sim_duration,
        namespace={
            'rate_timed_array_all': rate_timed_array_all,
            'rate_timed_array_inh': rate_timed_array_inh,
            'rate_timed_array_exc': rate_timed_array_exc,
            'rate_timed_array_imp': rate_timed_array_imp,
            'Qe': Qe,
            'Qi': Qi
        }
    )

    ###############################################################
    # 6. Save results
    ###############################################################

    os.makedirs(sim_folder, exist_ok=True)
   
    fname = (
        f"{sim_folder}/spikes_{model_FS}_{model_RS}_{model_IMP}.h5"
    )    

    with h5py.File(fname, "w") as f:

        # metadata
        meta = f.create_group("metadata")
        meta.create_dataset("sim_duration", data=sim_duration / b2.second)     
        meta.create_dataset("p_start", data = 5)  #seconds
        meta.create_dataset("p_end", data = 15)   #seconds  
        meta.create_dataset("num_FS_neurons", data=N_cell_FS)
        meta.create_dataset("num_RS_neurons", data=N_cell_RS)
        meta.create_dataset("num_IMP_neurons", data=N_cell_IMP)
        meta.create_dataset("Z0_value", data=G_imp.Z0[0] / b2.mV)
        meta.create_dataset("input_FS", data=input_FS)
        meta.create_dataset("input_RS", data=input_RS)
        meta.create_dataset("input_IMP", data=input_IMP)

        # spikes
        spikes = f.create_group("spikes")

        gFS = spikes.create_group("FS")
        gFS.create_dataset("i", data=mon_spike_FS.i, compression="gzip")
        gFS.create_dataset("t", data=mon_spike_FS.t, compression="gzip")

        gRS = spikes.create_group("RS")
        gRS.create_dataset("i", data=mon_spike_RS.i, compression="gzip")
        gRS.create_dataset("t", data=mon_spike_RS.t, compression="gzip")

        gIMP = spikes.create_group("IMP")
        gIMP.create_dataset("i", data=mon_spike_IMP.i, compression="gzip")
        gIMP.create_dataset("t", data=mon_spike_IMP.t, compression="gzip")

    print(f"Saved: {fname}")

    ###############################################################
    # 7. Saving also a plot
    ###############################################################

    os.makedirs(sim_folder + '/figures', exist_ok=True)

    fig = network_raster_plot(pop1 = mon_spike_FS, 
                              pop2 = mon_spike_RS, 
                              pop3 = mon_spike_IMP,
                              N_pop1 = N_cell_FS,
                              N_pop2 = N_cell_RS,
                              N_pop3 = N_cell_IMP,
                              x_lim = [0, sim_duration / b2.second])
  
    plt.savefig(f"{sim_folder}/figures/raster_{model_FS}_{model_RS}_{model_IMP}.png")    
    
    fig = plotting_pop_freq_and_std(sim_duration = sim_duration, 
                                  pop1 = mon_spike_FS, 
                                  pop2 = mon_spike_RS, 
                                  pop3 = mon_spike_IMP,
                                  N_pop1 = N_cell_FS,
                                  N_pop2 = N_cell_RS,
                                  N_pop3 = N_cell_IMP,
                                  ylim = (-10,275))

    plt.savefig(f"{sim_folder}/figures/sd_{model_FS}_{model_RS}_{model_IMP}.png")

    fig = plotting_pop_freq_and_std(sim_duration = sim_duration, 
                                  pop1 = mon_spike_FS, 
                                  pop2 = mon_spike_RS, 
                                  pop3 = mon_spike_IMP,
                                  N_pop1 = N_cell_FS,
                                  N_pop2 = N_cell_RS,
                                  N_pop3 = N_cell_IMP,
                                  bin_size = 500 * b2.ms
                                  )

    plt.savefig(f"{sim_folder}/figures/sd_{model_FS}_{model_RS}_{model_IMP}_bin500msec.png")    

    ###############################################################
    # 8. Adding Synchronicity analysis
    ###############################################################

    mean_rate_FS  = compute_population_rate(mon_spike_FS.t, bin_edges, N_cell_FS)
    mean_rate_RS  = compute_population_rate(mon_spike_RS.t, bin_edges, N_cell_RS)
    mean_rate_IMP = compute_population_rate(mon_spike_IMP.t, bin_edges, N_cell_IMP) 


    ictal_thresholds = {
        'FS': 100,
        'RS': 100,   # it will be never reached in our simulations, but good to keep
        'IMP': 125   # different threshold
    }
    
    interictal_thresholds = {
        'FS': 10,
        'RS': 10,
        'IMP': 10
    }
    
    results = extract_all_regimes(
        time_bins,
        mean_rate_FS,
        mean_rate_RS,
        mean_rate_IMP,
        ictal_thresholds,
        interictal_thresholds,
        min_duration=0.5  # 500 ms to avoid noise
    )
    
    ictal_FS = results['ictal_FS']
    ictal_RS = results['ictal_RS']
    ictal_IMP = results['ictal_IMP']
    
    interictal_FS = results['interictal_FS']
    interictal_RS = results['interictal_RS']
    interictal_IMP = results['interictal_IMP']

    fig = plot_regime_timeline(results, split_time = 15)
    plt.savefig(f"{sim_folder}/figures/regime_timeline_{model_IMP}.png")
    pops = ['FS','RS','IMP']
    fig = plot_ictal_regime_duration(results, pops, split_time = 15)
    plt.savefig(f"{sim_folder}/figures/ictal_regime_{model_IMP}.png")
    fig = plot_interictal_regime_duration(results, pops, split_time = 15)
    plt.savefig(f"{sim_folder}/figures/interictal_regime_{model_IMP}.png")
    fig = plot_regime_summary(time_bins,
                            pops,
                            mean_rate_FS,
                            mean_rate_RS,
                            mean_rate_IMP,
                            results,
                            split_time=15)

    plt.savefig(f"{sim_folder}/figures/regime_panel_{model_IMP}.png")
    sttc_window = 0.2    # 200ms sliding window
    sttc_dt = 0.002      # 2.5ms correlation window
    
    
    # Number of random neuron pairs to average per population
    # Here we are taking the 10% of each population
    n_sampled_pairs_FS = int(N_cell_FS * 0.1) 
    n_sampled_pairs_RS = int(N_cell_RS * 0.1) 
    n_sampled_pairs_IMP = int(N_cell_IMP * 0.1) 
    
    
    # 2. Compute STTC for each population
    print("Computing STTC for FS...")
    sync_FS = compute_sliding_sttc_dardel(mon_spike_FS.t, mon_spike_FS.i, time_bins, sttc_window, 
                                   n_pairs=n_sampled_pairs_FS, dt_corr=sttc_dt)
    
    print("Computing STTC for RS...")
    sync_RS = compute_sliding_sttc_dardel(mon_spike_RS.t, mon_spike_RS.i, time_bins, sttc_window, 
                                   n_pairs=n_sampled_pairs_RS, dt_corr=sttc_dt)
    
    print("Computing STTC for IMP...")
    sync_IMP = compute_sliding_sttc_dardel(mon_spike_IMP.t, mon_spike_IMP.i, time_bins, sttc_window, 
                                    n_pairs=n_sampled_pairs_IMP, dt_corr=sttc_dt)

    ### saving ictal, interictal and sttc index
    fname = (
        f"{sim_folder}/ictal_interictal_synchro_{model_IMP}.h5"
    )    

    with h5py.File(fname, "w") as f:

        # metadata
        meta = f.create_group("metadata")
        meta.create_dataset("sim_duration", data=sim_duration / b2.second)     
        meta.create_dataset("p_start", data = 5)  #seconds
        meta.create_dataset("p_end", data = 15)   #seconds  
        meta.create_dataset("num_FS_neurons", data=N_cell_FS)
        meta.create_dataset("num_RS_neurons", data=N_cell_RS)
        meta.create_dataset("num_IMP_neurons", data=N_cell_IMP)
        meta.create_dataset("Z0_value", data=G_imp.Z0[0] / b2.mV)
        meta.create_dataset("input_FS", data=input_FS)
        meta.create_dataset("input_RS", data=input_RS)
        meta.create_dataset("input_IMP", data=input_IMP)
        meta.create_dataset("tau", data = model_IMP[-8:] )

        # ictal
        ictal = f.create_group("ictal")
        ictal.create_dataset("ictal_FS", data=ictal_FS, compression="gzip")
        ictal.create_dataset("ictal_RS", data=ictal_RS, compression="gzip")
        ictal.create_dataset("ictal_IMP", data=ictal_IMP, compression="gzip")

        # interictal
        interictal = f.create_group("interictal")
        interictal.create_dataset("interictal_FS", data=interictal_FS, compression="gzip")
        interictal.create_dataset("interictal_RS", data=interictal_RS, compression="gzip")
        interictal.create_dataset("interictal_IMP", data=interictal_IMP, compression="gzip")

        # synchro
        synchro = f.create_group("synchro")
        synchro.create_dataset("synchro_FS", data=sync_FS, compression="gzip")
        synchro.create_dataset("synchro_RS", data=sync_RS, compression="gzip")
        synchro.create_dataset("synchro_IMP", data=sync_IMP, compression="gzip")
        
    
    print(f"Saved: {fname}")
    

    def convert_monitors_to_spike_dict(mon_FS, mon_RS, mon_IMP):
        return {
        'FS': {'t': mon_FS.t / b2.second, 'i': np.array(mon_FS.i)},
        'RS': {'t': mon_RS.t / b2.second, 'i': np.array(mon_RS.i)},
        'IMP': {'t': mon_IMP.t / b2.second, 'i': np.array(mon_IMP.i)},
        }

    spike_data = convert_monitors_to_spike_dict(
        mon_spike_FS, mon_spike_RS, mon_spike_IMP
    )

    fig = plot_synchrony_panels_extended_5(results, pops, sync_FS, sync_RS, sync_IMP,
                                       mean_rate_FS, mean_rate_RS, mean_rate_IMP,
                                      time_bins, spike_data , split_time=15)
    plt.savefig(f"{sim_folder}/figures/synch_{model_IMP}.png")
    print(f"Done: FS={model_FS}, RS={model_RS}, IMP={model_IMP}")


from multiprocessing import Pool
import itertools


input_FS  = 1 #Hz
input_RS  = 1 #Hz
input_IMP = 1 #Hz


params = [
         ('FS_cr_tau_r_1',  'RS_cr_tau_r_1',  'IMP_cr_Z0_-50_tau_r_1'),
         ('FS_cr_tau_r_08', 'RS_cr_tau_r_08', 'IMP_cr_Z0_-50_tau_r_08')
]

#if __name__ == "__main__":
#    run_single_sim("FS_cr_tau_r_1", "RS_cr_tau_r_1", "IMP_cr_Z0_-50_tau_r_1")


if __name__ == "__main__":
    with Pool(processes = 2) as pool:   # run on 64 workers
        pool.starmap(run_single_sim, params)






















