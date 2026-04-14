import json
import numpy as np
import os

import matplotlib.pyplot as plt
import brian2 as b2

implemented_neuron_models = ['FS_cr', 'RS_cr', 'IMP_cr']

#--------------------------------------------------
color = {
    'FS':  '#cb181d',
    'RS':  '#238b45',
    'IMP': '#2171b5'
}

#--------------------------------------------------
# Definying the AdEx model #

AdEx_eqs='''
dv/dt = (-GsynE*(v-Ee)-GsynI*(v-Ei)-gl*(v-El)+ gl*Dt*exp((v-Vt)/Dt)-w + Is)/Cm : volt (unless refractory)
dw/dt = (a*(v-El)-w)/tau_w:ampere
dGsynI/dt = -GsynI/Tsyn : siemens
dGsynE/dt = -GsynE/Tsyn : siemens
Itot = (GsynI+GsynE)*v : ampere 
Is = current(t) : ampere
Cm:farad
gl:siemens
El:volt
a:siemens
tau_w:second
Dt:volt
Vt:volt
Ee:volt
Ei:volt
Tsyn:second
'''

AdEx_eqs_taus='''
dv/dt = (-GsynE*(v-Ee)-GsynI*(v-Ei)-gl*(v-El)+ gl*Dt*exp((v-Vt)/Dt)-w + Is)/Cm : volt (unless refractory)
dw/dt = (a*(v-El)-w)/tau_w:ampere
dGsynI/dt = -GsynI/Tsyn_i : siemens
dGsynE/dt = -GsynE/Tsyn_e : siemens
Itot = (GsynI+GsynE)*v : ampere 
Is = current(t) : ampere
Cm:farad
gl:siemens
El:volt
a:siemens
tau_w:second
Dt:volt
Vt:volt
Ee:volt
Ei:volt
Tsyn_i:second
Tsyn_e:second
'''

# Definying the extended AdEx model #
AdEx_3var_eqs ='''
dv/dt = (-GsynE*(v-Ee)-GsynI*(v-Ei)-gl*(v-(El+z))+ gl*Dt*exp((v-(Vt-bz*z))/Dt)-w-gp*z + Is)/Cm : volt (unless refractory)
dw/dt = (a*(v-(El+z))-w)/tau_w:ampere
dz/dt = (Z0-v-z)/eps:volt
dGsynI/dt = -GsynI/Tsyn : siemens
dGsynE/dt = -GsynE/Tsyn : siemens
Itot = (GsynI+GsynE)*v : ampere 
eps:second
Z0:volt
Is = current(t) : ampere
Cm:farad
gl:siemens
gp:siemens
El:volt
a:siemens
tau_w:second
Dt:volt
Vt:volt
bz:1
Ee:volt
Ei:volt
Tsyn:second
'''#% neuron_params

# Definying the extended AdEx model with different taus #
AdEx_3var_eqs_taus ='''
dv/dt = (-GsynE*(v-Ee)-GsynI*(v-Ei)-gl*(v-(El+z))+ gl*Dt*exp((v-(Vt-bz*z))/Dt)-w-gp*z + Is)/Cm : volt (unless refractory)
dw/dt = (a*(v-(El+z))-w)/tau_w:ampere
dz/dt = (Z0-v-z)/eps:volt
dGsynI/dt = -GsynI/Tsyn_i : siemens
dGsynE/dt = -GsynE/Tsyn_e : siemens
Itot = (GsynI+GsynE)*v : ampere 
eps:second
Z0:volt
Is = current(t) : ampere
Cm:farad
gl:siemens
gp:siemens
El:volt
a:siemens
tau_w:second
Dt:volt
Vt:volt
bz:1
Ee:volt
Ei:volt
Tsyn_i:second
Tsyn_e:second
'''#% neuron_params

#--------------------------------------------------
def setting_simulation_Brian(idx = None, N_cell = None, neuron_model = None, json_file_name = None, curr_inj = None, sim_info = False):
    if N_cell == None:
        N_cell = 1

    if neuron_model == None:
        raise ValueError("Plese, specify the neuron_model you wish to simulate")
    if neuron_model not in implemented_neuron_models:
        raise ValueError(f"neuron_model must be one of {implemented_neuron_models}, but got '{neuron_model}'.")

    if json_file_name == None:
        raise ValueError("Plese, specify the json_file_name containing the model parameters")    
    
    with open(json_file_name, 'r') as file:
        data = json.load(file)
    
    if sim_info == True:
        print(f'Imported data: {json_file_name}')
    
    if neuron_model == 'FS_cr':
        if sim_info == True:
            print(f'neuron model: {neuron_model}')
        V_th_value = data[0][idx]['model']['V_peak_detect']
        V_reset_value = data[0][idx]['model']['V_reset']
        t_ref_value = data[0][idx]['model']['t_ref']
        b_value = data[0][idx]['model']['b']

        G_inh = b2.NeuronGroup(N_cell, AdEx_eqs, threshold = f'v > {V_th_value} * mV',
                                            reset = f'v = {V_reset_value} * mV; w += {b_value} * nA',
                                            refractory = f'{t_ref_value} * ms',
                                            method = 'heun',
                                            name = neuron_model)
        #init variables:
        G_inh.v = data[0][idx]['init']['v']*b2.mV
        G_inh.w = data[0][idx]['init']['w']*b2.nA
        G_inh.GsynI = data[0][idx]['init']['g_I']*b2.nS
        G_inh.GsynE = data[0][idx]['init']['g_E']*b2.nS

        #parameter values:
        G_inh.Cm = data[0][idx]['model']['C_m'] * b2.nF
        G_inh.gl = data[0][idx]['model']['g_L'] * b2.uS
        G_inh.El = data[0][idx]['model']['E_L'] * b2.mV
        G_inh.Vt = data[0][idx]['model']['V_th'] * b2.mV
        G_inh.Dt = data[0][idx]['model']['Delta_T'] * b2.mV
        G_inh.tau_w = data[0][idx]['model']['tau_w'] * b2.ms
        G_inh.a = data[0][idx]['model']['a'] * b2.nS
        if data[0][idx]['model']['I_e'] != 0:
            print(f"!!! Attention!!! I_e = {data[0][idx]['model']['I_e']} nA. Set the current accordingly.")
        
        G_inh.Ee = data[0][idx]['model']['E_e'] * b2.mV
        G_inh.Ei = data[0][idx]['model']['E_i'] * b2.mV
        G_inh.Tsyn = data[0][idx]['model']['tau_syn'] * b2.ms

        return G_inh
        
    if neuron_model == 'RS_cr':
        if sim_info == True:
            print(f'neuron model: {neuron_model}')
        V_th_value = data[0][idx]['model']['V_peak_detect']
        V_reset_value = data[0][idx]['model']['V_reset']
        t_ref_value = data[0][idx]['model']['t_ref']
        b_value = data[0][idx]['model']['b']
        
        G_exc = b2.NeuronGroup(N_cell, AdEx_eqs, threshold = f'v > {V_th_value} * mV',
                                            reset = f'v = {V_reset_value} * mV; w += {b_value} * nA',
                                            refractory = f'{t_ref_value} * ms',
                                            method = 'heun',
                                            name = neuron_model)
        #init variables:
        G_exc.v = data[0][idx]['init']['v']*b2.mV
        G_exc.w = data[0][idx]['init']['w']*b2.nA
        G_exc.GsynI = data[0][idx]['init']['g_I']*b2.nS
        G_exc.GsynE = data[0][idx]['init']['g_E']*b2.nS
        #parameter values:
        G_exc.Cm = data[0][idx]['model']['C_m'] * b2.nF
        G_exc.gl = data[0][idx]['model']['g_L'] * b2.uS
        G_exc.El = data[0][idx]['model']['E_L'] * b2.mV
        G_exc.Vt = data[0][idx]['model']['V_th'] * b2.mV
        G_exc.Dt = data[0][idx]['model']['Delta_T'] * b2.mV
        G_exc.tau_w = data[0][idx]['model']['tau_w'] * b2.ms
        G_exc.a = data[0][idx]['model']['a'] * b2.nS
        if data[0][idx]['model']['I_e'] != 0:
            print(f"!!! Attention!!! I_e = {data[0][idx]['model']['I_e']} nA. Set the current accordingly.")
        
        G_exc.Ee = data[0][idx]['model']['E_e'] * b2.mV
        G_exc.Ei = data[0][idx]['model']['E_i'] * b2.mV
        G_exc.Tsyn = data[0][idx]['model']['tau_syn'] * b2.ms
    
        return G_exc

    if str('IMP') in neuron_model:
        if sim_info == True:
            print(f'neuron model: {neuron_model}')
        V_th_value = data[0][idx]['model']['V_peak_detect']
        V_reset_value = data[0][idx]['model']['V_reset']
        t_ref_value = data[0][idx]['model']['t_ref']
        b_value = data[0][idx]['model']['b']
        
        G_imp = b2.NeuronGroup(N_cell, AdEx_3var_eqs, threshold = f'v > {V_th_value} * mV',
                                reset = f'v = {V_reset_value} * mV; w += {b_value} * nA',
                                refractory = f'{t_ref_value} * ms',
                                method = 'heun',
                                #name = neuron_model)
                                name = 'IMP')
        #init variables:
        G_imp.v = data[0][idx]['init']['v']*b2.mV
        G_imp.w = data[0][idx]['init']['w']*b2.nA
        G_imp.GsynI = data[0][idx]['init']['g_I']*b2.nS
        G_imp.GsynE = data[0][idx]['init']['g_E']*b2.nS
        #parameter values:
        G_imp.Cm = data[0][idx]['model']['C_m'] * b2.nF
        G_imp.gl = data[0][idx]['model']['g_L'] * b2.uS
        G_imp.El = data[0][idx]['model']['E_L'] * b2.mV
        G_imp.Vt = data[0][idx]['model']['V_th'] * b2.mV
        G_imp.Dt = data[0][idx]['model']['Delta_T'] * b2.mV
        G_imp.tau_w = data[0][idx]['model']['tau_w'] * b2.ms
        G_imp.a = data[0][idx]['model']['a'] * b2.nS
        G_imp.Z0 = data[0][idx]['model']['Z0'] * b2.mV
        G_imp.eps = data[0][idx]['model']['eps'] * b2.ms
        G_imp.gp = data[0][idx]['model']['gp'] * b2.uS
        G_imp.bz = data[0][idx]['model']['bz']

        if data[0][idx]['model']['I_e'] != 0:
            print(f"!!! Attention!!! I_e = {data[0][idx]['model']['I_e']} nA. Set the current accordingly.")
        
        G_imp.Ee = data[0][idx]['model']['E_e'] * b2.mV
        G_imp.Ei = data[0][idx]['model']['E_i'] * b2.mV
        G_imp.Tsyn = data[0][idx]['model']['tau_syn'] * b2.ms
    
        return G_imp

#--------------------------------------------------
def network_creation(conn_prob = None, 
                     pop_1 = None, pop_2 = None, pop_3 = None,
                     Qe = None, Qi = None,
                     seed = None):

    if seed is not None:
        b2.seed(seed)  # to control the connectivity

    S_11 = b2.Synapses(pop_1, pop_1, on_pre='GsynI_post+=Qi', name = 'S_11')
    S_11.connect('i!=j',p=conn_prob)
    
    S_12 = b2.Synapses(pop_1, pop_2, on_pre='GsynI_post+=Qi', name = 'S_12')
    S_12.connect(p=conn_prob)
    
    S_13 = b2.Synapses(pop_1, pop_3, on_pre='GsynI_post+=Qi', name = 'S_13') 
    S_13.connect(p=conn_prob)
    

    S_21 = b2.Synapses(pop_2, pop_1, on_pre='GsynE_post+=Qe', name = 'S_21') 
    S_21.connect(p=conn_prob)
    
    S_22 = b2.Synapses(pop_2, pop_2, on_pre='GsynE_post+=Qe', name = 'S_22') 
    S_22.connect('i!=j', p=conn_prob)
    
    S_23 = b2.Synapses(pop_2, pop_3, on_pre='GsynE_post+=Qe', name = 'S_23') 
    S_23.connect(p=conn_prob)
    
    
    S_31 = b2.Synapses(pop_3, pop_1, on_pre='GsynE_post+=Qe', name = 'S_31') 
    S_31.connect(p=conn_prob)
    
    S_32 = b2.Synapses(pop_3, pop_2, on_pre='GsynE_post+=Qe', name = 'S_32') 
    S_32.connect(p=conn_prob)
    
    S_33 = b2.Synapses(pop_3, pop_3, on_pre='GsynE_post+=Qe', name = 'S_33') 
    S_33.connect('i!=j', p=conn_prob)

    return S_11, S_12, S_13, S_21, S_22, S_23, S_31, S_32, S_33

#--------------------------------------------------
def extracting_pop_freq_and_std(sim_duration = None, 
                              p_start = None,
                              p_end = None,
                              pop1 = None, 
                              pop2 = None, 
                              pop3 = None,
                              N_pop1 = None,
                              N_pop2 = None,
                              N_pop3 = None,
                              bin_size = None):

    # Parameters
    if bin_size == None:
        bin_size = 0.1  # seconds
    bin_edges = np.arange(0, sim_duration / b2.second + bin_size, bin_size)
    time_bins = bin_edges[:-1]
    
    # Create spike matrix: (n_neurons, n_time_bins)
    spike_matrix_FS = np.zeros((N_pop1, len(time_bins)))
    spike_matrix_RS = np.zeros((N_pop2, len(time_bins)))
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
    
    for i, t in zip(pop3.i, pop3.t / b2.second):
        bin_idx = int(t // bin_size)
        if bin_idx < len(time_bins):
            spike_matrix_IMP[i, bin_idx] += 1
    
    # Convert to rate (Hz)
    spike_matrix_FS /= bin_size
    spike_matrix_RS /= bin_size
    spike_matrix_IMP /= bin_size
    
    
    # Compute mean and std
    mean_rate_FS = np.mean(spike_matrix_FS, axis=0)
    std_rate_FS = np.std(spike_matrix_FS, axis=0)
    mean_rate_RS = np.mean(spike_matrix_RS, axis=0)
    std_rate_RS = np.std(spike_matrix_RS, axis=0)
    mean_rate_IMP = np.mean(spike_matrix_IMP, axis=0)
    std_rate_IMP = np.std(spike_matrix_IMP, axis=0)

    # Defining the stimulation interval
    
    if p_end == sim_duration - p_start:
        left_bound = int((p_start / bin_size).item())
        right_bound = - left_bound + 1
        
    # Computing the average only over the stimulation
    mean_rate_FS_stim = np.mean(mean_rate_FS[left_bound:right_bound])
    mean_rate_RS_stim = np.mean(mean_rate_RS[left_bound:right_bound])
    mean_rate_IMP_stim = np.mean(mean_rate_IMP[left_bound:right_bound])

    mean_rates = [mean_rate_FS_stim, mean_rate_RS_stim, mean_rate_IMP_stim]

    # Computing the standard deviation only over the stimulation
    std_rate_FS_stim = np.std(std_rate_FS[left_bound:right_bound])
    std_rate_RS_stim = np.std(std_rate_RS[left_bound:right_bound])
    std_rate_IMP_stim = np.std(std_rate_IMP[left_bound:right_bound])

    std_rates = [std_rate_FS_stim, std_rate_RS_stim, std_rate_IMP_stim]

    if len(time_bins) != len(mean_rate_FS):
        import pdb
        pdb.set_trace()
    
    return mean_rates, std_rates

#--------------------------------------------------
def extracting_single_pop_freq_and_std(sim_duration = None, 
                              p_start = None,
                              p_end = None,
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

    # Defining the stimulation interval
    left_bound = int((p_start / bin_size).item())
    right_bound = left_bound + int(((p_end - p_start) / bin_size).item()) + 1
    
    # Computing the average only over the stimulation
    mean_rate_stim = np.mean(mean_rate[left_bound:right_bound])

    # Computing the standard deviation only over the stimulation
    std_rate_stim = np.std(std_rate[left_bound:right_bound])

    if len(time_bins) != len(mean_rate):
        import pdb
        pdb.set_trace()
    
    return mean_rate_stim, std_rate_stim

#--------------------------------------------------
def setting_simulation_Brian_taus(idx = None, N_cell = None, neuron_model = None, json_file_name = None, curr_inj = None, sim_info = False):
    if N_cell == None:
        N_cell = 1

    if neuron_model == None:
        raise ValueError("Plese, specify the neuron_model you wish to simulate")
    if neuron_model not in implemented_neuron_models:
        raise ValueError(f"neuron_model must be one of {implemented_neuron_models}, but got '{neuron_model}'.")

    if json_file_name == None:
        raise ValueError("Plese, specify the json_file_name containing the model parameters")    
    
    with open(json_file_name, 'r') as file:
        data = json.load(file)
    
    if sim_info == True:
        print(f'Imported data: {json_file_name}')
    
    if str('FS_cr') in neuron_model:
        if sim_info == True:
            print(f'neuron model: {neuron_model}')
        V_th_value = data[0][idx]['model']['V_peak_detect']
        V_reset_value = data[0][idx]['model']['V_reset']
        t_ref_value = data[0][idx]['model']['t_ref']
        b_value = data[0][idx]['model']['b']

        G_inh = b2.NeuronGroup(N_cell, AdEx_eqs_taus, threshold = f'v > {V_th_value} * mV',
                                            reset = f'v = {V_reset_value} * mV; w += {b_value} * nA',
                                            refractory = f'{t_ref_value} * ms',
                                            method = 'heun',
                                            name = neuron_model)
        #init variables:
        G_inh.v = data[0][idx]['init']['v']*b2.mV
        G_inh.w = data[0][idx]['init']['w']*b2.nA
        G_inh.GsynI = data[0][idx]['init']['g_I']*b2.nS
        G_inh.GsynE = data[0][idx]['init']['g_E']*b2.nS

        #parameter values:
        G_inh.Cm = data[0][idx]['model']['C_m'] * b2.nF
        G_inh.gl = data[0][idx]['model']['g_L'] * b2.uS
        G_inh.El = data[0][idx]['model']['E_L'] * b2.mV
        G_inh.Vt = data[0][idx]['model']['V_th'] * b2.mV
        G_inh.Dt = data[0][idx]['model']['Delta_T'] * b2.mV
        G_inh.tau_w = data[0][idx]['model']['tau_w'] * b2.ms
        G_inh.a = data[0][idx]['model']['a'] * b2.nS
        if data[0][idx]['model']['I_e'] != 0:
            print(f"!!! Attention!!! I_e = {data[0][idx]['model']['I_e']} nA. Set the current accordingly.")
        
        G_inh.Ee = data[0][idx]['model']['E_e'] * b2.mV
        G_inh.Ei = data[0][idx]['model']['E_i'] * b2.mV
        G_inh.Tsyn_e = data[0][idx]['model']['tau_syn_e'] * b2.ms
        G_inh.Tsyn_i = data[0][idx]['model']['tau_syn_i'] * b2.ms
        
        
        return G_inh

    if str('RS_cr') in neuron_model:
        if sim_info == True:
            print(f'neuron model: {neuron_model}')
        V_th_value = data[0][idx]['model']['V_peak_detect']
        V_reset_value = data[0][idx]['model']['V_reset']
        t_ref_value = data[0][idx]['model']['t_ref']
        b_value = data[0][idx]['model']['b']
        
        G_exc = b2.NeuronGroup(N_cell, AdEx_eqs_taus, threshold = f'v > {V_th_value} * mV',
                                            reset = f'v = {V_reset_value} * mV; w += {b_value} * nA',
                                            refractory = f'{t_ref_value} * ms',
                                            method = 'heun',
                                            name = neuron_model)
        #init variables:
        G_exc.v = data[0][idx]['init']['v']*b2.mV
        G_exc.w = data[0][idx]['init']['w']*b2.nA
        G_exc.GsynI = data[0][idx]['init']['g_I']*b2.nS
        G_exc.GsynE = data[0][idx]['init']['g_E']*b2.nS
        #parameter values:
        G_exc.Cm = data[0][idx]['model']['C_m'] * b2.nF
        G_exc.gl = data[0][idx]['model']['g_L'] * b2.uS
        G_exc.El = data[0][idx]['model']['E_L'] * b2.mV
        G_exc.Vt = data[0][idx]['model']['V_th'] * b2.mV
        G_exc.Dt = data[0][idx]['model']['Delta_T'] * b2.mV
        G_exc.tau_w = data[0][idx]['model']['tau_w'] * b2.ms
        G_exc.a = data[0][idx]['model']['a'] * b2.nS
        if data[0][idx]['model']['I_e'] != 0:
            print(f"!!! Attention!!! I_e = {data[0][idx]['model']['I_e']} nA. Set the current accordingly.")
        
        G_exc.Ee = data[0][idx]['model']['E_e'] * b2.mV
        G_exc.Ei = data[0][idx]['model']['E_i'] * b2.mV
        G_exc.Tsyn_e = data[0][idx]['model']['tau_syn_e'] * b2.ms
        G_exc.Tsyn_i = data[0][idx]['model']['tau_syn_i'] * b2.ms    

        
        return G_exc

    if str('IMP') in neuron_model:
        if sim_info == True:
            print(f'neuron model: {neuron_model}')
        V_th_value = data[0][idx]['model']['V_peak_detect']
        V_reset_value = data[0][idx]['model']['V_reset']
        t_ref_value = data[0][idx]['model']['t_ref']
        b_value = data[0][idx]['model']['b']
        
        G_imp = b2.NeuronGroup(N_cell, AdEx_3var_eqs_taus, threshold = f'v > {V_th_value} * mV',
                                reset = f'v = {V_reset_value} * mV; w += {b_value} * nA',
                                refractory = f'{t_ref_value} * ms',
                                method = 'heun',
                                #name = neuron_model)
                                name = 'IMP')
        #init variables:
        G_imp.v = data[0][idx]['init']['v']*b2.mV
        G_imp.w = data[0][idx]['init']['w']*b2.nA
        G_imp.GsynI = data[0][idx]['init']['g_I']*b2.nS
        G_imp.GsynE = data[0][idx]['init']['g_E']*b2.nS
        #parameter values:
        G_imp.Cm = data[0][idx]['model']['C_m'] * b2.nF
        G_imp.gl = data[0][idx]['model']['g_L'] * b2.uS
        G_imp.El = data[0][idx]['model']['E_L'] * b2.mV
        G_imp.Vt = data[0][idx]['model']['V_th'] * b2.mV
        G_imp.Dt = data[0][idx]['model']['Delta_T'] * b2.mV
        G_imp.tau_w = data[0][idx]['model']['tau_w'] * b2.ms
        G_imp.a = data[0][idx]['model']['a'] * b2.nS
        G_imp.Z0 = data[0][idx]['model']['Z0'] * b2.mV
        G_imp.eps = data[0][idx]['model']['eps'] * b2.ms
        G_imp.gp = data[0][idx]['model']['gp'] * b2.uS
        G_imp.bz = data[0][idx]['model']['bz']

        if data[0][idx]['model']['I_e'] != 0:
            print(f"!!! Attention!!! I_e = {data[0][idx]['model']['I_e']} nA. Set the current accordingly.")
        
        G_imp.Ee = data[0][idx]['model']['E_e'] * b2.mV
        G_imp.Ei = data[0][idx]['model']['E_i'] * b2.mV
        G_imp.Tsyn_e = data[0][idx]['model']['tau_syn_e'] * b2.ms
        G_imp.Tsyn_i = data[0][idx]['model']['tau_syn_i'] * b2.ms

    
        return G_imp

