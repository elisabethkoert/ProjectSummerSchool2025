# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 17:28:23 2025

@author: Elisabeth
"""

import brian2
import numpy as np
import matplotlib.pyplot as plt

from brian2 import *
from brian2.units import *

import scipy
from scipy.signal import periodogram

from generate_connectivity_matrix import generate_connectivity_matrix


def builAndRunModel(NE = 1000, NI = 250,p_intra=0.5,p_bg=0.1,p_inter=0.25, backgroundRate=9000,exitScaleFactor=1,inhibitionScaleFactor=1):
    """
    NE:  Number of excitatory cells
    NI:  Number of inhibitory cells
    backgroundRate: rate of background noise inputs
    """
    
    # 1. Define single neuron model and network parameters for the model 
    # single neuron model 
    gl = 10.0*nsiemens   # Leak conductance
    el = -70*mV          # Resting potential
    er = -80*mV          # Inhibitory reversal potential
    vt = -50.*mV         # Spiking threshold
    memc = 200.0*pfarad  # Membrane capacitance
    bgcurrent = 50*pA   # External current

    eqs_neurons='''
    dg_ampa/dt = -g_ampa/tau_ampa : siemens # post-synaptic exc. conductance
    dg_gaba/dt = -g_gaba/tau_gaba : siemens # post-synaptic inh. conductance 
    dv/dt=(-gl*(v-el)-(g_ampa*v+g_gaba*(v-er))+bgcurrent)/memc : volt (unless refractory)
    E_syn = -g_ampa*v : ampere
    I_syn = -g_gaba*(v-er): ampere
    '''
    
    #network parameters 
    N=NE+NI           # Total number of neurons
    
    num_of_blocks=10
    
    # synaptic parameters 
    tau_ampa = 1.0*ms   # Glutamatergic synaptic time constant
    tau_gaba = 2.0*ms  # GABAergic synaptic time constant
    #epsilon = 0.1      # Sparseness of synaptic connections (For the I->I, I->E, E->I connections)
    
    #time parameters 
    sim_dt = 0.1*ms
    simtime = 10*second # Simulation time
    defaultclock.dt = sim_dt
    
    
    # Create Neuron Populations
    neurons = NeuronGroup(NE+NI, model=eqs_neurons, threshold='v > vt',
                        reset='v=el', refractory=2*ms, method='euler')
    Pe = neurons[:NE]
    Pi = neurons[NE:]
    
    #add background noise to all neurons to acivate the network 
    background_rate = backgroundRate*Hz
    G_background = PoissonGroup(NE+NI, background_rate)
    noise_strength=0.5
    
    S_background = Synapses(G_background, neurons, on_pre='g_ampa += noise_strength*nS') # here we need to change the noise level. 
    # S_background.connect(True, p=0.1)
    S_background.connect(j='i')
    
    print('Network and neuron models are created and connected to background noise')
    # 2.1 Create the connectivity matrix 
    np.random.seed(42)  # For reproducibility
    connected_pairs=np.random.randint(0,num_of_blocks,size=(10,2))
    #connected_pairs=[[3,4],[4,5],[5,6],[4,3],[5,4],[6,5]]
    M = generate_connectivity_matrix(NE,num_of_blocks,connected_pairs,p_intra,p_bg,p_inter)
    epsilon=sum(M)/(NE*NE)
    # show the matrix
    fig, ax = plt.subplots()
    ax.imshow(M, cmap='binary')
    # Save the plot to a file
    # plt.savefig('image.png')
    #plt.show()
    
    # 2.2 Convert the connectivity matrix to a format for brian2  
    pre_idx, post_idx=M.nonzero()
    
    # 3.1 Build Brian 2 model with the connectivity matrix (2) and model parameters (1)
    value_e2e=0.2*exitScaleFactor
    con_e2e = Synapses(Pe, Pe, on_pre=f'g_ampa += {value_e2e}*nS')
    con_e2e.connect(i=pre_idx,j=post_idx) 
    
    value_e2i=0.5*exitScaleFactor
    con_e2i = Synapses(Pe, Pi, on_pre='g_ampa +={value_e2i}*nS')
    con_e2i.connect(p=epsilon)
    
    value_i2e=4*inhibitionScaleFactor
    con_i2e = Synapses(Pi, Pe, on_pre='g_gaba += {value_i2e}*nS')
    con_i2e.connect(p=epsilon)
    
    value_i2i=6*inhibitionScaleFactor
    con_i2i = Synapses(Pi, Pi, on_pre='g_gaba += {value_i2i}*nS')
    con_i2i.connect(p=epsilon)
    
    # 3.2 Set up the monitors to read out spike rate of all 
    #    neurons and E/I currents at some random neurons
    sm = SpikeMonitor(neurons)
    
    state_mon = StateMonitor(Pe, ['E_syn', 'I_syn'], record=True) 
    
    
    
    # 3.3 Run the model with white noise input at 1-10 different noise intensity levels
    brian2.run(1.*second)   
    
    
    return sm, state_mon


def makePlots(sm,state_mon):
    # 4. Make a spike rate map, spike rate correlation map, spike rate covariance map
    #    -----> Export covariance map to make a dynamics graph

    avg_firing_rate = len(sm.i)/(NE+NI)

    plt.figure(figsize=(15,6))
    plt.plot(sm.t, sm.i, 'k.',ms = 1)
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron ID")
    plt.title(f"avg firing rate: {avg_firing_rate:.2f} Hz")
    plt.show()

    # 1. Extract spike times and neuron indices
    spike_trains = sm.spike_trains()  # dictionary: neuron index -> array of spike times

    # 2. Bin the spike times (e.g., in 10 ms windows)
    bin_size = 10*ms
    bins = np.arange(0.1, 1*second/bin_size + 1) * bin_size
    n_neurons = len(spike_trains)
    binned_counts = np.zeros((n_neurons, len(bins)-1))

    for i, train in spike_trains.items():
        counts, _ = np.histogram(train, bins=bins)
        binned_counts[i] = counts
    spikingneurons = np.where(np.sum(binned_counts,axis=1)>1)[0]


    # 3. Compute covariance matrix
    # Rows: neurons, Columns: time bins
    cov_matrix = np.corrcoef(binned_counts[spikingneurons,10:])

    print("Correlation matrix shape:", cov_matrix.shape)
    # Option 2: Using matplotlib directly
    plt.figure(figsize=(15, 6))
    plt.imshow(cov_matrix, interpolation='nearest', cmap='magma')
    plt.title(f"Mean Correlation = {np.mean(cov_matrix):.2f}")
    plt.colorbar(label='Correlation')
    plt.xlabel("Neuron Index")
    plt.ylabel("Neuron Index")
    plt.show()


    # 5. Plot the E/I distributions
    E=np.array(state_mon.E_syn[:,1000:]/nS)
    I=np.array(state_mon.I_syn[:,1000:]/nS)
    Time=np.array(state_mon.t[1000:]/ms)

    E_mean=np.mean(E,axis=0)
    I_mean=np.mean(I,axis=0)

    #%matplotlib qt
    plt.figure(figsize=(15,6))
    bins=np.linspace(0,1.01*np.max([I_mean,E_mean]),100)
    plt.plot(Time,E_mean,color='blue',label='E')
    plt.plot(Time,I_mean,color='red',label='I')
    plt.title(f"E/I currents, mean E={np.mean(E_mean):.2f} nA, mean I={np.mean(I_mean):.2f} nA")
    #plt.plot(Time,E_mean-I_mean,color='black',label='difference')
    plt.xlabel('Time (ms)')
    plt.legend()
    
    
    

NE = 1000
NI = 250
p_intra=0.5
p_bg=0.1
p_inter=0.25
backgroundRate=9000
exitScaleFactor=1
inhibitionScaleFactor=1


sm,state_mon=builAndRunModel(NE,NI,p_intra,p_bg,p_inter,backgroundRate,exitScaleFactor,inhibitionScaleFactor)

makePlots(sm,state_mon)





# if __name__ == "__main__":
#     main()
