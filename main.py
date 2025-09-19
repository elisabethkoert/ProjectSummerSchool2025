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


def main():

    print("Main function started")
    # 1. Define single neuron model and network parameters for the model 
    # single neuron model 
    gl = 10.0*nsiemens   # Leak conductance
    el = -70*mV          # Resting potential
    er = -80*mV          # Inhibitory reversal potential
    vt = -50.*mV         # Spiking threshold
    memc = 200.0*pfarad  # Membrane capacitance
    bgcurrent = 50*pA   # External current

    eqs_neurons='''
    dv/dt=(-gl*(v-el)-(g_ampa*v+g_gaba*(v-er))+bgcurrent)/memc : volt (unless refractory)
    dg_ampa/dt = -g_ampa/tau_ampa : siemens # post-synaptic exc. conductance
    dg_gaba/dt = -g_gaba/tau_gaba : siemens # post-synaptic inh. conductance
    '''

    #network parameters 
    NE = 1000          # Number of excitatory cells
    NI = 250          # Number of inhibitory cells
    N=NE+NI           # Total number of neurons
    
    num_of_blocks=10
    
    # synaptic parameters 
    tau_ampa = 1.0*ms   # Glutamatergic synaptic time constant
    tau_gaba = 2.0*ms  # GABAergic synaptic time constant
    epsilon = 0.1      # Sparseness of synaptic connections (For the I->I, I->E, E->I connections)

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
    background_rate = 6000*Hz
    G_background = PoissonGroup(NE+NI, background_rate)
    noise_strength=0.5
    S_background = Synapses(G_background, neurons, on_pre='g_ampa += noise_strength*nS') # here we need to change the noise level. 
    # S_background.connect(True, p=0.1)
    S_background.connect(j='i')

    print('Network and neuron models are created and connected to background noise')
    # 2.1 Create the connectivity matrix 

    #connected_pairs=np.random.randint(0,n,size=(4,2))
    connected_pairs=[[3,4],[4,5],[5,6],[4,3],[5,4],[6,5]]
    M = generate_connectivity_matrix(N,num_of_blocks,connected_pairs)
    
    # show the matrix
    fig, ax = plt.subplots()
    ax.imshow(M, cmap='binary')
    # Save the plot to a file
    # plt.savefig('image.png')
    plt.show()
    
    print(M)

    # 2.2 Convert the connectivity matrix to a format for brian2  
    pre_idx, post_idx=M.nonzero()

    # 3.1 Build Brian 2 model with the connectivity matrix (2) and model parameters (1)
    con_e2e = Synapses(Pe, Pe, on_pre='g_ampa += 0.2*nS')
    con_e2e.connect(i=pre_idx,j=post_idx) 

    con_e2i = Synapses(Pe, Pi, on_pre='g_ampa += 0.5*nS')
    con_e2i.connect(p=epsilon)

    con_i2e = Synapses(Pi, Pe, on_pre='g_gaba += 4*nS')
    con_i2e.connect(p=epsilon)

    con_i2i = Synapses(Pi, Pi, on_pre='g_gaba += 6*nS')
    con_i2i.connect(p=epsilon)



    # 3.2 Set up the monitors to read out spike rate of all 
    #    neurons and E/I currents at some random neurons
    sm = SpikeMonitor(neurons)

    state_mon = StateMonitor(Pe, ['g_ampa', 'g_gaba'], record=np.arange(10)) 
    
    
    # 3.3 Run the model with white noise input at 1-10 different noise intensity levels
    
    
    # 4. Make a spike rate map, spike rate correlation map, spike rate covariance map
    #    -----> Export covariance map to make a dynamics graph
    
    
    # 5. Plot the E/I distributions
    


if __name__ == "__main__":
    main()