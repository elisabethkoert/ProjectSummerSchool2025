# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 17:28:23 2025

@author: Elisabeth
"""

import numpy as np
import matplotlib.pyplot as plt

from brian2 import *
from brian2.units import *

import scipy
from scipy.signal import periodogram

def createConnectivityMatrix():
    
    return M

def main():

    
    print("Main function started")
    # 1. Define parameters for the model (Neuron equations) mostly copy form existing model
    N=1000 # num of neurons in the model
    
    
    # 2. Create the connectivity matrix 
    #     ->> Export to make a structural Graph of the model
    
    
        
    # 3.1 Build Brian 2 model with the connectivity matrix (2) and model parameters (1)
    
    # 3.2 Set up the monitors to read out spike rate of all 
    #    neurons and E/I currents at some random neurons
    
    
    
    # 3.3 Run the model with white noise input at 1-10 different noise intensity levels
    
    
    # 4. Make a spike rate map, spike rate correlation map, spike rate covariance map
    #    -----> Export covariance map to make a dynamics graph
    
    
    # 5. Plot the E/I distributions
    


if __name__ == "__main__":
    main()