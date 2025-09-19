# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 19:01:50 2025

@author: Elisabeth
"""
import numpy as np
import matplotlib.pyplot as plt

def set_random_value(arr, p):
    # Get the shape of the array
    n = arr.shape[0]

    # Get the indices of the random value to set to 1
    i, j = np.random.choice(n, 2, replace=False)

    # Set the value in the array to 1
    arr[i, j] = 1

    return arr


def createConnectivityMatrix(N=10,num_subpopulations=10,size_subpopulation=50):
    # initaite full matrix with backgorund connectivity
    background_connectivity=0.05
    M=np.zeros((N,N))
    M=set_random_value(M, background_connectivity)

    
    # Go into clusters, set connectivity zero and refill with higher connectivity
    
    # define an array that will contain the neuron IDs for all subcluster neurons in the first 
    # collumn and the cluster ID in the second collumn
    cluster_neurons=np.zeros((num_subpopulations*size_subpopulation,2))
    cluster_neurons[:,0]=np.linspace(0,num_subpopulations*size_subpopulation-1)
    print(cluster_neurons[:,1])
    start_ix=0
    for cluster_ix=1:num_subpopulations:
        cluster_neurons(start_ix:start_ix+size_subpopulation,2)=cluster_ix
    
    
    
    # Connect clusters to output cluster
    
    # Plot matrix
    fig, ax = plt.subplots()
    ax.imshow(M, cmap='binary')
    # Save the plot to a file
    # plt.savefig('image.png')
    plt.show()
    
    return M, cluster_neurons

