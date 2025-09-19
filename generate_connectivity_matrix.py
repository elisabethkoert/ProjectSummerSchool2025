# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import matplotlib.pyplot as plt

def generate_connectivity_matrix(Num, num_blocks, connected_blocks, p_intra=0.25, p_bg=0.05, p_inter=0.15):
    mat=np.random.choice([0,1],size=(Num,Num),p=[1-p_bg,p_bg]) ## generate background matrix with p_bg
    block_size=int(Num/num_blocks)
    #print("block size: ", block_size)
    block_indices=np.arange(0,Num+1,block_size)
    #print("block indexes: ", block_indices)
    for i1 in range(num_blocks): ### for intra block probabilities
        start_index=block_indices[i1]
        stop_index=block_indices[i1+1]
        mat[start_index:stop_index,start_index:stop_index]=np.random.choice([0,1],size=(block_size,block_size),p=[1-p_intra,p_intra])
    for j1,k1 in connected_blocks: ### for inter block probabilities
        mat[block_indices[j1]:block_indices[j1+1],block_indices[k1]:block_indices[k1+1]]=np.random.choice([0,1],size=(block_size,block_size),p=[1-p_inter,p_inter])
    #This function returns the connectivity matrix
    return mat


