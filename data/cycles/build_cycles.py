"""
    The function defined in this file is called in prepare_cycles.ipynb notebook/
    No need to run this file otherwise!
"""

import os
import torch
import pickle
import numpy as np
import networkx as nx
import numpy.random as npr


rootdir = '.'

"""
    Code adapted from 
    https://github.com/cvignac/SMP
"""

def build_dataset():
    """ Given pickle files, split the dataset into one per value of n
    Run once before running the experiments. """
    n_samples = 10000
    # for k in [4, 6, 8]: # Originally in cvignac/SMP
    for k in [6]:         # We use only k=6 to prepare CYCLES dataset:
        with open(os.path.join(rootdir, 'datasets_kcycle_k={}_nsamples=10000.pickle'.format(k)), 'rb') as f:
            datasets_params, datasets = pickle.load(f)
            # Split by graph size
            for params, dataset in zip(datasets_params, datasets):
                n = params['n']
                # also saving 1000 samples from train set as the val set
                train, val, test = dataset[:n_samples-1000], dataset[n_samples-1000:n_samples], dataset[n_samples:]
                n_samples_train, n_samples_val, n_samples_test = len(train), len(val), len(test)
                
                torch.save(train, rootdir + f'/{k}cycles_n{n}_{n_samples_train}samples_train.pt')
                torch.save(val, rootdir + f'/{k}cycles_n{n}_{n_samples_val}samples_val.pt')
                torch.save(test, rootdir + f'/{k}cycles_n{n}_{n_samples_test}samples_test.pt')
                # torch.save(test, '{}cycles_n{}_{}samples_test.pt'.format(k, n, n_samples))

if __name__ == '__main__':
    build_dataset()