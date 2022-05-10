import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np
import networkx as nx

import csv

import dgl

from scipy import sparse as sp
import numpy as np

"""
    Part of this file is adapted from
    https://github.com/cvignac/SMP
"""


class CyclesDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, n, k, n_samples):
        self.data_dir = data_dir
        self.split = split
        self.n, self. k = n, k  # n is the number of the nodes, and k is the cycle len
        self.n_samples = n_samples
        
        self.data = torch.load(os.path.join(self.data_dir, f'{self.k}cycles_n{self.n}_{self.n_samples}samples_{self.split}.pt'))

        self.graph_lists = []
        self.graph_labels = []
        self._prepare()
    
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.n_samples, self.split.upper()))
        
        for sample in self.data:
            nx_graph, __, label = sample
            edge_list = nx.to_edgelist(nx_graph)

            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(nx_graph.number_of_nodes())
            
            # const 1 features for all nodes and edges; no node features
            g.ndata['feat'] = torch.ones(nx_graph.number_of_nodes(), 1, dtype=torch.float)
            
            for src, dst, _ in edge_list:
                g.add_edges(src, dst)
                g.add_edges(dst, src)
            g.edata['feat'] = torch.ones(2*len(edge_list), 1, dtype=torch.float)

            y = torch.tensor([1], dtype=torch.long) if label == 'has-kcycle' else torch.tensor([0], dtype=torch.long)

            self.graph_lists.append(g)
            self.graph_labels.append(y)
        del self.data
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]
    
    
class CyclesDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name='Cycles', n=56, k=6):
        t0 = time.time()
        self.name = name
        self.n = n
        self.k = k
        data_dir = './data/cycles'
        # data_dir = './cycle_detection'
        
        self.train = CyclesDGL(data_dir, 'train', n, k, n_samples=9000)
        self.val = CyclesDGL(data_dir, 'val', n, k, n_samples=1000)
        self.test = CyclesDGL(data_dir, 'test', n, k, n_samples=10000)
        print("Time taken: {:.4f}s".format(time.time()-t0))
        


        
def positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    # # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    # EigVec = EigVec[:, EigVal.argsort()] # increasing order
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float() 
    
    return g
        
        
        

class CyclesDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading Cycles datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        n = 56
        k = 6
        data_dir = 'data/cycles/'
        try:
            with open(data_dir+name+'_'+str(k)+'_'+str(n)+'.pkl',"rb") as f:
                f = pickle.load(f)
                self.train = f[0]
                self.val = f[1]
                self.test = f[2]
                self.n = n
                self.k = k
            print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
            print("[I] Finished loading.")
        except FileNotFoundError:
            print("[E] Data pkl files not found for k={} and n={}. Please prepare the pkl files for the corresponding k and n first.".format(k,n))
            
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels      

    def _add_positional_encodings(self, pos_enc_dim):
        
        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists]
        self.test.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists]



class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """
    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.graph_labels = lists[1]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])
    