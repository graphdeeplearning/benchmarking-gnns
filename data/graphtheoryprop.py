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
    https://github.com/lukecavabarrett/pna/
"""


class GraphTheoryPropDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, only_graph=False, only_nodes=False):
        self.data_dir = data_dir
        self.split = split
        self.only_graph = only_graph
        self.only_nodes = only_nodes
        
        # self.data = torch.load(os.path.join(self.data_dir, f'{self.k}cycles_n{self.n}_{self.n_samples}samples_{self.split}.pt'))
        
        with open(self.data_dir, 'rb') as f:
            (adj, features, node_labels, graph_labels) = pickle.load(f)
        
        max_node_labels = torch.cat([nls.max(0)[0].max(0)[0].unsqueeze(0) for nls in node_labels['train']]).max(0)[0]
        max_graph_labels = torch.cat([gls.max(0)[0].unsqueeze(0) for gls in graph_labels['train']]).max(0)[0]
        
        node_labels[self.split] = [nls / max_node_labels for nls in node_labels[self.split]]
        graph_labels[self.split] = [gls / max_graph_labels for gls in graph_labels[self.split]]
        
        self.adj = adj[self.split]
        self.features = features[self.split]
        self._node_labels = node_labels[self.split]
        self._graph_labels = graph_labels[self.split]
        
        self.graph_lists = []
        self.graph_labels = []
        self.node_labels = []
        self._prepare()
    
    def _prepare(self):
        print("preparing graphs for the %s set..." % (self.split.upper()))
        
        for i, adj_one_size in enumerate(self.adj):
            for j, adj in enumerate(adj_one_size):
                
                # Create the DGL Graph
                g = dgl.DGLGraph()
                g = dgl.from_scipy(sp.coo_matrix(adj))

                g.ndata['feat'] = self.features[i][j] #.long()
                
                # const 0 features for all edges; no edge features
                g.edata['feat'] = torch.zeros(g.number_of_edges()).long()

                self.graph_lists.append(g)
                self.graph_labels.append(self._graph_labels[i][j])
                self.node_labels.append(self._node_labels[i][j])
                
        del self.adj
        del self.features
        del self._graph_labels
        del self._node_labels
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graph_lists)

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
        return self.graph_lists[idx], self.graph_labels[idx], self.node_labels[idx]
    
    
class GraphTheoryPropDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name='GraphTheoryProp'):
        t0 = time.time()
        self.name = name
        
        data_dir = './data/graphtheoryprop/multitask_dataset.pkl'
        # data_dir = './multitask_dataset.pkl'
        
        self.train = GraphTheoryPropDGL(data_dir, 'train')
        self.val = GraphTheoryPropDGL(data_dir, 'val')
        self.test = GraphTheoryPropDGL(data_dir, 'test')
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
        
        
        

class GraphTheoryPropDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading GraphTheoryProp datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        n = 56
        k = 6
        data_dir = 'data/graphtheoryprop/'
    
        with open(data_dir + name + '.pkl',"rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]
        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
            
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, graph_labels, node_labels = map(list, zip(*samples))
        graph_labels = torch.stack(graph_labels)
        node_labels = torch.cat(node_labels)
    
        batched_graph = dgl.batch(graphs)
        return batched_graph, node_labels, graph_labels      

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
    