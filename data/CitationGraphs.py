import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np

import csv

import dgl
from dgl.data import CoraDataset
from dgl.data import CitationGraphDataset
import networkx as nx

import random
random.seed(42)


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in CitationGraphsDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']
    
    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)
    
    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g



    
class CitationGraphsDataset(torch.utils.data.Dataset):
    def __init__(self, name):
        t0 = time.time()
        self.name = name.lower()
        
        if self.name == 'cora':
            dataset = CoraDataset()
        else:
            dataset = CitationGraphDataset(self.name)
        dataset.graph.remove_edges_from(nx.selfloop_edges(dataset.graph))
        graph = dgl.DGLGraph(dataset.graph)
        E = graph.number_of_edges()
        N = graph.number_of_nodes()
        D = dataset.features.shape[1]
        graph.ndata['feat'] = torch.Tensor(dataset.features)
        graph.edata['feat'] = torch.zeros((E, D))
        graph.batch_num_nodes = [N]


        self.norm_n = torch.FloatTensor(N,1).fill_(1./float(N)).sqrt()
        self.norm_e = torch.FloatTensor(E,1).fill_(1./float(E)).sqrt()
        self.graph = graph
        self.train_mask = torch.BoolTensor(dataset.train_mask)
        self.val_mask = torch.BoolTensor(dataset.val_mask)
        self.test_mask = torch.BoolTensor(dataset.test_mask)
        self.labels = torch.LongTensor(dataset.labels)
        self.num_classes = dataset.num_labels
        self.num_dims = D



        print("[!] Dataset: ", self.name)

        
        print("Time taken: {:.4f}s".format(time.time()-t0))
    
    
    def _add_self_loops(self):
        # function for adding self loops
        # this function will be called only if self_loop flag is True
        self.graph = self_loop(self.graph)
        norm = torch.pow(self.graph.in_degrees().float().clamp(min=1), -0.5)
        shp = norm.shape + (1,) * (self.graph.ndata['feat'].dim() - 1)
        self.norm_n = torch.reshape(norm, shp)

