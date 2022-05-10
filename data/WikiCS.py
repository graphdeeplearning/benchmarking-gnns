import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np

import dgl
import json
import networkx as nx
from scipy import sparse as sp

import itertools

class WikiCSDataset(torch.utils.data.Dataset):
    """
        Wiki-CS Dataset
        Adapted from 
        https://github.com/pmernyei/wiki-cs-dataset/
    """
    def __init__(self, DATASET_NAME='WikiCS', path="data/WikiCS/"):
        self.name = DATASET_NAME
        self.data = json.load(open(os.path.join(path, 'data.json')))
        
        self.g, self.labels = None, None
        self.train_masks, self.stopping_masks, self.val_masks, self.test_mask = None, None, None, None
        self.num_classes, self.n_feats = None, None
        
        self._load()
        
    def _load(self):
        t0 = time.time()
        print("[I] Loading WikiCS ...")
        features = torch.FloatTensor(np.array(self.data['features']))
        self.labels = torch.LongTensor(np.array(self.data['labels']))
        
        self.train_masks = [torch.BoolTensor(tr) for tr in self.data['train_masks']]
        self.val_masks = [torch.BoolTensor(val) for val in self.data['val_masks']]
        self.stopping_masks = [torch.BoolTensor(st) for st in self.data['stopping_masks']]
        self.test_mask = torch.BoolTensor(self.data['test_mask'])
        
        self.n_feats = features.shape[1]
        self.num_classes = len(set(self.data['labels']))

        self.g = dgl.DGLGraph()
        self.g.add_nodes(len(self.data['features']))
        edge_list = list(itertools.chain(*[[(i, nb) for nb in nbs] for i,nbs in enumerate(self.data['links'])]))

        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        self.g.add_edges(src, dst)
        # edges are directional in DGL; make them bi-directional
        self.g.add_edges(dst, src)
        
        self.g.ndata['feat'] = features # available features
        self.g.edata['feat'] = torch.zeros(self.g.number_of_edges(), 1)
        
        print("[I] Finished loading after {:.4f}s".format(time.time()-t0))
        
    def _add_positional_encodings(self, pos_enc_dim):
        
        # Graph positional encoding v/ Laplacian eigenvectors
        g = self.g
        
        # Laplacian
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(g.number_of_nodes()) - N * A * N

        # Eigenvectors with numpy
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
        g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
        
        self.g = g