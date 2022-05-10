import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np

import csv

import dgl

from scipy import sparse as sp
import numpy as np
import torch.nn.functional as F

# *NOTE
# The dataset pickle and index files are in ./zinc_molecules/ dir
# [<split>.pickle and <split>.index; for split 'train', 'val' and 'test']




class MoleculeDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, num_graphs=None):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs
        
        with open(data_dir + "/%s.pickle" % self.split,"rb") as f:
            self.data = pickle.load(f)

        if self.num_graphs in [10000, 1000]:
            # loading the sampled indices from file ./zinc_molecules/<split>.index
            with open(data_dir + "/%s.index" % self.split,"r") as f:
                data_idx = [list(map(int, idx)) for idx in csv.reader(f)]
                self.data = [ self.data[i] for i in data_idx[0] ]

            assert len(self.data)==num_graphs, "Sample num_graphs again; available idx: train/val/test => 10k/1k/1k"
        
        """
        data is a list of Molecule dict objects with following attributes
        
          molecule = data[idx]
        ; molecule['num_atom'] : nb of atoms, an integer (N)
        ; molecule['atom_type'] : tensor of size N, each element is an atom type, an integer between 0 and num_atom_type
        ; molecule['bond_type'] : tensor of size N x N, each element is a bond type, an integer between 0 and num_bond_type
        ; molecule['logP_SA_cycle_normalized'] : the chemical property to regress, a float variable
        """
        
        self.graph_lists = []
        self.graph_labels = []
        self.n_samples = len(self.data)
        self._prepare()
    
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        
        for molecule in self.data:
            node_features = molecule['atom_type'].long()
            
            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
            
            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
            
            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features
            
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features
            
            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])
        
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
    
    
class MoleculeAqSolDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, num_graphs=None):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs
        
        with open(data_dir + "/%s.pickle" % self.split,"rb") as f:
            self.data = pickle.load(f)
        
        """
        data is a list of tuple objects with following elements
        graph_object = (node_feat, edge_feat, edge_index, solubility)  
        """
        self.graph_lists = []
        self.graph_labels = []
        self.n_samples = len(self.data)
        assert num_graphs == self.n_samples
        
        self._prepare()
    
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        
        count_filter1, count_filter2 = 0,0
        
        for molecule in self.data:
            node_features = torch.LongTensor(molecule[0])
            edge_features = torch.LongTensor(molecule[1])
            
            # Create the DGL Graph
            g = dgl.graph((molecule[2][0], molecule[2][1]))
                        
            if g.num_nodes() == 0:
                count_filter1 += 1
                continue # skipping graphs with no bonds/edges
            
            if g.num_nodes() != len(node_features):
                count_filter2 += 1
                continue # cleaning <10 graphs with this discrepancy
            
            
            g.edata['feat'] = edge_features    
            g.ndata['feat'] = node_features
           
            self.graph_lists.append(g)
            self.graph_labels.append(torch.Tensor([molecule[3]]))
        print("Filtered graphs type 1/2: ", count_filter1, count_filter2)
        print("Filtered graphs: ", self.n_samples - len(self.graph_lists))
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graph_lists)

    def __getitem__(self, idx):
        return self.graph_lists[idx], self.graph_labels[idx]
    
    
class MoleculeDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name='Zinc'):
        t0 = time.time()
        self.name = name
        
        if self.name == 'AqSol':
            self.num_atom_type = 65 # known meta-info about the AqSol dataset; can be calculated as well 
            self.num_bond_type = 5 # known meta-info about the AqSol dataset; can be calculated as well
        else:            
            self.num_atom_type = 28 # known meta-info about the zinc dataset; can be calculated as well
            self.num_bond_type = 4 # known meta-info about the zinc dataset; can be calculated as well
        
        data_dir='./data/molecules'
        
        if self.name == 'ZINC-full':
            data_dir='./data/molecules/zinc_full'
            self.train = MoleculeDGL(data_dir, 'train', num_graphs=220011)
            self.val = MoleculeDGL(data_dir, 'val', num_graphs=24445)
            self.test = MoleculeDGL(data_dir, 'test', num_graphs=5000)
        elif self.name == 'ZINC':            
            self.train = MoleculeDGL(data_dir, 'train', num_graphs=10000)
            self.val = MoleculeDGL(data_dir, 'val', num_graphs=1000)
            self.test = MoleculeDGL(data_dir, 'test', num_graphs=1000)
        elif self.name == 'AqSol': 
            data_dir='./data/molecules/asqol_graph_raw'
            self.train = MoleculeAqSolDGL(data_dir, 'train', num_graphs=7985)
            self.val = MoleculeAqSolDGL(data_dir, 'val', num_graphs=998)
            self.test = MoleculeAqSolDGL(data_dir, 'test', num_graphs=999)
        print("Time taken: {:.4f}s".format(time.time()-t0))
        


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in MoleculeDataset class.
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
    
    n = g.number_of_nodes()
    if n <= pos_enc_dim:
        g.ndata['pos_enc'] = F.pad(g.ndata['pos_enc'], (0, pos_enc_dim - n + 1), value=float('0'))
    
    return g



class MoleculeDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading Moleccular datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = 'data/molecules/'
        with open(data_dir+name+'.pkl',"rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]
            self.num_atom_type = f[3]
            self.num_bond_type = f[4]
        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        #tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        #tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        #snorm_n = torch.cat(tab_snorm_n).sqrt()  
        #tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        #tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        #snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)       
        
        return batched_graph, labels
    
    # prepare dense tensors for GNNs using them; such as RingGNN, 3WLGNN
    def collate_dense_gnn(self, samples, edge_feat):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        #tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        #tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        #snorm_n = tab_snorm_n[0][0].sqrt()  
        
        #batched_graph = dgl.batch(graphs)
    
        g = graphs[0]
        adj = self._sym_normalize_adj(g.adjacency_matrix().to_dense())        
        """
            Adapted from https://github.com/leichen2018/Ring-GNN/
            Assigning node and edge feats::
            we have the adjacency matrix in R^{n x n}, the node features in R^{d_n} and edge features R^{d_e}.
            Then we build a zero-initialized tensor, say T, in R^{(1 + d_n + d_e) x n x n}. T[0, :, :] is the adjacency matrix.
            The diagonal T[1:1+d_n, i, i], i = 0 to n-1, store the node feature of node i. 
            The off diagonal T[1+d_n:, i, j] store edge features of edge(i, j).
        """

        zero_adj = torch.zeros_like(adj)

        if edge_feat:
            # use edge feats also to prepare adj
            adj_with_edge_feat = torch.stack([zero_adj for j in range(self.num_atom_type + self.num_bond_type)])
            adj_with_edge_feat = torch.cat([adj.unsqueeze(0), adj_with_edge_feat], dim=0)

            us, vs = g.edges()      
            for idx, edge_label in enumerate(g.edata['feat']):
                adj_with_edge_feat[edge_label.item()+1+self.num_atom_type][us[idx]][vs[idx]] = 1

            for node, node_label in enumerate(g.ndata['feat']):
                adj_with_edge_feat[node_label.item()+1][node][node] = 1
            
            x_with_edge_feat = adj_with_edge_feat.unsqueeze(0)
            
            return None, x_with_edge_feat, labels
        
        else:
            # use only node feats to prepare adj
            adj_no_edge_feat = torch.stack([zero_adj for j in range(self.num_atom_type)])
            adj_no_edge_feat = torch.cat([adj.unsqueeze(0), adj_no_edge_feat], dim=0)

            for node, node_label in enumerate(g.ndata['feat']):
                adj_no_edge_feat[node_label.item()+1][node][node] = 1

            x_no_edge_feat = adj_no_edge_feat.unsqueeze(0)

            return x_no_edge_feat, None, labels
    
    def _sym_normalize_adj(self, adj):
        deg = torch.sum(adj, dim = 0)#.squeeze()
        deg_inv = torch.where(deg>0, 1./torch.sqrt(deg), torch.zeros(deg.size()))
        deg_inv = torch.diag(deg_inv)
        return torch.mm(deg_inv, torch.mm(adj, deg_inv))
    
    def _add_self_loops(self):
        
        # function for adding self loops
        # this function will be called only if self_loop flag is True
            
        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]

    def _add_positional_encodings(self, pos_enc_dim):
        
        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists]
        self.test.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists]




