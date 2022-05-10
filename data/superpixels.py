import os
import pickle
from scipy.spatial.distance import cdist
import numpy as np
import itertools

import dgl
import torch
import torch.utils.data

import time

import csv
from sklearn.model_selection import StratifiedShuffleSplit




def sigma(dists, kth=8):
    # Compute sigma and reshape
    try:
        # Get k-nearest neighbors for each node
        knns = np.partition(dists, kth, axis=-1)[:, kth::-1]
        sigma = knns.sum(axis=1).reshape((knns.shape[0], 1))/kth
    except ValueError:     # handling for graphs with num_nodes less than kth
        num_nodes = dists.shape[0]
        # this sigma value is irrelevant since not used for final compute_edge_list
        sigma = np.array([1]*num_nodes).reshape(num_nodes,1)
        
    return sigma + 1e-8 # adding epsilon to avoid zero value of sigma


def compute_adjacency_matrix_images(coord, feat, use_feat=True, kth=8):
    coord = coord.reshape(-1, 2)
    # Compute coordinate distance
    c_dist = cdist(coord, coord)
    
    if use_feat:
        # Compute feature distance
        f_dist = cdist(feat, feat)
        # Compute adjacency
        A = np.exp(- (c_dist/sigma(c_dist))**2 - (f_dist/sigma(f_dist))**2 )
    else:
        A = np.exp(- (c_dist/sigma(c_dist))**2)
        
    # Convert to symmetric matrix
    A = 0.5 * (A + A.T)
    A[np.diag_indices_from(A)] = 0
    return A        


def compute_edges_list(A, kth=8+1):
    # Get k-similar neighbor indices for each node

    num_nodes = A.shape[0]
    new_kth = num_nodes - kth
    
    if num_nodes > 9:
        knns = np.argpartition(A, new_kth-1, axis=-1)[:, new_kth:-1]
        knn_values = np.partition(A, new_kth-1, axis=-1)[:, new_kth:-1] # NEW
    else:
        # handling for graphs with less than kth nodes
        # in such cases, the resulting graph will be fully connected
        knns = np.tile(np.arange(num_nodes), num_nodes).reshape(num_nodes, num_nodes)
        knn_values = A # NEW
        
        # removing self loop
        if num_nodes != 1:
            knn_values = A[knns != np.arange(num_nodes)[:,None]].reshape(num_nodes,-1) # NEW
            knns = knns[knns != np.arange(num_nodes)[:,None]].reshape(num_nodes,-1)
    return knns, knn_values # NEW


class SuperPixDGL(torch.utils.data.Dataset):
    def __init__(self,
                 data_dir,
                 dataset,
                 split,
                 use_mean_px=True,
                 use_coord=True):

        self.split = split
        
        self.graph_lists = []
        
        if dataset == 'MNIST':
            self.img_size = 28
            with open(os.path.join(data_dir, 'mnist_75sp_%s.pkl' % split), 'rb') as f:
                self.labels, self.sp_data = pickle.load(f)
                self.graph_labels = torch.LongTensor(self.labels)
        elif dataset == 'CIFAR10':
            self.img_size = 32
            with open(os.path.join(data_dir, 'cifar10_150sp_%s.pkl' % split), 'rb') as f:
                self.labels, self.sp_data = pickle.load(f)
                self.graph_labels = torch.LongTensor(self.labels)
                
        self.use_mean_px = use_mean_px
        self.use_coord = use_coord
        self.n_samples = len(self.labels)
        
        self._prepare()
    
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.n_samples, self.split.upper()))
        self.Adj_matrices, self.node_features, self.edges_lists, self.edge_features = [], [], [], []
        for index, sample in enumerate(self.sp_data):
            mean_px, coord = sample[:2]
            
            try:
                coord = coord / self.img_size
            except AttributeError:
                VOC_has_variable_image_sizes = True
                
            if self.use_mean_px:
                A = compute_adjacency_matrix_images(coord, mean_px) # using super-pixel locations + features
            else:
                A = compute_adjacency_matrix_images(coord, mean_px, False) # using only super-pixel locations
            edges_list, edge_values_list = compute_edges_list(A) # NEW

            N_nodes = A.shape[0]
            
            mean_px = mean_px.reshape(N_nodes, -1)
            coord = coord.reshape(N_nodes, 2)
            x = np.concatenate((mean_px, coord), axis=1)

            edge_values_list = edge_values_list.reshape(-1) # NEW # TO DOUBLE-CHECK !
            
            self.node_features.append(x)
            self.edge_features.append(edge_values_list) # NEW
            self.Adj_matrices.append(A)
            self.edges_lists.append(edges_list)
        
        for index in range(len(self.sp_data)):
            g = dgl.DGLGraph()
            g.add_nodes(self.node_features[index].shape[0])
            g.ndata['feat'] = torch.Tensor(self.node_features[index]).half() 

            for src, dsts in enumerate(self.edges_lists[index]):
                # handling for 1 node where the self loop would be the only edge
                # since, VOC Superpixels has few samples (5 samples) with only 1 node
                if self.node_features[index].shape[0] == 1:
                    g.add_edges(src, dsts)
                else:
                    g.add_edges(src, dsts[dsts!=src])
            
            # adding edge features for Residual Gated ConvNet
            edge_feat_dim = g.ndata['feat'].shape[1] # dim same as node feature dim
            #g.edata['feat'] = torch.ones(g.number_of_edges(), edge_feat_dim).half() 
            g.edata['feat'] = torch.Tensor(self.edge_features[index]).unsqueeze(1).half()  # NEW 

            self.graph_lists.append(g)

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
    
    
class SuperPixDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name, num_val=5000):
        """
            Takes input standard image dataset name (MNIST/CIFAR10) 
            and returns the superpixels graph.
            
            This class uses results from the above SuperPix class.
            which contains the steps for the generation of the Superpixels
            graph from a superpixel .pkl file that has been given by
            https://github.com/bknyaz/graph_attention_pool
            
            Please refer the SuperPix class for details.
        """
        t_data = time.time()
        self.name = name

        use_mean_px = True # using super-pixel locations + features
        use_mean_px = False # using only super-pixel locations
        if use_mean_px:
            print('Adj matrix defined from super-pixel locations + features')
        else:
            print('Adj matrix defined from super-pixel locations (only)')
        use_coord = True
        self.test = SuperPixDGL("./data/superpixels", dataset=self.name, split='test', 
                            use_mean_px=use_mean_px, 
                            use_coord=use_coord)

        self.train_ = SuperPixDGL("./data/superpixels", dataset=self.name, split='train', 
                             use_mean_px=use_mean_px, 
                             use_coord=use_coord)

        _val_graphs, _val_labels = self.train_[:num_val]
        _train_graphs, _train_labels = self.train_[num_val:]

        self.val = DGLFormDataset(_val_graphs, _val_labels)
        self.train = DGLFormDataset(_train_graphs, _train_labels)

        print("[I] Data load time: {:.4f}s".format(time.time()-t_data))
        


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in SuperPixDataset class.
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

    

class SuperPixDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading Superpixels datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = 'data/superpixels/'
        with open(data_dir+name+'.pkl',"rb") as f:
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
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))
        #tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        #tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        #snorm_n = torch.cat(tab_snorm_n).sqrt()  
        #tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        #tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        #snorm_e = torch.cat(tab_snorm_e).sqrt()
        for idx, graph in enumerate(graphs):
            graphs[idx].ndata['feat'] = graph.ndata['feat'].float()
            graphs[idx].edata['feat'] = graph.edata['feat'].float()
        batched_graph = dgl.batch(graphs)
        
        return batched_graph, labels
    
    
    # prepare dense tensors for GNNs using them; such as RingGNN, 3WLGNN
    def collate_dense_gnn(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))
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
        
        in_dim = g.ndata['feat'].shape[1]
        
        # use node feats to prepare adj
        adj_node_feat = torch.stack([zero_adj for j in range(in_dim)])
        adj_node_feat = torch.cat([adj.unsqueeze(0), adj_node_feat], dim=0)
        
        for node, node_feat in enumerate(g.ndata['feat']):
            adj_node_feat[1:, node, node] = node_feat

        x_node_feat = adj_node_feat.unsqueeze(0)
        
        return x_node_feat, labels
    
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
        
        self.train = DGLFormDataset(self.train.graph_lists, self.train.graph_labels)
        self.val = DGLFormDataset(self.val.graph_lists, self.val.graph_labels)
        self.test = DGLFormDataset(self.test.graph_lists, self.test.graph_labels)

                            

