import numpy as np, time, pickle, random, csv
import torch
from torch.utils.data import DataLoader, Dataset

import os
import pickle
import numpy as np

import dgl

from sklearn.model_selection import StratifiedKFold, train_test_split

random.seed(42)

from scipy import sparse as sp


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
    
def format_dataset(dataset):  
    """
        Utility function to recover data,
        INTO-> dgl/pytorch compatible format 
    """
    graphs = [data[0] for data in dataset]
    labels = [data[1] for data in dataset]

    for graph in graphs:
        #graph.ndata['feat'] = torch.FloatTensor(graph.ndata['feat'])
        graph.ndata['feat'] = graph.ndata['feat'].float() # dgl 4.0
        # adding edge features for Residual Gated ConvNet, if not there
        if 'feat' not in graph.edata.keys():
            edge_feat_dim = graph.ndata['feat'].shape[1] # dim same as node feature dim
            graph.edata['feat'] = torch.ones(graph.number_of_edges(), edge_feat_dim)

    return DGLFormDataset(graphs, labels)


def get_all_split_idx(dataset):
    """
        - Split total number of graphs into 3 (train, val and test) in 3:1:1
        - Stratified split proportionate to original distribution of data with respect to classes
        - Using sklearn to perform the split and then save the indexes
        - Preparing 5 such combinations of indexes split to be used in Graph NNs
        - As with KFold, each of the 5 fold have unique test set.
    """
    root_idx_dir = './data/CSL/'
    if not os.path.exists(root_idx_dir):
        os.makedirs(root_idx_dir)
    all_idx = {}
    
    # If there are no idx files, do the split and store the files
    if not (os.path.exists(root_idx_dir + dataset.name + '_train.index')):
        print("[!] Splitting the data into train/val/test ...")
        
        # Using 5-fold cross val as used in RP-GNN paper
        k_splits = 5

        cross_val_fold = StratifiedKFold(n_splits=k_splits, shuffle=True)
        k_data_splits = []
        
        # this is a temporary index assignment, to be used below for val splitting
        for i in range(len(dataset.graph_lists)):
            dataset[i][0].a = lambda: None
            setattr(dataset[i][0].a, 'index', i)
            
        for indexes in cross_val_fold.split(dataset.graph_lists, dataset.graph_labels):
            remain_index, test_index = indexes[0], indexes[1]    

            remain_set = format_dataset([dataset[index] for index in remain_index])

            # Gets final 'train' and 'val'
            train, val, _, __ = train_test_split(remain_set,
                                                    range(len(remain_set.graph_lists)),
                                                    test_size=0.25,
                                                    stratify=remain_set.graph_labels)

            train, val = format_dataset(train), format_dataset(val)
            test = format_dataset([dataset[index] for index in test_index])

            # Extracting only idxs
            idx_train = [item[0].a.index for item in train]
            idx_val = [item[0].a.index for item in val]
            idx_test = [item[0].a.index for item in test]

            f_train_w = csv.writer(open(root_idx_dir + dataset.name + '_train.index', 'a+'))
            f_val_w = csv.writer(open(root_idx_dir + dataset.name + '_val.index', 'a+'))
            f_test_w = csv.writer(open(root_idx_dir + dataset.name + '_test.index', 'a+'))
            
            f_train_w.writerow(idx_train)
            f_val_w.writerow(idx_val)
            f_test_w.writerow(idx_test)

        print("[!] Splitting done!")
        
    # reading idx from the files
    for section in ['train', 'val', 'test']:
        with open(root_idx_dir + dataset.name + '_'+ section + '.index', 'r') as f:
            reader = csv.reader(f)
            all_idx[section] = [list(map(int, idx)) for idx in reader]
    return all_idx



class CSL(torch.utils.data.Dataset):
    """
        Circular Skip Link Graphs: 
        Source: https://github.com/PurdueMINDS/RelationalPooling/
    """
    
    def __init__(self, path="data/CSL/"):
        self.name = "CSL"
        self.adj_list = pickle.load(open(os.path.join(path, 'graphs_Kary_Deterministic_Graphs.pkl'), 'rb'))
        self.graph_labels = torch.load(os.path.join(path, 'y_Kary_Deterministic_Graphs.pt'))
        self.graph_lists = []
        
        self.n_samples = len(self.graph_labels)
        self.num_node_type = 1 #41
        self.num_edge_type = 1 #164
        self._prepare()
        
    def _prepare(self):
        t0 = time.time()
        print("[I] Preparing Circular Skip Link Graphs v4 ...")
        for sample in self.adj_list:
            _g = dgl.from_scipy(sample)
            g = dgl.transform.remove_self_loop(_g)
            g.ndata['feat'] = torch.zeros(g.number_of_nodes()).long()
            #g.ndata['feat'] = torch.arange(0, g.number_of_nodes()).long() # v1
            #g.ndata['feat'] = torch.randperm(g.number_of_nodes()).long() # v3
                
            # adding edge features as generic requirement
            g.edata['feat'] = torch.zeros(g.number_of_edges()).long()
            #g.edata['feat'] = torch.arange(0, g.number_of_edges()).long() # v1
            #g.edata['feat'] = torch.ones(g.number_of_edges()).long() # v2
            
            # NOTE: come back here, to define edge features as distance between the indices of the edges
            ###################################################################
            # srcs, dsts = new_g.edges()
            # edge_feat = []
            # for edge in range(len(srcs)):
            #     a = srcs[edge].item()
            #     b = dsts[edge].item()
            #     edge_feat.append(abs(a-b))
            # g.edata['feat'] = torch.tensor(edge_feat, dtype=torch.int).long()
            ###################################################################
            
            self.graph_lists.append(g)
        self.num_node_type = self.graph_lists[0].ndata['feat'].size(0)
        self.num_edge_type = self.graph_lists[0].edata['feat'].size(0)
        print("[I] Finished preparation after {:.4f}s".format(time.time()-t0))
            
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.graph_lists[idx], self.graph_labels[idx]
    
    
    
def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in TUsDataset class.
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
    n = g.number_of_nodes()
    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(n) - N * A * N
    # Eigenvectors
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()
    return g


class CSLDataset(torch.utils.data.Dataset):
    def __init__(self, name='CSL'):
        t0 = time.time()
        self.name = name
        
        dataset = CSL()

        print("[!] Dataset: ", self.name)

        # this function splits data into train/val/test and returns the indices
        self.all_idx = get_all_split_idx(dataset)
        
        self.all = dataset
        self.train = [self.format_dataset([dataset[idx] for idx in self.all_idx['train'][split_num]]) for split_num in range(5)]
        self.val = [self.format_dataset([dataset[idx] for idx in self.all_idx['val'][split_num]]) for split_num in range(5)]
        self.test = [self.format_dataset([dataset[idx] for idx in self.all_idx['test'][split_num]]) for split_num in range(5)]
        
        print("Time taken: {:.4f}s".format(time.time()-t0))
    
    def format_dataset(self, dataset):  
        """
            Utility function to recover data,
            INTO-> dgl/pytorch compatible format 
        """
        graphs = [data[0] for data in dataset]
        labels = [data[1] for data in dataset]

        return DGLFormDataset(graphs, labels)
    
    
    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels
    

    # prepare dense tensors for GNNs using them; such as RingGNN, 3WLGNN
    def collate_dense_gnn(self, samples, pos_enc):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))
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
        if pos_enc:
            in_dim = g.ndata['pos_enc'].shape[1]        
            # use node feats to prepare adj
            adj_node_feat = torch.stack([zero_adj for j in range(in_dim)])
            adj_node_feat = torch.cat([adj.unsqueeze(0), adj_node_feat], dim=0)
            for node, node_feat in enumerate(g.ndata['pos_enc']):
                adj_node_feat[1:, node, node] = node_feat
            x_node_feat = adj_node_feat.unsqueeze(0)
            return x_node_feat, labels
        else: # no node features here
            in_dim = 1
            # use node feats to prepare adj
            adj_node_feat = torch.stack([zero_adj for j in range(in_dim)])
            adj_node_feat = torch.cat([adj.unsqueeze(0), adj_node_feat], dim=0)
            for node, node_feat in enumerate(g.ndata['feat']):
                adj_node_feat[1:, node, node] = node_feat
            x_no_node_feat = adj_node_feat.unsqueeze(0)
            return x_no_node_feat, labels
    
    def _sym_normalize_adj(self, adj):
        deg = torch.sum(adj, dim = 0)#.squeeze()
        deg_inv = torch.where(deg>0, 1./torch.sqrt(deg), torch.zeros(deg.size()))
        deg_inv = torch.diag(deg_inv)
        return torch.mm(deg_inv, torch.mm(adj, deg_inv))
    



    
    def _add_self_loops(self):

        # function for adding self loops
        # this function will be called only if self_loop flag is True
        for split_num in range(5):
            self.train[split_num].graph_lists = [self_loop(g) for g in self.train[split_num].graph_lists]
            self.val[split_num].graph_lists = [self_loop(g) for g in self.val[split_num].graph_lists]
            self.test[split_num].graph_lists = [self_loop(g) for g in self.test[split_num].graph_lists]
            
        for split_num in range(5):
            self.train[split_num] = DGLFormDataset(self.train[split_num].graph_lists, self.train[split_num].graph_labels)
            self.val[split_num] = DGLFormDataset(self.val[split_num].graph_lists, self.val[split_num].graph_labels)
            self.test[split_num] = DGLFormDataset(self.test[split_num].graph_lists, self.test[split_num].graph_labels)


    def _add_positional_encodings(self, pos_enc_dim):
        
        # Graph positional encoding v/ Laplacian eigenvectors
        for split_num in range(5):
            self.train[split_num].graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.train[split_num].graph_lists]
            self.val[split_num].graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.val[split_num].graph_lists]
            self.test[split_num].graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.test[split_num].graph_lists]





