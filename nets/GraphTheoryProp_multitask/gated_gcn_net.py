import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
from layers.gated_gcn_layer import GatedGCNLayer
from layers.mlp_readout_layer import MLPReadout
from dgl.nn.pytorch import glob

class GatedGCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        hidden_dim = net_params['hidden_dim']        
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.device = net_params['device']
        self.batch_size = net_params['batch_size']
        self.use_gru = net_params['use_gru']
        
            
        self.pos_enc = net_params['pos_enc']
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)

        in_dim = 2
        self.embedding_h = nn.Linear(in_dim, hidden_dim)

        self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        
        self.layers = nn.ModuleList([ GatedGCNLayer(hidden_dim, hidden_dim, dropout, self.batch_norm,
                                                    self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GatedGCNLayer(hidden_dim, hidden_dim, dropout, self.batch_norm,
                                         self.residual))
       
        
        if self.use_gru:
            self.gru = nn.GRU(hidden_dim, hidden_dim)
        
        self.S2S = dgl.nn.pytorch.glob.Set2Set(hidden_dim, 1, 1)
        self.MLP_layer_graph = MLPReadout(hidden_dim*2, 3) # for 3 graph predictions
        self.MLP_layer_nodes = MLPReadout(hidden_dim, 3) # for 3 node predictions
        
        self.g = None     
        # Created self.g 
        # Need the batch of graphs to be accessed while node loss computation;
        # see loss() at the end of this file

        
    def forward(self, g, h, e, pos_enc=None):
        
        # input embedding
        h = self.embedding_h(h)
        if self.pos_enc:
            h_pos_enc = self.embedding_pos_enc(pos_enc) 
            h = h + h_pos_enc

        h = self.in_feat_dropout(h)
        
        # edge feature set to 1
        e = torch.ones(e.size(0),1).to(self.device) 
        e = self.embedding_e(e)
        
        
        # convnets
        for conv in self.layers:
            h_t, e = conv(g, h, e)
            
            if self.use_gru:
                # Use GRU
                h_t = h_t.unsqueeze(0)
                h = h.unsqueeze(0)
                h = self.gru(h, h_t)[1]

                # Recover back in regular form
                h = h.squeeze()
            else:
                h = h_t
            
        g.ndata['h'] = h
        
        # Set2Set Readout for Graph Tasks
        hg = self.S2S(g, h)

        self.g = g # For util; To be accessed in single_loss() for node loss computation
        
        return self.MLP_layer_nodes(h), self.MLP_layer_graph(hg)
        
    def loss(self, pred, label):
        nodes_loss = self.single_loss(pred[0], label[0], node_level=True)
        graph_loss = self.single_loss(pred[1], label[1])
        specific_loss = torch.cat((nodes_loss, graph_loss))
        return torch.mean(specific_loss), specific_loss
        
    def single_loss(self, pred, label, node_level=False):
        # for node-level
        if node_level:
            average_nodes = label.shape[0] / self.batch_size
            nodes_loss = (pred - label) ** 2

            # Implementing global add pool of the node losses, reference here
            # https://github.com/cvignac/SMP/blob/62161485150f4544ba1255c4fcd39398fe2ca18d/multi_task_utils/util.py#L99
            self.g.ndata['nodes_loss'] = nodes_loss
            global_add_pool_error = dgl.sum_nodes(self.g, 'nodes_loss') / average_nodes
            loss = torch.mean(global_add_pool_error, dim=0)
            return loss
        
        # for graph-level
        loss = torch.mean((pred - label) ** 2, dim=0)
        return loss
