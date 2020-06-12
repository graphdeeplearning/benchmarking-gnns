import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""

from layers.gin_layer import GINLayer, ApplyNodeFunc, MLP
from layers.mlp_readout_layer import MLPReadout

class GINNet(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        dropout = net_params['dropout']
        self.n_layers = net_params['L']
        n_mlp_layers = net_params['n_mlp_GIN']               # GIN
        learn_eps = net_params['learn_eps_GIN']              # GIN
        neighbor_aggr_type = net_params['neighbor_aggr_GIN'] # GIN
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        self.device = net_params['device']
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        for layer in range(self.n_layers):
            mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, batch_norm, residual, 0, learn_eps))
        
        self.MLP_layer = MLPReadout(2*hidden_dim, 1)
    
    def forward(self, g, h, e):
        h = self.embedding_h(h.float())
        
        for conv in self.ginlayers:
            h = conv(g, h)
        g.ndata['h'] = h
        
        return h
    
    def edge_predictor(self, h_i, h_j):
        x = torch.cat([h_i, h_j], dim=1)
        x = self.MLP_layer(x)
        
        return torch.sigmoid(x)
    
    def loss(self, pos_out, neg_out):
        pos_loss = -torch.log(pos_out + 1e-15).mean()  # positive samples
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()  # negative samples
        loss = pos_loss + neg_loss
        
        return loss
