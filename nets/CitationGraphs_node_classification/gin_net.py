import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""

from layers.gin_layer import GINLayer, ApplyNodeFunc, MLP

class GINNet(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        self.n_layers = net_params['L']
        n_mlp_layers = net_params['n_mlp_GIN']               # GIN
        learn_eps = net_params['learn_eps_GIN']              # GIN
        neighbor_aggr_type = net_params['neighbor_aggr_GIN'] # GIN
        graph_norm = net_params['graph_norm']      
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']     
        
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
       
        # Input layer
        mlp = MLP(1, in_dim, hidden_dim, hidden_dim)
        self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                       dropout, graph_norm, batch_norm,
                                       residual, 0, learn_eps,
                                       activation=F.relu))

        # Hidden layers
        for layer in range(self.n_layers-1):
            mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
            
            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp),
                neighbor_aggr_type, dropout, graph_norm, batch_norm, residual,
                0, learn_eps, activation=F.relu))

        # Output layer
        mlp = MLP(1, hidden_dim, n_classes, n_classes)
        self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                       dropout, graph_norm, batch_norm,
                                       residual, 0, learn_eps))
        
    def forward(self, g, h, e, snorm_n, snorm_e):
        
        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h, snorm_n)


        return h
        
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
