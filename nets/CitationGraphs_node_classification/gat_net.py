import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GATConv

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
from layers.gat_layer import GATLayer

class GATNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']

        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = net_params['device']
        self.dgl_builtin = net_params['builtin']

        feat_drop = dropout
        attn_drop = dropout 
        negative_slope = 0.2
        residual = False
        self.layers = nn.ModuleList()
        self.activation = F.elu
        # input projection (no residual)
        self.layers.append(GATLayer(
            in_dim, hidden_dim, num_heads,
            dropout, self.graph_norm, self.batch_norm, self.residual,
            activation=self.activation, dgl_builtin=self.dgl_builtin))
        # hidden layers
        for l in range(1, n_layers):
            # due to multi-head, the in_dim = hidden_dim * num_heads
            self.layers.append(GATLayer(
                hidden_dim * num_heads, hidden_dim, num_heads,
                dropout, self.graph_norm, self.batch_norm, self.residual,
                activation=self.activation, dgl_builtin=self.dgl_builtin))
        # output projection
        self.layers.append(GATLayer(
                hidden_dim * num_heads, n_classes, 1,
                dropout, self.graph_norm, self.batch_norm, self.residual,
                activation=None, dgl_builtin=self.dgl_builtin))

    def forward(self, g, h, e, snorm_n, snorm_e):

        for conv in self.layers[:-1]:
            h = conv(g, h, snorm_n)

        h = self.layers[-1](g, h, snorm_n)

        return h
    
    
    def loss(self, pred, label):
        # Cross-entropy 
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)

        return loss
