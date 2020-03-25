import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import numpy as np

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
from layers.gcn_layer import GCNLayer
from dgl.nn.pytorch import GraphConv

class GCNNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.n_classes = n_classes
        self.device = net_params['device']
        self.dgl_builtin = net_params['builtin']

        self.layers = nn.ModuleList()
        # input
        self.layers.append(GCNLayer(in_dim, hidden_dim, F.relu, dropout,
            self.graph_norm, self.batch_norm, self.residual,
            dgl_builtin=self.dgl_builtin))

        # hidden
        self.layers.extend(nn.ModuleList([GCNLayer(hidden_dim, hidden_dim,
            F.relu, dropout, self.graph_norm, self.batch_norm, self.residual,
            dgl_builtin=self.dgl_builtin)
            for _ in range(n_layers-1)]))

        # output
        self.layers.append(GCNLayer(hidden_dim, n_classes, None, 0,
            self.graph_norm, self.batch_norm, self.residual,
            dgl_builtin=self.dgl_builtin))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, h, e, snorm_n, snorm_e):
      
        # GCN
        for i, conv in enumerate(self.layers):
            h = conv(g, h, snorm_n)
        return h

    
    def loss(self, pred, label):

        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)

        return loss











