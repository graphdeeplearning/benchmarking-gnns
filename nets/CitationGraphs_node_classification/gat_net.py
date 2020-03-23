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
        
        feat_drop = dropout
        attn_drop = dropout 
        negative_slope = 0.2
        residual = False
        self.layers = nn.ModuleList()
        self.activation = F.elu
        # input projection (no residual)
        self.layers.append(GATConv(
            in_dim, hidden_dim, num_heads,
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, n_layers):
            # due to multi-head, the in_dim = hidden_dim * num_heads
            self.layers.append(GATConv(
                hidden_dim * num_heads, hidden_dim, num_heads,
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.layers.append(GATConv(
            hidden_dim * num_heads, n_classes, 1,
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g, h, e, snorm_n, snorm_e):

        # GAT built-in
        for conv in self.layers[:-1]:
            h = conv(g, h).flatten(1)

        h = self.layers[-1](g, h).mean(1)
            
        return h
    
    
    def loss(self, pred, label):
        # Cross-entropy 
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)

        return loss



        
