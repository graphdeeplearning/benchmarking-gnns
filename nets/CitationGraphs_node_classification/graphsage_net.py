import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
"""

from layers.graphsage_layer import GraphSageLayer
from layers.mlp_readout_layer import MLPReadout
from dgl.nn.pytorch.conv import SAGEConv

class GraphSageNet(nn.Module):
    """
    Grahpsage network with multiple GraphSageLayer layers
    """
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        aggregator_type = net_params['sage_aggregator']
        n_layers = net_params['L']
        self.residual = net_params['residual']
        dgl_builtin = net_params['builtin']
        bnorm = net_params['batch_norm']
        
        self.layers = nn.ModuleList()
        # Input
        self.layers.append(GraphSageLayer(in_dim, hidden_dim, F.relu,
            dropout, aggregator_type, self.residual,
            bn=bnorm, dgl_builtin=dgl_builtin))
        # Hidden layers
        self.layers.extend(nn.ModuleList([GraphSageLayer(hidden_dim,
            hidden_dim, F.relu, dropout, aggregator_type, self.residual,
            bn=bnorm, dgl_builtin=dgl_builtin) for _ in range(n_layers-1)]))
        # Output layer
        self.layers.append(GraphSageLayer(hidden_dim, n_classes, None,
            dropout, aggregator_type, self.residual, bn=bnorm,
            dgl_builtin=dgl_builtin))
        
    def forward(self, g, h, e, snorm_n, snorm_e):
        for conv in self.layers:
            h = conv(g, h)
        return h

        
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
    
