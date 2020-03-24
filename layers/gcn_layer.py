import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn.pytorch import GraphConv

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
    
# Sends a message of node feature h
# Equivalent to => return {'m': edges.src['h']}
msg = fn.copy_src(src='h', out='m')

def mean_reduce(nodes):
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}

def sum_reduce(nodes):
    accum = torch.sum(nodes.mailbox['m'], 1)
    return {'h': accum}

class NodeApplyModule(nn.Module):
    # Update node feature h_v with (Wh_v+b)
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, node):
        h = self.linear(node.data['h'])
        return {'h': h}

class GCNLayer(nn.Module):
    """
        Param: [in_dim, out_dim]
    """
    def __init__(self, in_dim, out_dim, activation, dropout, graph_norm, batch_norm, residual=False, reducer='mean', dgl_builtin=False):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.dgl_builtin = dgl_builtin
        
        if in_dim != out_dim:
            self.residual = False
        
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        if self.dgl_builtin == False:
            self.apply_mod = NodeApplyModule(in_dim, out_dim)
        else:
            self.conv = GraphConv(in_dim, out_dim)

        if reducer == 'mean':
            self.reduce = mean_reduce
        elif reducer == 'sum':
            self.reduce = sum_reduce
        
    def forward(self, g, feature, snorm_n):
        h_in = feature   # to be used for residual connection
        if self.dgl_builtin == False:
            if self.graph_norm:
                feature = feature * snorm_n # normalize activation w.r.t. graph size
            g.ndata['h'] = feature
            g.update_all(msg, self.reduce)
            g.apply_nodes(func=self.apply_mod)
            h = g.ndata['h'] # result of graph convolution
            # Note that is already done in builtin
            if self.graph_norm:
                h = h * snorm_n # normalize activation w.r.t. graph size
        else:
            h = self.conv(g, feature)

        
        if self.batch_norm:
            h = self.batchnorm_h(h) # batch normalization  
       
        if self.activation:
            h = self.activation(h)
        
        if self.residual:
            h = h_in + h # residual connection
            
        h = self.dropout(h)
        return h
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.residual)
