import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
"""

from layers.sage_aggregator_layer import MaxPoolAggregator, MeanAggregator, LSTMAggregator
from layers.node_apply_layer import NodeApply

class GraphSageLayer(nn.Module):

    def __init__(self, in_feats, out_feats, activation, dropout,
                 aggregator_type, residual=False, bn=False, bias=True):
        super().__init__()
        self.in_channels = in_feats
        self.out_channels = out_feats
        self.aggregator_type = aggregator_type
        self.residual = residual
        
        if in_feats != out_feats:
            self.residual = False
        
        self.nodeapply = NodeApply(in_feats, out_feats, activation, dropout,
                               bias=bias)
        self.dropout = nn.Dropout(p=dropout)

        if aggregator_type == "maxpool":
            self.aggregator = MaxPoolAggregator(in_feats, in_feats,
                                                activation, bias)
        elif aggregator_type == "lstm":
            self.aggregator = LSTMAggregator(in_feats, in_feats)
        else:
            self.aggregator = MeanAggregator()
        
        self.batchnorm_h = nn.BatchNorm1d(out_feats)

    def forward(self, g, h, snorm_n=None):
        h_in = h              # for residual connection
        h = self.dropout(h)
        g.ndata['h'] = h
        g.update_all(fn.copy_src(src='h', out='m'), self.aggregator,
                     self.nodeapply)
        h = g.ndata['h']
        if snorm_n is not None:
            h = h * snorm_n
        h = self.batchnorm_h(h)
        
        if self.residual:
            h = h_in + h       # residual connection
        
        return h
    
#     def __repr__(self):
#         return '{}(in_channels={}, out_channels={}, aggregator={}, residual={})'.format(self.__class__.__name__,
#                                              self.in_channels,
#                                              self.out_channels, self.aggregator_type, self.residual)