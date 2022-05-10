import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
"""

from layers.graphsage_layer import GraphSageLayer, GraphSageLayerEdgeFeat, GraphSageLayerEdgeReprFeat
from layers.mlp_readout_layer import MLPReadout

class GraphSageNet(nn.Module):
    """
    Grahpsage network with multiple GraphSageLayer layers
    """
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        in_dim_edge = net_params['in_dim_edge']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        aggregator_type = net_params['sage_aggregator']
        n_layers = net_params['L']  
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        self.readout = net_params['readout']
        self.device = net_params['device']
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layer_type = {
            "edgereprfeat": GraphSageLayerEdgeReprFeat,
            "edgefeat": GraphSageLayerEdgeFeat,
            "isotropic": GraphSageLayer,
        }.get(net_params['layer_type'], GraphSageLayer)
        
        if self.layer_type == GraphSageLayerEdgeReprFeat:
            self.edge_feat = net_params['edge_feat']
            self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)
        
        self.layers = nn.ModuleList([self.layer_type(hidden_dim, hidden_dim, F.relu,
                                              dropout, aggregator_type, batch_norm, residual) for _ in range(n_layers-1)])
        self.layers.append(self.layer_type(hidden_dim, out_dim, F.relu, dropout, aggregator_type, batch_norm, residual))
        self.MLP_layer = MLPReadout(2*out_dim, 1)
        
    def forward(self, g, h, e):
        h = self.embedding_h(h.float())
        h = self.in_feat_dropout(h)

        if self.layer_type == GraphSageLayerEdgeReprFeat:
            if not self.edge_feat:
                e = torch.ones_like(e).to(self.device)
            e = self.embedding_e(e.float())

            for conv in self.layers:
                h, e = conv(g, h, e)
        else:
            for conv in self.layers:
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
