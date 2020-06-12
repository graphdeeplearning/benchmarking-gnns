import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
from layers.gated_gcn_layer import GatedGCNLayer, GatedGCNLayerEdgeFeatOnly, GatedGCNLayerIsotropic
from layers.mlp_readout_layer import MLPReadout

class GatedGCNNet(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        in_dim_edge = net_params['in_dim_edge']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.n_classes = n_classes
        self.device = net_params['device']
        
        self.layer_type = {
            "edgereprfeat": GatedGCNLayer,
            "edgefeat": GatedGCNLayerEdgeFeatOnly,
            "isotropic": GatedGCNLayerIsotropic,
        }.get(net_params['layer_type'], GatedGCNLayer)
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)
        self.layers = nn.ModuleList([ self.layer_type(hidden_dim, hidden_dim, dropout,
                                                      self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(self.layer_type(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(2*out_dim, n_classes)
        
    def forward(self, g, h, e):
        
        h = self.embedding_h(h.float())
        if not self.edge_feat:
            e = torch.ones_like(e).to(self.device)
        e = self.embedding_e(e.float())
        
        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h
        
        def _edge_feat(edges):
            e = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
            e = self.MLP_layer(e)
            return {'e': e}
        g.apply_edges(_edge_feat)
        
        return g.edata['e']
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss(weight=None)
        loss = criterion(pred, label)

        return loss
    