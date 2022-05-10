import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
from layers.gated_gcn_layer import GatedGCNLayer
from layers.mlp_readout_layer import MLPReadout

class GatedGCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_node_type = net_params['num_node_type']
        num_edge_type = net_params['num_edge_type']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.pos_enc = net_params['pos_enc']
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        else:
            in_dim = 1
            self.embedding_h = nn.Embedding(in_dim, hidden_dim)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_edge_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                       self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)

        
    def forward(self, g, h, e, pos_enc=None):

        # input embedding
        if self.pos_enc:
            h = self.embedding_pos_enc(pos_enc) 
        else:
            h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if not self.edge_feat: # edge feature set to 1
            e = torch.ones(e.size(0),1).to(self.device)
        e = self.embedding_e(e)   
        
        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        return self.MLP_layer(hg)
        
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
