import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
from layers.gcn_layer import GCNLayer
from layers.mlp_readout_layer import MLPReadout

class GCNNet(nn.Module):
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
        self.pos_enc = net_params['pos_enc']
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        else:
            in_dim = 1
            self.embedding_h = nn.Embedding(in_dim, hidden_dim)
        
                
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        #self.embedding_h = nn.Embedding(num_node_type, hidden_dim)
        
        self.layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim, F.relu,
                                              dropout, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GCNLayer(hidden_dim, out_dim, F.relu,
                                    dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)
        

    def forward(self, g, h, e, pos_enc=None):
        # input embedding
        
        if self.pos_enc:
            h = self.embedding_pos_enc(pos_enc) 
        else:
            h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        
        for conv in self.layers:
            h = conv(g, h)
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