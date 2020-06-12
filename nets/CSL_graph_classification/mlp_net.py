import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from layers.mlp_readout_layer import MLPReadout

class MLPNet(nn.Module):
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
        self.gated = net_params['gated']
        self.readout = net_params['readout']
        self.pos_enc = net_params['pos_enc']
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        else:
            in_dim = 1
            self.embedding_h = nn.Embedding(in_dim, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        feat_mlp_modules = [
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(n_layers-1):
            feat_mlp_modules.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
            feat_mlp_modules.append(nn.ReLU())
            feat_mlp_modules.append(nn.Dropout(dropout))
        self.feat_mlp = nn.Sequential(*feat_mlp_modules)
        
        if self.gated:
            self.gates = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        self.readout_mlp = MLPReadout(out_dim, n_classes)
        

    def forward(self, g, h, e, pos_enc=None):

        # input embedding
        if self.pos_enc:
            h = self.embedding_pos_enc(pos_enc) 
        else:
            h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        h = self.feat_mlp(h)
        if self.gated:
            h = torch.sigmoid(self.gates(h)) * h
            g.ndata['h'] = h       
            hg = dgl.sum_nodes(g, 'h')
            # hg = torch.cat(
            #     (
            #         dgl.sum_nodes(g, 'h'),
            #         dgl.max_nodes(g, 'h')
            #     ),
            #     dim=1
            # )
        
        else:
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')
        
        return self.readout_mlp(hg)
        
        
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
       