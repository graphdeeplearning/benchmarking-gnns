import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from layers.mlp_readout_layer import MLPReadout

class MatrixFactorization(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_embs = net_params['num_embs']
        hidden_dim = net_params['hidden_dim']
        self.device = net_params['device']
        
        # MF trains a hidden embedding per graph node
        self.emb = torch.nn.Embedding(num_embs, hidden_dim)
        
        self.readout_mlp = MLPReadout(2*hidden_dim, 1)

    def forward(self, g, h, e):
        # Return the entire node embedding matrix
        return self.emb.weight
        
    def edge_predictor(self, h_i, h_j):
        x = torch.cat([h_i, h_j], dim=1)
        x = self.readout_mlp(x)
        
        return torch.sigmoid(x)
    
    def loss(self, pos_out, neg_out):
        pos_loss = -torch.log(pos_out + 1e-15).mean()  # positive samples
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()  # negative samples
        loss = pos_loss + neg_loss
        
        return loss
