import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""

from layers.gin_layer import GINLayer, ApplyNodeFunc, MLP

class GINNet(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()
        num_node_type = net_params['num_node_type']
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        self.n_layers = net_params['L']
        n_mlp_layers = net_params['n_mlp_GIN']               # GIN
        learn_eps = net_params['learn_eps_GIN']              # GIN
        neighbor_aggr_type = net_params['neighbor_aggr_GIN'] # GIN
        readout = net_params['readout']                      # this is graph_pooling_type
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        self.pos_enc = net_params['pos_enc']
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        else:
            in_dim = 1
            self.embedding_h = nn.Embedding(in_dim, hidden_dim)
        
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        
        for layer in range(self.n_layers):
            mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
            
            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, batch_norm, residual, 0, learn_eps))

        # Linear function for graph poolings (readout) of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(self.n_layers+1):
            self.linears_prediction.append(nn.Linear(hidden_dim, n_classes))
        
        if readout == 'sum':
            self.pool = SumPooling()
        elif readout == 'mean':
            self.pool = AvgPooling()
        elif readout == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

        
    def forward(self, g, h, e, pos_enc=None):
        
        if self.pos_enc:
            h = self.embedding_pos_enc(pos_enc) 
        else:
            h = self.embedding_h(h)
        
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h)
            hidden_rep.append(h)

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.linears_prediction[i](pooled_h)

        return score_over_layer
        
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss