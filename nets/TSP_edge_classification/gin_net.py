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
        in_dim = net_params['in_dim']
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
        self.n_classes = n_classes
        self.device = net_params['device']
        
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        
        for layer in range(self.n_layers):
            mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
            
            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, batch_norm, residual, 0, learn_eps))

        # Non-linear function for output of each layer
        # which maps the output of different layers into a prediction score
        self.prediction = torch.nn.ModuleList()

        for layer in range(self.n_layers + 1):
            self.prediction.append(
                nn.Sequential(
                    nn.Linear(2*hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, n_classes)
                )
            )
        
    def forward(self, g, h, e):
        
        def _edge_feat(edges):
            e = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
            return {'e': e}
        
        h = self.embedding_h(h.float())
        g.ndata['h'] = h
        g.apply_edges(_edge_feat)
        
        # list of hidden representation at each layer (including input)
        hidden_rep = [g.edata['e']]
        
        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h)
            g.ndata['h'] = h
            g.apply_edges(_edge_feat)
            hidden_rep.append(g.edata['e'])

        score_over_layer = 0
        for i, e in enumerate(hidden_rep):
            score_over_layer += self.prediction[i](e)
        
        return score_over_layer
        
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss(weight=None)
        loss = criterion(pred, label)

        return loss