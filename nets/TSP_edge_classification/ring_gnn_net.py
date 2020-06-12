import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import time

"""
    Ring-GNN
    On the equivalence between graph isomorphism testing and function approximation with GNNs (Chen et al, 2019)
    https://arxiv.org/pdf/1905.12560v1.pdf
"""
from layers.ring_gnn_equiv_layer import RingGNNEquivLayer
from layers.mlp_readout_layer import MLPReadout

class RingGNNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.in_dim_node = net_params['in_dim']
        self.in_dim_edge = net_params['in_dim_edge']
        avg_node_num = net_params['avg_node_num'] 
        radius = net_params['radius'] 
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.layer_norm = net_params['layer_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        
        if self.edge_feat:
            self.depth = [torch.LongTensor([1+self.in_dim_node+self.in_dim_edge])] + [torch.LongTensor([hidden_dim])] * n_layers
        else:
            self.depth = [torch.LongTensor([1+self.in_dim_node])] + [torch.LongTensor([hidden_dim])] * n_layers
            
        self.equi_modulelist = nn.ModuleList([RingGNNEquivLayer(self.device, m, n,
                                                                 layer_norm=self.layer_norm,
                                                                 residual=self.residual,
                                                                 dropout=dropout,
                                                                 radius=radius,
                                                                 k2_init=0.5/avg_node_num) for m, n in zip(self.depth[:-1], self.depth[1:])])
        
        self.prediction = MLPReadout(torch.sum(torch.stack(self.depth)).item()*2, n_classes)

    def forward(self, x_no_edge_feat, x_with_edge_feat, edge_list):
        """
            CODE ADPATED FROM https://github.com/leichen2018/Ring-GNN/
        """

        x = x_no_edge_feat
        
        if self.edge_feat:
            x = x_with_edge_feat
        
        # this x is the tensor with all info available => adj, node feat, and edge feat (if edge_feat flag True)
        
        x_list = [x]
        for layer in self.equi_modulelist:    
            x = layer(x)
            x_list.append(x)
        
        x_list = [torch.sum(x, dim=2) for x in x_list]
        x_list = torch.cat(x_list, dim=1)
        
        # node_feats will be of size (num nodes, features)
        node_feats = x_list.squeeze(0).permute(1,0)
        
        # edge sources and destinations which are node indexes
        srcs, dsts = edge_list
        
        
        # To make a prediction for each edge e_{ij}, we first concatenate
        # node features h_i and h_j from the final GNN layer. 
        # The concatenated features are then passed to an MLP for prediction.
        edge_outs = [torch.cat([node_feats[srcs[idx].item()], node_feats[dsts[idx].item()]]) for idx in range(len(srcs))]        
        edge_outs = self.prediction(torch.stack(edge_outs))
        
        return edge_outs
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss(weight=None)
        loss = criterion(pred, label)

        return loss

