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
        self.num_node_type = net_params['in_dim']    # 'num_node_type' is 'nodeclasses' as in RingGNN original repo
        # node_classes = net_params['node_classes']
        avg_node_num = net_params['avg_node_num'] 
        radius = net_params['radius'] 
        hidden_dim = net_params['hidden_dim']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.n_classes = net_params['n_classes']
        self.layer_norm = net_params['layer_norm']
        self.residual = net_params['residual']
        self.device = net_params['device']
        
        self.depth = [torch.LongTensor([1+self.num_node_type])] + [torch.LongTensor([hidden_dim])] * n_layers
            
        self.equi_modulelist = nn.ModuleList([RingGNNEquivLayer(self.device, m, n,
                                                                 layer_norm=self.layer_norm,
                                                                 residual=self.residual,
                                                                 dropout=dropout,
                                                                 radius=radius,
                                                                 k2_init=0.5/avg_node_num) for m, n in zip(self.depth[:-1], self.depth[1:])])
        
        self.prediction = MLPReadout(torch.sum(torch.stack(self.depth)).item(), self.n_classes)

    def forward(self, x_with_node_feat):
        """
            CODE ADPATED FROM https://github.com/leichen2018/Ring-GNN/
            : preparing input to the model in form new_adj
        """
        x = x_with_node_feat
        # this x is the tensor with all info available => adj, node feat
        
        x_list = [x]
        for layer in self.equi_modulelist:    
            x = layer(x)
            x_list.append(x)
        
        # readout
        x_list = [torch.sum(x, dim=2) for x in x_list]
        x_list = torch.cat(x_list, dim=1)

        # reshaping in form of [n x d_out]
        x_out = x_list.squeeze().permute(1,0)
        
        x_out = self.prediction(x_out)

        return x_out
    
    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss
