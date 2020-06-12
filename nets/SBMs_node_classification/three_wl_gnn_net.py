import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import time

"""
    3WLGNN / ThreeWLGNN
    Provably Powerful Graph Networks (Maron et al., 2019)
    https://papers.nips.cc/paper/8488-provably-powerful-graph-networks.pdf
    
    CODE adapted from https://github.com/hadarser/ProvablyPowerfulGraphNetworks_torch/
"""

from layers.three_wl_gnn_layers import RegularBlock, MlpBlock, SkipConnection, FullyConnected, diag_offdiag_maxpool
from layers.mlp_readout_layer import MLPReadout

class ThreeWLGNNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        
        self.num_node_type = net_params['in_dim']
        depth_of_mlp = net_params['depth_of_mlp']
        hidden_dim = net_params['hidden_dim']
        dropout = net_params['dropout']
        n_layers = net_params['L']              
        self.n_classes = net_params['n_classes']
        self.layer_norm = net_params['layer_norm']
        self.residual = net_params['residual']
        self.device = net_params['device']
        self.gin_like_readout = True        # if True, uses GIN like readout, but without diag poool, since node task
        
        block_features = [hidden_dim] * n_layers  # L here is the block number
        
        original_features_num = self.num_node_type + 1  # Number of features of the input

        # sequential mlp blocks
        last_layer_features = original_features_num
        self.reg_blocks = nn.ModuleList()
        for layer, next_layer_features in enumerate(block_features):
            mlp_block = RegularBlock(depth_of_mlp, last_layer_features, next_layer_features, self.residual)
            self.reg_blocks.append(mlp_block)
            last_layer_features = next_layer_features

            
        if self.gin_like_readout:
            self.fc_layers = nn.ModuleList()
            for output_features in block_features:
                # each block's output will be pooled (thus have 2*output_features), and pass through a fully connected
                fc = FullyConnected(output_features, self.n_classes, activation_fn=None)
                self.fc_layers.append(fc)
        else:
            self.mlp_prediction = MLPReadout(sum(block_features)+original_features_num, self.n_classes)    
        
        
    def forward(self, x_with_node_feat):
        x = x_with_node_feat
        # this x is the tensor with all info available => adj, node feat
        
        if self.gin_like_readout:
            scores = torch.tensor(0, device=self.device, dtype=x.dtype)
        else:
            x_list = [x]
        
        for i, block in enumerate(self.reg_blocks):

            x = block(x)
            if self.gin_like_readout:
                x_out = torch.sum(x, dim=2)        # from [1 x d_out x n x n] to [1 x d_out x n]
                x_out = x_out.squeeze().permute(1,0)   # reshaping in form of [n x d_out]
                scores = self.fc_layers[i](x_out) + scores
            else:
                x_list.append(x)
        
        if self.gin_like_readout:
            return scores
        else:
            # readout
            x_list = [torch.sum(x, dim=2) for x in x_list]
            x_list = torch.cat(x_list, dim=1)

            # reshaping in form of [n x d_out]
            x_out = x_list.squeeze().permute(1,0)

            x_out = self.mlp_prediction(x_out)

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
