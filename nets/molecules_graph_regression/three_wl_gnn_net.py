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
        self.num_atom_type = net_params['num_atom_type']    # 'num_atom_type' is 'nodeclasses' as in RingGNN original repo
        self.num_bond_type = net_params['num_bond_type']
        depth_of_mlp = net_params['depth_of_mlp']
        hidden_dim = net_params['hidden_dim']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.layer_norm = net_params['layer_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.diag_pool_readout = True                     # if True, uses the new_suffix readout from original code
        num_classes = 1 # since regression problem
        
        block_features = [hidden_dim] * n_layers  # L here is the block number
        
        if not self.edge_feat:
            original_features_num = self.num_atom_type + 1  # Number of features of the input
        else:
            original_features_num = self.num_atom_type + self.num_bond_type + 1  # Number of features of the input

        # sequential mlp blocks
        last_layer_features = original_features_num
        self.reg_blocks = nn.ModuleList()
        for layer, next_layer_features in enumerate(block_features):
            mlp_block = RegularBlock(depth_of_mlp, last_layer_features, next_layer_features, self.residual)
            self.reg_blocks.append(mlp_block)
            last_layer_features = next_layer_features
        
        
        if self.diag_pool_readout:
            self.fc_layers = nn.ModuleList()
            for output_features in block_features:
                # each block's output will be pooled (thus have 2*output_features), and pass through a fully connected
                fc = FullyConnected(2*output_features, num_classes, activation_fn=None)
                self.fc_layers.append(fc)
        else:
            self.mlp_prediction = MLPReadout(sum(block_features)+original_features_num, num_classes)

    def forward(self, x_no_edge_feat, x_with_edge_feat):        
        x = x_no_edge_feat
        
        if self.edge_feat:
            x = x_with_edge_feat
        
        if self.diag_pool_readout:
            scores = torch.tensor(0, device=self.device, dtype=x.dtype)
        else:
            x_list = [x]
        
        for i, block in enumerate(self.reg_blocks):

            x = block(x)
            if self.diag_pool_readout:
                scores = self.fc_layers[i](diag_offdiag_maxpool(x)) + scores
            else:
                x_list.append(x)
        
        if self.diag_pool_readout:
            return scores
        else:
            # readout like RingGNN
            x_list = [torch.sum(torch.sum(x, dim=3), dim=2) for x in x_list]
            x_list = torch.cat(x_list, dim=1)
            
            x_out = self.mlp_prediction(x_list)
            return x_out
        

    
    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss
    