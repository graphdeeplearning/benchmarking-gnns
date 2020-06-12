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
        self.in_dim_node = net_params['in_dim']
        self.in_dim_edge = net_params['in_dim_edge']
        depth_of_mlp = net_params['depth_of_mlp']
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.layer_norm = net_params['layer_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.gin_like_readout = False        # if True, uses GIN like readout, but without diag poool, since node task
        
        block_features = [hidden_dim] * n_layers  # L here is the block number
        
        if not self.edge_feat:
            original_features_num = self.in_dim_node + 1  # Number of features of the input
        else:
            original_features_num = self.in_dim_node + self.in_dim_edge + 1  # Number of features of the input

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
                fc = FullyConnected(2*output_features, n_classes, activation_fn=None)
                self.fc_layers.append(fc)
        else:
            self.mlp_prediction = MLPReadout(2*(sum(block_features)+original_features_num), n_classes)


    def forward(self, x_no_edge_feat, x_with_edge_feat, edge_list):

        x = x_no_edge_feat
        
        if self.edge_feat:
            x = x_with_edge_feat
            
        # this x is the tensor with all info available => adj, node feat, and edge feat (if edge_feat flag True)
        
        if self.gin_like_readout:
            scores = torch.tensor(0, device=self.device, dtype=x.dtype)
        else:
            x_list = [x]
            
        for i, block in enumerate(self.reg_blocks):

            x = block(x)
            if self.gin_like_readout:
                x_out = torch.sum(x, dim=2)        # from [1 x d_out x n x n] to [1 x d_out x n]
                node_feats = x_out.squeeze().permute(1,0)   # reshaping in form of [n x d_out]
                
                # edge sources and destinations which are node indexes
                srcs, dsts = edge_list

                # To make a prediction for each edge e_{ij}, we first concatenate
                # node features h_i and h_j from the final GNN layer. 
                # The concatenated features are then passed to an MLP for prediction.
                edge_outs = [torch.cat([node_feats[srcs[idx].item()], node_feats[dsts[idx].item()]]) for idx in range(len(srcs))]        
                    
                scores = self.fc_layers[i](torch.stack(edge_outs)) + scores
            else:
                x_list.append(x)
        
        if self.gin_like_readout:
            return scores
        else:
            # readout    
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
            edge_outs = self.mlp_prediction(torch.stack(edge_outs))

            return edge_outs

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss(weight=None)
        loss = criterion(pred, label)

        return loss
    