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
        hidden_dim = net_params['hidden_dim']
        dropout = net_params['dropout']
        self.n_layers = net_params['L']
        n_mlp_layers = net_params['n_mlp_GIN']               # GIN
        learn_eps = net_params['learn_eps_GIN']              # GIN
        neighbor_aggr_type = net_params['neighbor_aggr_GIN'] # GIN
        readout = net_params['readout']                      # this is graph_pooling_type
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        
        self.batch_size = net_params['batch_size']
        self.use_gru = net_params['use_gru']
        
        self.pos_enc = net_params['pos_enc']
        
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)

        in_dim = 2
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        
        
        
        self.layers = nn.ModuleList([GINLayer(ApplyNodeFunc(MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim)), neighbor_aggr_type,
                                       dropout, batch_norm, residual, 0, learn_eps) for _
                                     in range(self.n_layers - 1)])
        self.layers.append(GINLayer(ApplyNodeFunc(MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim)), neighbor_aggr_type,
                                       dropout, batch_norm, residual, 0, learn_eps))
            

        # Linear function for graph poolings (readout) of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction_graph = torch.nn.ModuleList()
        self.linears_prediction_nodes = torch.nn.ModuleList()
    
        if self.use_gru:
            self.gru = nn.GRU(hidden_dim, hidden_dim)
    
        for layer in range(self.n_layers+1):
            self.linears_prediction_graph.append(nn.Linear(hidden_dim*2, 3)) # for 3 graph predictions
            
        for layer in range(self.n_layers+1):
            self.linears_prediction_nodes.append(nn.Linear(hidden_dim, 3)) # for 3 node predictions
            
        self.pool = dgl.nn.pytorch.glob.Set2Set(hidden_dim, 1, 1)
        
        self.g = None     
        # Created self.g 
        # Need the batch of graphs to be accessed while node loss computation;
        # see loss() at the end of this file

        
    def forward(self, g, h, e, pos_enc=None):
        
        # input embedding
        h = self.embedding_h(h)
        if self.pos_enc:
            h_pos_enc = self.embedding_pos_enc(pos_enc) 
            h = h + h_pos_enc
        
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for conv in self.layers:
            h_t = conv(g, h)
            
            if self.use_gru:
                # Use GRU
                h_t = h_t.unsqueeze(0)
                h = h.unsqueeze(0)
                h = self.gru(h, h_t)[1]

                # Recover back in regular form
                h = h.squeeze()
            else:
                h = h_t
            
            hidden_rep.append(h)

        self.g = g # To be accessed in single_loss() for node loss computation    
        
        score_over_layer_nodes, score_over_layer_graph = 0, 0

        for i, h in enumerate(hidden_rep):
            score_over_layer_nodes += self.linears_prediction_nodes[i](h)
        
        # perform S2S pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer_graph += self.linears_prediction_graph[i](pooled_h)

        return score_over_layer_nodes, score_over_layer_graph
        
    def loss(self, pred, label):
        nodes_loss = self.single_loss(pred[0], label[0], node_level=True)
        graph_loss = self.single_loss(pred[1], label[1])
        specific_loss = torch.cat((nodes_loss, graph_loss))
        return torch.mean(specific_loss), specific_loss
        
    def single_loss(self, pred, label, node_level=False):
        # for node-level
        if node_level:
            average_nodes = label.shape[0] / self.batch_size
            nodes_loss = (pred - label) ** 2

            # Implementing global add pool of the node losses, reference here
            # https://github.com/cvignac/SMP/blob/62161485150f4544ba1255c4fcd39398fe2ca18d/multi_task_utils/util.py#L99
            self.g.ndata['nodes_loss'] = nodes_loss
            global_add_pool_error = dgl.sum_nodes(self.g, 'nodes_loss') / average_nodes
            loss = torch.mean(global_add_pool_error, dim=0)
            return loss
        
        # for graph-level
        loss = torch.mean((pred - label) ** 2, dim=0)
        return loss