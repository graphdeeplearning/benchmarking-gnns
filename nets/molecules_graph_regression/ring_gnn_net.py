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
        self.num_atom_type = net_params['num_atom_type']    # 'num_atom_type' is 'nodeclasses' as in RingGNN original repo
        self.num_bond_type = net_params['num_bond_type']
        avg_node_num = net_params['avg_node_num'] 
        radius = net_params['radius'] 
        hidden_dim = net_params['hidden_dim']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.layer_norm = net_params['layer_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        
        if self.edge_feat:
            self.depth = [torch.LongTensor([1+self.num_atom_type+self.num_bond_type])] + [torch.LongTensor([hidden_dim])] * n_layers
        else:
            self.depth = [torch.LongTensor([1+self.num_atom_type])] + [torch.LongTensor([hidden_dim])] * n_layers
            
        self.equi_modulelist = nn.ModuleList([RingGNNEquivLayer(self.device, m, n,
                                                                 layer_norm=self.layer_norm,
                                                                 residual=self.residual,
                                                                 dropout=dropout,
                                                                 radius=radius,
                                                                 k2_init=0.5/avg_node_num) for m, n in zip(self.depth[:-1], self.depth[1:])])
        
        self.prediction = MLPReadout(torch.sum(torch.stack(self.depth)).item(), 1) # 1 out dim since regression problem

    def forward(self, x_no_edge_feat, x_with_edge_feat):
        """
            CODE ADPATED FROM https://github.com/leichen2018/Ring-GNN/
        """
        
        x = x_no_edge_feat
        
        if self.edge_feat:
            x = x_with_edge_feat

        # this x is the tensor with all info available => adj, node feat and edge feat (if flag edge_feat true)

        x_list = [x]
        for layer in self.equi_modulelist:    
            x = layer(x)
            x_list.append(x)
        
        # # readout
        x_list = [torch.sum(torch.sum(x, dim=3), dim=2) for x in x_list]
        x_list = torch.cat(x_list, dim=1)
        
        x_out = self.prediction(x_list)

        return x_out
    
    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss



"""
OLD CODE BELOW Thu 14 May,2020 for ROLLBACK, just in case.
using the following code and depth of only 29->64->64 achieved 0.44 test MAE on ZINC

"""


##############################################################################################################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import dgl
# import time

# """
#     Ring-GNN
#     On the equivalence between graph isomorphism testing and function approximation with GNNs (Chen et al, 2019)
#     https://arxiv.org/pdf/1905.12560v1.pdf
# """
# from layers.gated_gcn_layer import GatedGCNLayer
# from layers.mlp_readout_layer import MLPReadout

# class RingGNNNet(nn.Module):
#     def __init__(self, net_params):
#         super().__init__()
#         self.num_atom_type = net_params['num_atom_type']    # 'num_atom_type' is 'nodeclasses' as in RingGNN original repo
#         self.num_bond_type = net_params['num_bond_type']
#         # node_classes = net_params['node_classes']
#         avg_node_num = net_params['avg_node_num'] #10
#         radius = net_params['radius'] #4
#         hidden_dim = net_params['hidden_dim']
#         out_dim = net_params['out_dim']
#         in_feat_dropout = net_params['in_feat_dropout']
#         dropout = net_params['dropout']
#         n_layers = net_params['L']
#         self.readout = net_params['readout']
#         self.graph_norm = net_params['graph_norm']
#         self.batch_norm = net_params['batch_norm']
#         self.residual = net_params['residual']
#         self.edge_feat = net_params['edge_feat']
#         self.device = net_params['device']
        
#         self.depth = [torch.LongTensor([self.num_atom_type+1]), torch.LongTensor([22]), torch.LongTensor([22]), torch.LongTensor([22]), torch.LongTensor([22])]  
#         #print(self.depth)
        
#         # for m, n in zip(self.depth[:-1], self.depth[1:]):
#         #     print(m,n)
        
#         self.equi_modulelist = nn.ModuleList([equi_2_to_2(self.device, m, n, radius = radius,
#                                                           k2_init = 0.5/avg_node_num) for m, n in zip(self.depth[:-1], self.depth[1:])])
#         #print(self.equi_modulelist)
#         self.prediction = MLPReadout(torch.sum(torch.stack(self.depth)).item(), 1) # 1 out dim since regression problem

#     def forward(self, g, h, e, snorm_n, snorm_e):
#         """
#             CODE ADPATED FROM https://github.com/leichen2018/Ring-GNN/
#             : preparing input to the model in form new_adj
#             : new_adj is of shape [num_atom_type x num_nodes_in_g x num_nodes_in_g]
#         """
#         adj = self._sym_normalize_adj(g.adjacency_matrix().to_dense())
#         nlabel_dict = {} 
#         for i in range(self.num_atom_type): nlabel_dict[i] = i 
#         new_adj = torch.stack([adj for j in range(self.num_atom_type + 1)])

#         for node, node_label in enumerate(g.ndata['feat']):
#             new_adj[nlabel_dict[node_label.item()]+1][node][node] = 1
#         """"""
        
#         h = new_adj.unsqueeze(0).to(self.device)
        
#         h_list = [h]
#         for layer in self.equi_modulelist:
#             h = F.relu(layer(h))
#             h_list.append(h)
        
#         h_list = [torch.sum(torch.sum(h, dim=3), dim=2) for h in h_list]
#         h_list = torch.cat(h_list, dim=1)
        
#         h_out = self.prediction(h_list)

#         return h_out
    
#     def _sym_normalize_adj(self, adj):
#         deg = torch.sum(adj, dim = 0)#.squeeze()
#         deg_inv = torch.where(deg>0, 1./torch.sqrt(deg), torch.zeros(deg.size()))
#         deg_inv = torch.diag(deg_inv)
#         return torch.mm(deg_inv, torch.mm(adj, deg_inv))
    
#     def loss(self, scores, targets):
#         # loss = nn.MSELoss()(scores,targets)
#         loss = nn.L1Loss()(scores, targets)
#         return loss

# class equi_2_to_2(nn.Module):
#     def __init__(self, device, input_depth, output_depth, normalization='inf', normalization_val=1.0, radius=2, k2_init = 0.1):
#         super(equi_2_to_2, self).__init__()
#         self.device = device
#         basis_dimension = 15
#         self.radius = radius
#         coeffs_values = lambda i, j, k: torch.randn([i, j, k]) * torch.sqrt(2. / (i + j).float())
#         self.diag_bias_list = nn.ParameterList([])

#         for i in range(radius):
#             for j in range(i+1):
#                 self.diag_bias_list.append(nn.Parameter(torch.zeros(1, output_depth, 1, 1)))

#         self.all_bias = nn.Parameter(torch.zeros(1, output_depth, 1, 1))
#         self.coeffs_list = nn.ParameterList([])

#         for i in range(radius):
#             for j in range(i+1):
#                 self.coeffs_list.append(nn.Parameter(coeffs_values(input_depth, output_depth, basis_dimension)))

#         self.switch = nn.ParameterList([nn.Parameter(torch.FloatTensor([1])), nn.Parameter(torch.FloatTensor([k2_init]))])
#         self.output_depth = output_depth

#         self.normalization = normalization
#         self.normalization_val = normalization_val


#     def forward(self, inputs):
#         m = inputs.size()[3]

#         ops_out = ops_2_to_2(inputs, m, normalization=self.normalization)
#         ops_out = torch.stack(ops_out, dim = 2)


#         output_list = []

#         for i in range(self.radius):
#             for j in range(i+1):
#                 output_i = torch.einsum('dsb,ndbij->nsij', self.coeffs_list[i*(i+1)//2 + j], ops_out)

#                 mat_diag_bias = torch.eye(inputs.size()[3]).unsqueeze(0).unsqueeze(0).to(self.device) * self.diag_bias_list[i*(i+1)//2 + j]
#                 # mat_diag_bias = torch.eye(inputs.size()[3]).to('cuda:0').unsqueeze(0).unsqueeze(0) * self.diag_bias_list[i*(i+1)//2 + j]
#                 if j == 0:
#                     output = output_i + mat_diag_bias
#                 else:
#                     output = torch.einsum('abcd,abde->abce', output_i, output)
#             output_list.append(output)

#         output = 0
#         for i in range(self.radius):
#             output += output_list[i] * self.switch[i]

#         output = output + self.all_bias
#         return output


# def ops_2_to_2(inputs, dim, normalization='inf', normalization_val=1.0): # N x D x m x m
#     # input: N x D x m x m
#     diag_part = torch.diagonal(inputs, dim1 = 2, dim2 = 3) # N x D x m
#     sum_diag_part = torch.sum(diag_part, dim=2, keepdim = True) # N x D x 1
#     sum_of_rows = torch.sum(inputs, dim=3) # N x D x m
#     sum_of_cols = torch.sum(inputs, dim=2) # N x D x m
#     sum_all = torch.sum(sum_of_rows, dim=2) # N x D

#     # op1 - (1234) - extract diag
#     op1 = torch.diag_embed(diag_part) # N x D x m x m
    
#     # op2 - (1234) + (12)(34) - place sum of diag on diag
#     op2 = torch.diag_embed(sum_diag_part.repeat(1, 1, dim))
    
#     # op3 - (1234) + (123)(4) - place sum of row i on diag ii
#     op3 = torch.diag_embed(sum_of_rows)

#     # op4 - (1234) + (124)(3) - place sum of col i on diag ii
#     op4 = torch.diag_embed(sum_of_cols)

#     # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
#     op5 = torch.diag_embed(sum_all.unsqueeze(2).repeat(1, 1, dim))
    
#     # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
#     op6 = sum_of_cols.unsqueeze(3).repeat(1, 1, 1, dim)
    
#     # op7 - (14)(23) + (23)(1)(4) + (234)(1) + (123)(4) + (1234) - place sum of row i on row i
#     op7 = sum_of_rows.unsqueeze(3).repeat(1, 1, 1, dim)

#     # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
#     op8 = sum_of_cols.unsqueeze(2).repeat(1, 1, dim, 1)

#     # op9 - (13)(24) + (13)(2)(4) + (134)(2) + (123)(4) + (1234) - place sum of row i on col i
#     op9 = sum_of_rows.unsqueeze(2).repeat(1, 1, dim, 1)

#     # op10 - (1234) + (14)(23) - identity
#     op10 = inputs

#     # op11 - (1234) + (13)(24) - transpose
#     op11 = torch.transpose(inputs, -2, -1)

#     # op12 - (1234) + (234)(1) - place ii element in row i
#     op12 = diag_part.unsqueeze(3).repeat(1, 1, 1, dim)

#     # op13 - (1234) + (134)(2) - place ii element in col i
#     op13 = diag_part.unsqueeze(2).repeat(1, 1, dim, 1)

#     # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
#     op14 = sum_diag_part.unsqueeze(3).repeat(1, 1, dim, dim)

#     # op15 - sum of all ops - place sum of all entries in all entries
#     op15 = sum_all.unsqueeze(2).unsqueeze(3).repeat(1, 1, dim, dim)

#     #A_2 = torch.einsum('abcd,abde->abce', inputs, inputs)
#     #A_4 = torch.einsum('abcd,abde->abce', A_2, A_2)
#     #op16 = torch.where(A_4>1, torch.ones(A_4.size()), A_4)

#     if normalization is not None:
#         float_dim = float(dim)
#         if normalization is 'inf':
#             op2 = torch.div(op2, float_dim)
#             op3 = torch.div(op3, float_dim)
#             op4 = torch.div(op4, float_dim)
#             op5 = torch.div(op5, float_dim**2)
#             op6 = torch.div(op6, float_dim)
#             op7 = torch.div(op7, float_dim)
#             op8 = torch.div(op8, float_dim)
#             op9 = torch.div(op9, float_dim)
#             op14 = torch.div(op14, float_dim)
#             op15 = torch.div(op15, float_dim**2)
    
#     #return [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15, op16]
#     '''
#     l = [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15]
#     for i, ls in enumerate(l):
#         print(i+1)
#         print(torch.sum(ls))
#     print("$%^&*(*&^%$#$%^&*(*&^%$%^&*(*&^%$%^&*(")
#     '''
#     return [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15]    