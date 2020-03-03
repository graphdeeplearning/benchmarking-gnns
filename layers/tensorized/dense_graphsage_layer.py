import torch
from torch import nn as nn
from torch.nn import functional as F

"""
    <Dense/Tensorzied version of the GraphSage layer>
    
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
    
    ! code started from the dgl diffpool examples dir
"""

class DenseGraphSage(nn.Module):
    def __init__(self, infeat, outfeat, residual=False, use_bn=True,
                 mean=False, add_self=False):
        super().__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        self.residual = residual
        
        if infeat != outfeat:
            self.residual = False
        
        self.W = nn.Linear(infeat, outfeat, bias=True)

        nn.init.xavier_uniform_(
            self.W.weight,
            gain=nn.init.calculate_gain('relu'))

    def forward(self, x, adj):
        h_in = x               # for residual connection
        
        if self.use_bn and not hasattr(self, 'bn'):
            self.bn = nn.BatchNorm1d(adj.size(1)).to(adj.device)

        if self.add_self:
            adj = adj + torch.eye(adj.size(0)).to(adj.device)

        if self.mean:
            adj = adj / adj.sum(1, keepdim=True)

        h_k_N = torch.matmul(adj, x)
        h_k = self.W(h_k_N)
        h_k = F.normalize(h_k, dim=2, p=2)
        h_k = F.relu(h_k)
        
        if self.residual:
            h_k = h_in + h_k    # residual connection
        
        if self.use_bn:
            h_k = self.bn(h_k)
        return h_k

    def __repr__(self):
        if self.use_bn:
            return 'BN' + super(DenseGraphSage, self).__repr__()
        else:
            return super(DenseGraphSage, self).__repr__()