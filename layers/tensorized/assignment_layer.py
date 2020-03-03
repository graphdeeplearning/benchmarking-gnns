import torch

from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

"""
    This layer is the generating the Assignment matrix as shown in
    equation (6) of the DIFFPOOL paper.
    ! code started from dgl diffpool examples dir
"""

from .dense_graphsage_layer import DenseGraphSage

class DiffPoolAssignment(nn.Module):
    def __init__(self, nfeat, nnext):
        super().__init__()
        self.assign_mat = DenseGraphSage(nfeat, nnext, use_bn=True)

    def forward(self, x, adj, log=False):
        s_l_init = self.assign_mat(x, adj)
        s_l = F.softmax(s_l_init, dim=-1)
        return s_l