import torch
from torch import nn as nn

"""
    <Dense/Tensorzied version of the Diffpool layer>
    
    DIFFPOOL:
    Z. Ying, J. You, C. Morris, X. Ren, W. Hamilton, and J. Leskovec, 
    Hierarchical graph representation learning with differentiable pooling (NeurIPS 2018)
    https://arxiv.org/pdf/1806.08804.pdf
    
    ! code started from dgl diffpool examples dir
"""

from .assignment_layer import DiffPoolAssignment
from .dense_graphsage_layer import DenseGraphSage


class EntropyLoss(nn.Module):
    # Return Scalar
    # loss used in diffpool
    def forward(self, adj, anext, s_l):
        entropy = (torch.distributions.Categorical(
            probs=s_l).entropy()).sum(-1).mean(-1)
        assert not torch.isnan(entropy)
        return entropy


class LinkPredLoss(nn.Module):
    # loss used in diffpool
    def forward(self, adj, anext, s_l):
        link_pred_loss = (
            adj - s_l.matmul(s_l.transpose(-1, -2))).norm(dim=(1, 2))
        link_pred_loss = link_pred_loss / (adj.size(1) * adj.size(2))
        return link_pred_loss.mean()


class DenseDiffPool(nn.Module):
    def __init__(self, nfeat, nnext, nhid, link_pred=False, entropy=True):
        super().__init__()
        self.link_pred = link_pred
        self.log = {}
        self.link_pred_layer = self.LinkPredLoss()
        self.embed = DenseGraphSage(nfeat, nhid, use_bn=True)
        self.assign = DiffPoolAssignment(nfeat, nnext)
        self.reg_loss = nn.ModuleList([])
        self.loss_log = {}
        if link_pred:
            self.reg_loss.append(LinkPredLoss())
        if entropy:
            self.reg_loss.append(EntropyLoss())

    def forward(self, x, adj, log=False):
        z_l = self.embed(x, adj)
        s_l = self.assign(x, adj)
        if log:
            self.log['s'] = s_l.cpu().numpy()
        xnext = torch.matmul(s_l.transpose(-1, -2), z_l)
        anext = (s_l.transpose(-1, -2)).matmul(adj).matmul(s_l)

        for loss_layer in self.reg_loss:
            loss_name = str(type(loss_layer).__name__)
            self.loss_log[loss_name] = loss_layer(adj, anext, s_l)
        if log:
            self.log['a'] = anext.cpu().numpy()
        return xnext, anext

