import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import dgl.function as fn

"""
    GMM: Gaussian Mixture Model Convolution layer
    Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs (Federico Monti et al., CVPR 2017)
    https://arxiv.org/pdf/1611.08402.pdf
"""

class GMMLayer(nn.Module):
    """
    [!] code adapted from dgl implementation of GMMConv

    Parameters
    ----------
    in_dim : 
        Number of input features.
    out_dim : 
        Number of output features.
    dim : 
        Dimensionality of pseudo-coordinte.
    kernel : 
        Number of kernels :math:`K`.
    aggr_type : 
        Aggregator type (``sum``, ``mean``, ``max``).
    dropout :
        Required for dropout of output features.
    graph_norm : 
        boolean flag for output features normalization w.r.t. graph sizes.
    batch_norm :
        boolean flag for batch_norm layer.
    residual : 
        If True, use residual connection inside this layer. Default: ``False``.
    bias : 
        If True, adds a learnable bias to the output. Default: ``True``.
    
    """
    def __init__(self, in_dim, out_dim, dim, kernel, aggr_type, dropout,
                 graph_norm, batch_norm, residual=False, bias=True):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim = dim
        self.kernel = kernel
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout
        
        if aggr_type == 'sum':
            self._reducer = fn.sum
        elif aggr_type == 'mean':
            self._reducer = fn.mean
        elif aggr_type == 'max':
            self._reducer = fn.max
        else:
            raise KeyError("Aggregator type {} not recognized.".format(aggr_type))

        self.mu = nn.Parameter(torch.Tensor(kernel, dim))
        self.inv_sigma = nn.Parameter(torch.Tensor(kernel, dim))
        self.fc = nn.Linear(in_dim, kernel * out_dim, bias=False)
        
        self.bn_node_h = nn.BatchNorm1d(out_dim)
        
        if in_dim != out_dim:
            self.residual = False
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = init.calculate_gain('relu')
        init.xavier_normal_(self.fc.weight, gain=gain)
        init.normal_(self.mu.data, 0, 0.1)
        init.constant_(self.inv_sigma.data, 1)
        if self.bias is not None:
            init.zeros_(self.bias.data)
    
    def forward(self, g, h, pseudo, snorm_n):
        h_in = h # for residual connection
        
        g = g.local_var()
        g.ndata['h'] = self.fc(h).view(-1, self.kernel, self.out_dim)
        E = g.number_of_edges()
        
        # compute gaussian weight
        gaussian = -0.5 * ((pseudo.view(E, 1, self.dim) -
                            self.mu.view(1, self.kernel, self.dim)) ** 2)
        gaussian = gaussian * (self.inv_sigma.view(1, self.kernel, self.dim) ** 2)
        gaussian = torch.exp(gaussian.sum(dim=-1, keepdim=True)) # (E, K, 1)
        g.edata['w'] = gaussian
        g.update_all(fn.u_mul_e('h', 'w', 'm'), self._reducer('m', 'h'))
        h = g.ndata['h'].sum(1)
        
        if self.graph_norm:
            h = h* snorm_n # normalize activation w.r.t. graph size
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization  
        
        h = F.relu(h) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
        
        if self.bias is not None:
            h = h + self.bias
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h
