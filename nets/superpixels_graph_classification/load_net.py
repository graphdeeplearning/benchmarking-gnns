"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.superpixels_graph_classification.gated_gcn_net import GatedGCNNet
from nets.superpixels_graph_classification.gcn_net import GCNNet
from nets.superpixels_graph_classification.gat_net import GATNet
from nets.superpixels_graph_classification.graphsage_net import GraphSageNet
from nets.superpixels_graph_classification.gin_net import GINNet
from nets.superpixels_graph_classification.mo_net import MoNet as MoNet_
from nets.superpixels_graph_classification.diffpool_net import DiffPoolNet
from nets.superpixels_graph_classification.mlp_net import MLPNet


def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def GraphSage(net_params):
    return GraphSageNet(net_params)

def GIN(net_params):
    return GINNet(net_params)

def MoNet(net_params):
    return MoNet_(net_params)

def DiffPool(net_params):
    return DiffPoolNet(net_params)

def MLP(net_params):
    return MLPNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'GCN': GCN,
        'GAT': GAT,
        'GraphSage': GraphSage,
        'GIN': GIN,
        'MoNet': MoNet,
        'DiffPool': DiffPool,
        'MLP': MLP
    }
        
    return models[MODEL_NAME](net_params)