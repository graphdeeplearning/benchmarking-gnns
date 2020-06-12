"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.SBMs_node_classification.gated_gcn_net import GatedGCNNet
from nets.SBMs_node_classification.gcn_net import GCNNet
from nets.SBMs_node_classification.gat_net import GATNet
from nets.SBMs_node_classification.graphsage_net import GraphSageNet
from nets.SBMs_node_classification.mlp_net import MLPNet
from nets.SBMs_node_classification.gin_net import GINNet
from nets.SBMs_node_classification.mo_net import MoNet as MoNet_
from nets.SBMs_node_classification.ring_gnn_net import RingGNNNet
from nets.SBMs_node_classification.three_wl_gnn_net import ThreeWLGNNNet


def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def GraphSage(net_params):
    return GraphSageNet(net_params)

def MLP(net_params):
    return MLPNet(net_params)

def GIN(net_params):
    return GINNet(net_params)

def MoNet(net_params):
    return MoNet_(net_params)

def RingGNN(net_params):
    return RingGNNNet(net_params)

def ThreeWLGNN(net_params):
    return ThreeWLGNNNet(net_params)


def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'GCN': GCN,
        'GAT': GAT,
        'GraphSage': GraphSage,
        'MLP': MLP,
        'GIN': GIN,
        'MoNet': MoNet,
        'RingGNN': RingGNN,
        '3WLGNN': ThreeWLGNN
    }
        
    return models[MODEL_NAME](net_params)