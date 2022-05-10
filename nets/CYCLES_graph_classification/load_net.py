"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.CYCLES_graph_classification.gated_gcn_net import GatedGCNNet
from nets.CYCLES_graph_classification.gin_net import GINNet

def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GIN(net_params):
    return GINNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'GIN': GIN
    }
        
    return models[MODEL_NAME](net_params)