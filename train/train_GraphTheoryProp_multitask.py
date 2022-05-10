"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import numpy as np

"""
    For GCNs
"""
def train_epoch_sparse(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_MSE = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_node_labels, batch_graph_labels) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_node_labels = batch_node_labels.to(device)
        batch_graph_labels = batch_graph_labels.to(device)
        batch_labels = batch_node_labels, batch_graph_labels
        optimizer.zero_grad()
        try:
            batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
            sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)
        except:
            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
        loss, specific_loss = model.loss(batch_scores, batch_labels) 
        loss.backward()
        optimizer.step()
        loss_ = loss.detach().item()
        epoch_loss += loss_
        epoch_train_MSE += loss_
        
    epoch_loss /= (iter + 1)
    epoch_train_MSE /= (iter + 1)
    
    return epoch_loss, np.log10(epoch_train_MSE), optimizer

def evaluate_network_sparse(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_MSE = 0
    specific_test_MSE = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_node_labels, batch_graph_labels) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_node_labels = batch_node_labels.to(device)
            batch_graph_labels = batch_graph_labels.to(device)
            batch_labels = batch_node_labels, batch_graph_labels
            try:
                batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)
            except:
                batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            loss, specific_loss = model.loss(batch_scores, batch_labels) 
            loss_ = loss.detach().item()
            specific_loss_ = specific_loss.detach() #3
            epoch_test_loss += loss_
            epoch_test_MSE += loss_
            specific_test_MSE += specific_loss_
        epoch_test_loss /= (iter + 1)
        epoch_test_MSE /= (iter + 1)
        specific_test_MSE /= (iter + 1)
        
    return epoch_test_loss, np.log10(epoch_test_MSE), np.log10(specific_test_MSE.cpu())


