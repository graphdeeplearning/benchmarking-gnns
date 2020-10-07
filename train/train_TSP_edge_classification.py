"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl

from train.metrics import binary_f1_score

"""
    For GCNs
"""
def train_epoch_sparse(model, optimizer, device, data_loader, epoch):

    model.train()
    epoch_loss = 0
    epoch_train_f1 = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        
        batch_scores = model.forward(batch_graphs, batch_x, batch_e)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_f1 += binary_f1_score(batch_scores, batch_labels)
    epoch_loss /= (iter + 1)
    epoch_train_f1 /= (iter + 1)
    
    return epoch_loss, epoch_train_f1, optimizer


def evaluate_network_sparse(model, device, data_loader, epoch):
    
    model.eval()
    epoch_test_loss = 0
    epoch_test_f1 = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)

            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            loss = model.loss(batch_scores, batch_labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_f1 += binary_f1_score(batch_scores, batch_labels)
        epoch_test_loss /= (iter + 1)
        epoch_test_f1 /= (iter + 1)
        
    return epoch_test_loss, epoch_test_f1





"""
    For WL-GNNs
"""
def train_epoch_dense(model, optimizer, device, data_loader, epoch, batch_size):

    model.train()
    epoch_loss = 0
    epoch_train_f1 = 0
    nb_data = 0
    gpu_mem = 0
    optimizer.zero_grad()
    for iter, (x_no_edge_feat, x_with_edge_feat, labels, edge_list) in enumerate(data_loader):
        if x_no_edge_feat is not None:
            x_no_edge_feat = x_no_edge_feat.to(device)
        if x_with_edge_feat is not None:
            x_with_edge_feat = x_with_edge_feat.to(device)
        labels = labels.to(device)
        edge_list = edge_list[0].to(device), edge_list[1].to(device)
        
        scores = model.forward(x_no_edge_feat, x_with_edge_feat, edge_list)
        loss = model.loss(scores, labels)
        loss.backward()
        
        if not (iter%batch_size):
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss += loss.detach().item()
        epoch_train_f1 += binary_f1_score(scores, labels)
    epoch_loss /= (iter + 1)
    epoch_train_f1 /= (iter + 1)
    
    return epoch_loss, epoch_train_f1, optimizer


def evaluate_network_dense(model, device, data_loader, epoch):
    
    model.eval()
    epoch_test_loss = 0
    epoch_test_f1 = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (x_no_edge_feat, x_with_edge_feat, labels, edge_list) in enumerate(data_loader):
            if x_no_edge_feat is not None:
                x_no_edge_feat = x_no_edge_feat.to(device)
            if x_with_edge_feat is not None:
                x_with_edge_feat = x_with_edge_feat.to(device)
            labels = labels.to(device)
            edge_list = edge_list[0].to(device), edge_list[1].to(device)

            scores = model.forward(x_no_edge_feat, x_with_edge_feat, edge_list)
            loss = model.loss(scores, labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_f1 += binary_f1_score(scores, labels)
        epoch_test_loss /= (iter + 1)
        epoch_test_f1 /= (iter + 1)
        
    return epoch_test_loss, epoch_test_f1
