"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math

from train.metrics import MAE

"""
    For GCNs
"""
def train_epoch_sparse(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()
        try:
            batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
            sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)
        except:
            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(batch_scores, batch_targets)
        nb_data += batch_targets.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    
    return epoch_loss, epoch_train_mae, optimizer

def evaluate_network_sparse(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_targets = batch_targets.to(device)
            try:
                batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)
            except:
                batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        
    return epoch_test_loss, epoch_test_mae




"""
    For WL-GNNs
"""
def train_epoch_dense(model, optimizer, device, data_loader, epoch, batch_size):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0
    gpu_mem = 0
    optimizer.zero_grad()
    for iter, (x_no_edge_feat, x_with_edge_feat, targets) in enumerate(data_loader):
        if x_no_edge_feat is not None:
            x_no_edge_feat = x_no_edge_feat.to(device)
        if x_with_edge_feat is not None:
            x_with_edge_feat = x_with_edge_feat.to(device)
        targets = targets.to(device)
        
        scores = model.forward(x_no_edge_feat, x_with_edge_feat)
        loss = model.loss(scores, targets)
        loss.backward()
        
        if not (iter%batch_size):
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(scores, targets)
        nb_data += targets.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    
    return epoch_loss, epoch_train_mae, optimizer

def evaluate_network_dense(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (x_no_edge_feat, x_with_edge_feat, targets) in enumerate(data_loader):
            if x_no_edge_feat is not None:
                x_no_edge_feat = x_no_edge_feat.to(device)
            if x_with_edge_feat is not None:
                x_with_edge_feat = x_with_edge_feat.to(device)
            targets = targets.to(device)
            
            scores = model.forward(x_no_edge_feat, x_with_edge_feat)
            loss = model.loss(scores, targets)
            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(scores, targets)
            nb_data += targets.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        
    return epoch_test_loss, epoch_test_mae
