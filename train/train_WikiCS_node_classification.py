"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl

from train.metrics import accuracy_WikiCS as accuracy


def train_epoch(model, optimizer, device, graph, node_feat, edge_feat, train_mask, labels, epoch):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    
    try:
        pos_enc = graph.ndata['pos_enc'].to(device)
        sign_flip = torch.rand(pos_enc.size(1)).to(device)
        sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
        pos_enc = pos_enc * sign_flip.unsqueeze(0)
        logits = model.forward(graph, node_feat, edge_feat, pos_enc)
    except:
        logits = model(graph, node_feat, edge_feat)
    loss = model.loss(logits[train_mask], labels[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch_loss = loss.detach().item()
    epoch_train_acc = accuracy(logits[train_mask], labels[train_mask])
    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network(model, optimizer, device, graph, node_feat, edge_feat, mask, labels, epoch):

    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    
    with torch.no_grad():
        try:
            pos_enc = graph.ndata['pos_enc'].to(device)
            logits = model.forward(graph, node_feat, edge_feat, pos_enc)
        except:
            logits = model.forward(graph, node_feat, edge_feat)
        loss = model.loss(logits[mask], labels[mask])
        epoch_test_loss = loss.detach().item()
        epoch_test_acc = accuracy(logits[mask], labels[mask])

    return epoch_test_loss, epoch_test_acc