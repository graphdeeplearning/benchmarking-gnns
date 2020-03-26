"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl

from train.metrics import accuracy_CITATION_GRAPH as accuracy


def train_epoch(model, optimizer, device, graph, nfeat, efeat, norm_n, norm_e, train_mask, labels, epoch):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0

    #logits = model.forward(graph, nfeat, efeat, norm_n, norm_e)
    logits = model(graph, nfeat, efeat, norm_n, norm_e)
    loss = model.loss(logits[train_mask], labels[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch_loss = loss.detach().item()
    epoch_train_acc = accuracy(logits[train_mask], labels[train_mask])
    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network(model, optimizer, device, graph, nfeat, efeat, norm_n, norm_e, mask, labels, epoch):
    
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        logits = model.forward(graph, nfeat, efeat, norm_n, norm_e)
        loss = model.loss(logits[mask], labels[mask])
        epoch_test_loss = loss.detach().item()
        epoch_test_acc = accuracy(logits[mask], labels[mask])

    return epoch_test_loss, epoch_test_acc
