"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl
import numpy as np

from tqdm import tqdm

"""
    For GCNs
"""
def train_epoch_sparse(model, optimizer, device, graph, train_edges, batch_size, epoch, monet_pseudo=None):

    model.train()
    
    train_edges = train_edges.to(device)
    
    total_loss = total_examples = 0
    for perm in tqdm(DataLoader(range(train_edges.size(0)), batch_size, shuffle=True)):
        
        optimizer.zero_grad()
        
        graph = graph.to(device)
        x = graph.ndata['feat'].to(device)
        e = graph.edata['feat'].to(device).float()
        
        if monet_pseudo is not None:
            # Assign e as pre-computed pesudo edges for MoNet
            e = monet_pseudo.to(device)
        
        # Compute node embeddings
        try:
            x_pos_enc = graph.ndata['pos_enc'].to(device)
            sign_flip = torch.rand(x_pos_enc.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            x_pos_enc = x_pos_enc * sign_flip.unsqueeze(0)
            h = model(graph, x, e, x_pos_enc) 
        except:
            h = model(graph, x, e)
        
        # Positive samples
        edge = train_edges[perm].t()
        pos_out = model.edge_predictor( h[edge[0]], h[edge[1]] )
        
        # Just do some trivial random sampling
        edge = torch.randint(0, x.size(0), edge.size(), dtype=torch.long, device=x.device)
        neg_out = model.edge_predictor( h[edge[0]], h[edge[1]] )
        
        loss = model.loss(pos_out, neg_out)

        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.detach().item() * num_examples
        total_examples += num_examples

    return total_loss/total_examples, optimizer


def evaluate_network_sparse(model, device, graph, pos_train_edges, 
                     pos_valid_edges, neg_valid_edges, 
                     pos_test_edges, neg_test_edges, 
                     evaluator, batch_size, epoch, monet_pseudo=None):
    
    model.eval()
    with torch.no_grad():

        graph = graph.to(device)
        x = graph.ndata['feat'].to(device)
        e = graph.edata['feat'].to(device).float()
        
        if monet_pseudo is not None:
            # Assign e as pre-computed pesudo edges for MoNet
            e = monet_pseudo.to(device)

        # Compute node embeddings
        try:
            x_pos_enc = graph.ndata['pos_enc'].to(device)
            h = model(graph, x, e, x_pos_enc) 
        except:
            h = model(graph, x, e)

        pos_train_edges = pos_train_edges.to(device)
        pos_valid_edges = pos_valid_edges.to(device)
        neg_valid_edges = neg_valid_edges.to(device)
        pos_test_edges = pos_test_edges.to(device)
        neg_test_edges = neg_test_edges.to(device)

        pos_train_preds = []
        for perm in DataLoader(range(pos_train_edges.size(0)), batch_size):
            edge = pos_train_edges[perm].t()
            pos_train_preds += [model.edge_predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_train_pred = torch.cat(pos_train_preds, dim=0)

        pos_valid_preds = []
        for perm in DataLoader(range(pos_valid_edges.size(0)), batch_size):
            edge = pos_valid_edges[perm].t()
            pos_valid_preds += [model.edge_predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

        neg_valid_preds = []
        for perm in DataLoader(range(pos_valid_edges.size(0)), batch_size):
            edge = neg_valid_edges[perm].t()
            neg_valid_preds += [model.edge_predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

        pos_test_preds = []
        for perm in DataLoader(range(pos_test_edges.size(0)), batch_size):
            edge = pos_test_edges[perm].t()
            pos_test_preds += [model.edge_predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_test_pred = torch.cat(pos_test_preds, dim=0)

        neg_test_preds = []
        for perm in DataLoader(range(pos_test_edges.size(0)), batch_size):
            edge = neg_test_edges[perm].t()
            neg_test_preds += [model.edge_predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)

    
    train_hits = []
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits.append(
            evaluator.eval({
                'y_pred_pos': pos_train_pred,
                'y_pred_neg': neg_valid_pred,  # negative samples for valid == training
            })[f'hits@{K}']
        )
    
    valid_hits = []
    for K in [10, 50, 100]:
        evaluator.K = K
        valid_hits.append(
            evaluator.eval({
                'y_pred_pos': pos_valid_pred,
                'y_pred_neg': neg_valid_pred,
            })[f'hits@{K}']
        )
    
    test_hits = []
    for K in [10, 50, 100]:
        evaluator.K = K
        test_hits.append(
            evaluator.eval({
                'y_pred_pos': pos_test_pred,
                'y_pred_neg': neg_test_pred,
            })[f'hits@{K}']
        )

    return train_hits, valid_hits, test_hits
