#!/bin/bash

# bash script_one_code_to_rull_them_all.sh




tmux new -s benchmark_script -d
tmux send-keys "source activate benchmark_gnn" C-m
seed0=41


############
# TU
############

code=main_TUs_graph_classification.py 
dataset=ENZYMES
tmux send-keys "
python $code --dataset $dataset --gpu_id 0  --seed $seed0 --config 'configs/TUs_graph_classification_MLP_ENZYMES.json' &
python $code --dataset $dataset --gpu_id 0  --seed $seed0 --config 'configs/TUs_graph_classification_MLP_GATED_ENZYMES.json' &
python $code --dataset $dataset --gpu_id 1  --seed $seed0 --config 'configs/TUs_graph_classification_GIN_ENZYMES.json' &
python $code --dataset $dataset --gpu_id 2  --seed $seed0 --config 'configs/TUs_graph_classification_GCN_ENZYMES.json' &
python $code --dataset $dataset --gpu_id 3  --seed $seed0 --config 'configs/TUs_graph_classification_GraphSage_ENZYMES.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0  --seed $seed0 --config 'configs/TUs_graph_classification_GatedGCN_ENZYMES.json' &
python $code --dataset $dataset --gpu_id 1  --seed $seed0 --config 'configs/TUs_graph_classification_GAT_ENZYMES.json' &
python $code --dataset $dataset --gpu_id 2  --seed $seed0 --config 'configs/TUs_graph_classification_DiffPool_ENZYMES.json' &
python $code --dataset $dataset --gpu_id 3  --seed $seed0 --config 'configs/TUs_graph_classification_MoNet_ENZYMES.json' &
wait" C-m

dataset=DD
tmux send-keys "
python $code --dataset $dataset --gpu_id 0  --seed $seed0 --config 'configs/TUs_graph_classification_MLP_DD.json' &
python $code --dataset $dataset --gpu_id 0  --seed $seed0 --config 'configs/TUs_graph_classification_MLP_GATED_DD.json' &
python $code --dataset $dataset --gpu_id 1  --seed $seed0 --config 'configs/TUs_graph_classification_GIN_DD.json' &
python $code --dataset $dataset --gpu_id 2  --seed $seed0 --config 'configs/TUs_graph_classification_GCN_DD.json' &
python $code --dataset $dataset --gpu_id 3  --seed $seed0 --config 'configs/TUs_graph_classification_GraphSage_DD.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0  --seed $seed0 --config 'configs/TUs_graph_classification_GatedGCN_DD.json' &
python $code --dataset $dataset --gpu_id 1  --seed $seed0 --config 'configs/TUs_graph_classification_GAT_DD.json' &
python $code --dataset $dataset --gpu_id 2  --seed $seed0 --config 'configs/TUs_graph_classification_DiffPool_DD.json' &
python $code --dataset $dataset --gpu_id 3  --seed $seed0 --config 'configs/TUs_graph_classification_MoNet_DD.json' &
wait" C-m

dataset=PROTEINS_full
tmux send-keys "
python $code --dataset $dataset --gpu_id 0  --seed $seed0 --config 'configs/TUs_graph_classification_MLP_PROTEINS_full.json' &
python $code --dataset $dataset --gpu_id 0  --seed $seed0 --config 'configs/TUs_graph_classification_MLP_GATED_PROTEINS_full.json' &
python $code --dataset $dataset --gpu_id 1  --seed $seed0 --config 'configs/TUs_graph_classification_GIN_PROTEINS_full.json' &
python $code --dataset $dataset --gpu_id 2  --seed $seed0 --config 'configs/TUs_graph_classification_GCN_PROTEINS_full.json' &
python $code --dataset $dataset --gpu_id 3  --seed $seed0 --config 'configs/TUs_graph_classification_GraphSage_PROTEINS_full.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0  --seed $seed0 --config 'configs/TUs_graph_classification_GatedGCN_PROTEINS_full.json' &
python $code --dataset $dataset --gpu_id 1  --seed $seed0 --config 'configs/TUs_graph_classification_GAT_PROTEINS_full.json' &
python $code --dataset $dataset --gpu_id 2  --seed $seed0 --config 'configs/TUs_graph_classification_DiffPool_PROTEINS_full.json' &
python $code --dataset $dataset --gpu_id 3  --seed $seed0 --config 'configs/TUs_graph_classification_MoNet_PROTEINS_full.json' &
wait" C-m


############
# ZINC
############

code=main_molecules_graph_regression.py 
dataset=ZINC
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_MLP_ZINC.json' &
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_MLP_GATED_ZINC.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/molecules_graph_regression_GIN_ZINC.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/molecules_graph_regression_GCN_ZINC.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed0 --config 'configs/molecules_graph_regression_GraphSage_ZINC.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_GatedGCN_ZINC.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/molecules_graph_regression_GAT_ZINC.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/molecules_graph_regression_DiffPool_ZINC.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed0 --config 'configs/molecules_graph_regression_MoNet_ZINC.json' &
wait" C-m


############
# MNIST and CIFAR10
############

code=main_superpixels_graph_classification.py 
dataset=MNIST
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/superpixels_graph_classification_MLP_MNIST.json' &
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/superpixels_graph_classification_MLP_GATED_MNIST.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/superpixels_graph_classification_GIN_MNIST.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/superpixels_graph_classification_GCN_MNIST.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed0 --config 'configs/superpixels_graph_classification_GraphSage_MNIST.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/superpixels_graph_classification_GatedGCN_MNIST.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/superpixels_graph_classification_GAT_MNIST.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/superpixels_graph_classification_DiffPool_MNIST.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed0 --config 'configs/superpixels_graph_classification_MoNet_MNIST.json' &
wait" C-m

dataset=CIFAR10
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/superpixels_graph_classification_MLP_CIFAR10.json' &
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/superpixels_graph_classification_MLP_GATED_CIFAR10.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/superpixels_graph_classification_GIN_CIFAR10.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/superpixels_graph_classification_GCN_CIFAR10.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed0 --config 'configs/superpixels_graph_classification_GraphSage_CIFAR10.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/superpixels_graph_classification_GatedGCN_CIFAR10.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/superpixels_graph_classification_GAT_CIFAR10.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/superpixels_graph_classification_DiffPool_CIFAR10.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed0 --config 'configs/superpixels_graph_classification_MoNet_CIFAR10.json' &
wait" C-m


############
# PATTERN and CLUSTER 
############

code=main_SBMs_node_classification.py 
dataset=SBM_PATTERN
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_PATTERN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_PATTERN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_PATTERN.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_PATTERN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_PATTERN.json' &
wait" C-m

dataset=SBM_CLUSTER
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_CLUSTER.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER.json' &
wait" C-m


############
# TSP 
############

code=main_TSP_edge_classification.py 
dataset=TSP
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_MLP.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/TSP_edge_classification_MLP_GATED.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/TSP_edge_classification_GIN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed0 --config 'configs/TSP_edge_classification_GCN.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_GraphSage.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/TSP_edge_classification_GatedGCN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/TSP_edge_classification_GAT.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed0 --config 'configs/TSP_edge_classification_MoNet.json' &
wait" C-m


tmux send-keys "tmux kill-session -t benchmark_script" C-m

















