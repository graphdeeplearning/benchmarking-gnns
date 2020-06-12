#!/bin/bash


############
# Usage
############

# bash script_main_SBMs_node_classification_PATTERN_100k.sh



############
# GNNs
############

#MLP
#GCN
#GraphSage
#GatedGCN
#GAT
#MoNet
#GIN
#3WLGNN
#RingGNN



############
# SBM_PATTERN - 4 RUNS 
############

seed0=41
seed1=95
seed2=12
seed3=35
code=main_SBMs_node_classification.py 
tmux new -s benchmark -d
tmux send-keys "source activate benchmark_gnn" C-m
dataset=SBM_PATTERN
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_PATTERN_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GCN_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GCN_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_PATTERN_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_PATTERN_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_PATTERN_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_3WLGNN_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_3WLGNN_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_3WLGNN_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_3WLGNN_PATTERN_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_RingGNN_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_RingGNN_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_RingGNN_PATTERN_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_RingGNN_PATTERN_100k.json' &
wait" C-m
tmux send-keys "tmux kill-session -t benchmark" C-m









