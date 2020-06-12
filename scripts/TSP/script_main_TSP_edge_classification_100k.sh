#!/bin/bash

# check : 
# bash script.sh
# tmux attach -t script_tsp
# tmux detach
# pkill python

# bash script_main_TSP_edge_classification_100k.sh




############
# GNNs
############

#GatedGCN
#GCN
#GraphSage
#MLP
#GIN
#MoNet
#GAT





############
# TSP - 4 RUNS  
############

seed0=41
seed1=42
seed2=9
seed3=23
code=main_TSP_edge_classification.py 
tmux new -s benchmark_TSP_edge_classification -d
tmux send-keys "source activate benchmark_gnn" C-m
dataset=TSP
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_MLP_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_MLP_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_MLP_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_MLP_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_GCN_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_GCN_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_GCN_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_GCN_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_GIN_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_GIN_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_GIN_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_GIN_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_GraphSage_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_GraphSage_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_GraphSage_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_GraphSage_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_GatedGCN_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_GatedGCN_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_GatedGCN_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_GatedGCN_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_GatedGCN_100k.json' --edge_feat True &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_GatedGCN_100k.json' --edge_feat True &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_GatedGCN_100k.json' --edge_feat True &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_GatedGCN_100k.json' --edge_feat True &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_GAT_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_GAT_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_GAT_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_GAT_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_MoNet_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_MoNet_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_MoNet_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_MoNet_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_3WLGNN_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_3WLGNN_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_3WLGNN_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_3WLGNN_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_RingGNN_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_RingGNN_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_RingGNN_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_RingGNN_100k.json' &
wait" C-m
tmux send-keys "tmux kill-session -t benchmark_TSP_edge_classification" C-m
