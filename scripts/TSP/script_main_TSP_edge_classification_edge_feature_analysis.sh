#!/bin/bash

# check : 
# bash script.sh
# tmux attach -t script_tsp
# tmux detach
# pkill python

# bash script_main_TSP_edge_classification_edge_feature_analysis.sh




############
# GNNs
############

#GatedGCN
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
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_GatedGCN_isotropic.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_GatedGCN_isotropic.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_GatedGCN_isotropic.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_GatedGCN_isotropic.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_GatedGCN_edgefeat.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_GatedGCN_edgefeat.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_GatedGCN_edgefeat.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_GatedGCN_edgefeat.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_GatedGCN_edgereprfeat.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_GatedGCN_edgereprfeat.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_GatedGCN_edgereprfeat.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_GatedGCN_edgereprfeat.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_GatedGCN_edgereprfeat.json' --edge_feat True &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_GatedGCN_edgereprfeat.json' --edge_feat True &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_GatedGCN_edgereprfeat.json' --edge_feat True &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_GatedGCN_edgereprfeat.json' --edge_feat True &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_GAT_isotropic.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_GAT_isotropic.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_GAT_isotropic.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_GAT_isotropic.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_GAT_edgefeat.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_GAT_edgefeat.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_GAT_edgefeat.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_GAT_edgefeat.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_GAT_edgereprfeat.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_GAT_edgereprfeat.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_GAT_edgereprfeat.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_GAT_edgereprfeat.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_GAT_edgereprfeat.json' --edge_feat True &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/TSP_edge_classification_GAT_edgereprfeat.json' --edge_feat True &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/TSP_edge_classification_GAT_edgereprfeat.json' --edge_feat True &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/TSP_edge_classification_GAT_edgereprfeat.json' --edge_feat True &
wait" C-m
tmux send-keys "tmux kill-session -t benchmark_TSP_edge_classification" C-m
