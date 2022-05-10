#!/bin/bash


############
# Usage
############

# bash script_main_molecules_graph_regression_ZINC_100k.sh



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
# ZINC - 4 RUNS
############

seed0=41
seed1=95
seed2=12
seed3=35
code=main_molecules_graph_regression.py 
dataset=ZINC
tmux new -s benchmark -d
tmux send-keys "source activate benchmark_gnn" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_MLP_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_graph_regression_MLP_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_graph_regression_MLP_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_graph_regression_MLP_ZINC_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_GCN_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_graph_regression_GCN_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_graph_regression_GCN_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_graph_regression_GCN_ZINC_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_GraphSage_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_graph_regression_GraphSage_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_graph_regression_GraphSage_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_graph_regression_GraphSage_ZINC_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_GatedGCN_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_graph_regression_GatedGCN_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_graph_regression_GatedGCN_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_graph_regression_GatedGCN_ZINC_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_GAT_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_graph_regression_GAT_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_graph_regression_GAT_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_graph_regression_GAT_ZINC_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_MoNet_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_graph_regression_MoNet_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_graph_regression_MoNet_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_graph_regression_MoNet_ZINC_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_GIN_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_graph_regression_GIN_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_graph_regression_GIN_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_graph_regression_GIN_ZINC_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_3WLGNN_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_graph_regression_3WLGNN_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_graph_regression_3WLGNN_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_graph_regression_3WLGNN_ZINC_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_RingGNN_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_graph_regression_RingGNN_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_graph_regression_RingGNN_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_graph_regression_RingGNN_ZINC_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --edge_feat True --config 'configs/molecules_graph_regression_GatedGCN_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --edge_feat True --config 'configs/molecules_graph_regression_GatedGCN_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --edge_feat True --config 'configs/molecules_graph_regression_GatedGCN_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --edge_feat True --config 'configs/molecules_graph_regression_GatedGCN_ZINC_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --edge_feat True --config 'configs/molecules_graph_regression_3WLGNN_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --edge_feat True --config 'configs/molecules_graph_regression_3WLGNN_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --edge_feat True --config 'configs/molecules_graph_regression_3WLGNN_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --edge_feat True --config 'configs/molecules_graph_regression_3WLGNN_ZINC_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --edge_feat True --config 'configs/molecules_graph_regression_RingGNN_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --edge_feat True --config 'configs/molecules_graph_regression_RingGNN_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --edge_feat True --config 'configs/molecules_graph_regression_RingGNN_ZINC_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --edge_feat True --config 'configs/molecules_graph_regression_RingGNN_ZINC_100k.json' &
wait" C-m
tmux send-keys "tmux kill-session -t benchmark" C-m











