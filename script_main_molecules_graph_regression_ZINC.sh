#!/bin/bash

# check : 
# bash script.sh
# tmux attach -t script_mol_opt
# tmux detach
# pkill python

# bash script_main_molecules_graph_regression_ZINC.sh


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
#DiffPool





############
# ZINC - 4 RUNS
############

seed0=41
seed1=95
seed2=12
seed3=35
code=main_molecules_graph_regression.py 
dataset=ZINC
tmux new -s benchmark_molecules_graph_regression -d
tmux send-keys "source activate benchmark_gnn" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_MLP_ZINC.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_graph_regression_MLP_ZINC.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_graph_regression_MLP_ZINC.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_graph_regression_MLP_ZINC.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_MLP_GATED_ZINC.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_graph_regression_MLP_GATED_ZINC.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_graph_regression_MLP_GATED_ZINC.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_graph_regression_MLP_GATED_ZINC.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_GIN_ZINC.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_graph_regression_GIN_ZINC.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_graph_regression_GIN_ZINC.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_graph_regression_GIN_ZINC.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_GCN_ZINC.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_graph_regression_GCN_ZINC.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_graph_regression_GCN_ZINC.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_graph_regression_GCN_ZINC.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_GraphSage_ZINC.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_graph_regression_GraphSage_ZINC.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_graph_regression_GraphSage_ZINC.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_graph_regression_GraphSage_ZINC.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_GatedGCN_ZINC.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_graph_regression_GatedGCN_ZINC.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_graph_regression_GatedGCN_ZINC.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_graph_regression_GatedGCN_ZINC.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_GAT_ZINC.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_graph_regression_GAT_ZINC.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_graph_regression_GAT_ZINC.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_graph_regression_GAT_ZINC.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_DiffPool_ZINC.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_graph_regression_DiffPool_ZINC.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_graph_regression_DiffPool_ZINC.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_graph_regression_DiffPool_ZINC.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_MoNet_ZINC.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/molecules_graph_regression_MoNet_ZINC.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/molecules_graph_regression_MoNet_ZINC.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/molecules_graph_regression_MoNet_ZINC.json' &
wait" C-m
tmux send-keys "tmux kill-session -t benchmark_molecules_graph_regression" C-m











