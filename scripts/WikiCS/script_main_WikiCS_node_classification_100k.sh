#!/bin/bash


############
# Usage
############

# bash script_main_WikiCS_node_classification_100k.sh



############
# GNNs
############

#MLP
#GCN
#GraphSage
#GAT
#MoNet
#GIN



############
# WikiCS - 4 RUNS  
############

seed0=41
seed1=95
seed2=12
seed3=35
code=main_WikiCS_node_classification.py 
tmux new -s benchmark -d
tmux send-keys "source activate benchmark_gnn" C-m
dataset=WikiCS
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/WikiCS_node_classification_MLP_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/WikiCS_node_classification_MLP_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/WikiCS_node_classification_MLP_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/WikiCS_node_classification_MLP_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/WikiCS_node_classification_GCN_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/WikiCS_node_classification_GCN_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/WikiCS_node_classification_GCN_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/WikiCS_node_classification_GCN_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/WikiCS_node_classification_GraphSage_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/WikiCS_node_classification_GraphSage_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/WikiCS_node_classification_GraphSage_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/WikiCS_node_classification_GraphSage_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/WikiCS_node_classification_GAT_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/WikiCS_node_classification_GAT_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/WikiCS_node_classification_GAT_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/WikiCS_node_classification_GAT_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/WikiCS_node_classification_MoNet_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/WikiCS_node_classification_MoNet_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/WikiCS_node_classification_MoNet_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/WikiCS_node_classification_MoNet_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/WikiCS_node_classification_MoNet_PE_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/WikiCS_node_classification_MoNet_PE_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/WikiCS_node_classification_MoNet_PE_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/WikiCS_node_classification_MoNet_PE_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/WikiCS_node_classification_GIN_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/WikiCS_node_classification_GIN_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/WikiCS_node_classification_GIN_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/WikiCS_node_classification_GIN_100k.json' &
wait" C-m
tmux send-keys "tmux kill-session -t benchmark" C-m









