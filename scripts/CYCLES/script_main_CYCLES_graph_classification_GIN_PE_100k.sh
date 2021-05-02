#!/bin/bash


############
# Usage
############

# bash script_main_CYCLES_graph_classification_CYCLES_100k.sh



############
# GNNs
############

# GatedGCN



############
# CYCLES - 4 RUNS
############

seed0=41
seed1=95
seed2=12
seed3=35
code=main_CYCLES_graph_classification.py 
dataset=CYCLES
tmux new -s benchmark_CYCLES -d
tmux send-keys "source activate benchmark_gnn" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --num_train_data 200 --config 'configs/CYCLES_graph_classification_GIN_PE_CYCLES_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --num_train_data 200 --config 'configs/CYCLES_graph_classification_GIN_PE_CYCLES_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --num_train_data 200 --config 'configs/CYCLES_graph_classification_GIN_PE_CYCLES_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --num_train_data 200 --config 'configs/CYCLES_graph_classification_GIN_PE_CYCLES_100k.json' &
wait" C-m

tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --num_train_data 500 --config 'configs/CYCLES_graph_classification_GIN_PE_CYCLES_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --num_train_data 500 --config 'configs/CYCLES_graph_classification_GIN_PE_CYCLES_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --num_train_data 500 --config 'configs/CYCLES_graph_classification_GIN_PE_CYCLES_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --num_train_data 500 --config 'configs/CYCLES_graph_classification_GIN_PE_CYCLES_100k.json' &
wait" C-m

tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --num_train_data 1000 --config 'configs/CYCLES_graph_classification_GIN_PE_CYCLES_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --num_train_data 1000 --config 'configs/CYCLES_graph_classification_GIN_PE_CYCLES_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --num_train_data 1000 --config 'configs/CYCLES_graph_classification_GIN_PE_CYCLES_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --num_train_data 1000 --config 'configs/CYCLES_graph_classification_GIN_PE_CYCLES_100k.json' &
wait" C-m

tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --num_train_data 5000 --config 'configs/CYCLES_graph_classification_GIN_PE_CYCLES_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --num_train_data 5000 --config 'configs/CYCLES_graph_classification_GIN_PE_CYCLES_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --num_train_data 5000 --config 'configs/CYCLES_graph_classification_GIN_PE_CYCLES_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --num_train_data 5000 --config 'configs/CYCLES_graph_classification_GIN_PE_CYCLES_100k.json' &
wait" C-m
tmux send-keys "tmux kill-session -t benchmark_CYCLES" C-m











