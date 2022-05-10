#!/bin/bash


############
# Usage
############

# bash script_main_xxx.sh



############
# GNNs
############

# GatedGCN



##########################
# GraphTheoryProp - 4 RUNS
##########################

seed0=41
seed1=95
seed2=12
seed3=35
code=main_GraphTheoryProp_multitask.py 
dataset=GraphTheoryProp
tmux new -s benchmark_GraphTheoryProp -d
tmux send-keys "source activate benchmark_gnn" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/GraphTheoryProp_multitask_GatedGCN_PE_GraphTheoryProp_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/GraphTheoryProp_multitask_GatedGCN_PE_GraphTheoryProp_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/GraphTheoryProp_multitask_GatedGCN_PE_GraphTheoryProp_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/GraphTheoryProp_multitask_GatedGCN_PE_GraphTheoryProp_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/GraphTheoryProp_multitask_GatedGCN_GraphTheoryProp_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/GraphTheoryProp_multitask_GatedGCN_GraphTheoryProp_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/GraphTheoryProp_multitask_GatedGCN_GraphTheoryProp_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/GraphTheoryProp_multitask_GatedGCN_GraphTheoryProp_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/GraphTheoryProp_multitask_GIN_PE_GraphTheoryProp_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/GraphTheoryProp_multitask_GIN_PE_GraphTheoryProp_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/GraphTheoryProp_multitask_GIN_PE_GraphTheoryProp_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/GraphTheoryProp_multitask_GIN_PE_GraphTheoryProp_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/GraphTheoryProp_multitask_GIN_GraphTheoryProp_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/GraphTheoryProp_multitask_GIN_GraphTheoryProp_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/GraphTheoryProp_multitask_GIN_GraphTheoryProp_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/GraphTheoryProp_multitask_GIN_GraphTheoryProp_100k.json' &
wait" C-m
tmux send-keys "tmux kill-session -t benchmark_GraphTheoryProp" C-m











