#!/bin/bash


############
# Usage
############

# bash script_main_superpixels_graph_classification_CIFAR10_500k.sh



############
# GNNs
############


#3WLGNN
#RingGNN



############
# CIFAR10 - 4 RUNS  
############

seed0=41
seed1=95
seed2=12
seed3=35
code=main_superpixels_graph_classification.py 
tmux new -s benchmark_CIFAR10 -d
tmux send-keys "source activate benchmark_gnn" C-m
dataset=CIFAR10
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --hidden_dim 180 --config 'configs/superpixels_graph_classification_3WLGNN_CIFAR10_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --hidden_dim 180 --config 'configs/superpixels_graph_classification_3WLGNN_CIFAR10_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --hidden_dim 180 --config 'configs/superpixels_graph_classification_3WLGNN_CIFAR10_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --hidden_dim 180 --config 'configs/superpixels_graph_classification_3WLGNN_CIFAR10_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/superpixels_graph_classification_3WLGNN_CIFAR10_L8_500k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/superpixels_graph_classification_3WLGNN_CIFAR10_L8_500k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/superpixels_graph_classification_3WLGNN_CIFAR10_L8_500k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/superpixels_graph_classification_3WLGNN_CIFAR10_L8_500k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --hidden_dim 100 --config 'configs/superpixels_graph_classification_RingGNN_CIFAR10_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --hidden_dim 100 --config 'configs/superpixels_graph_classification_RingGNN_CIFAR10_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --hidden_dim 100 --config 'configs/superpixels_graph_classification_RingGNN_CIFAR10_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --hidden_dim 100 --config 'configs/superpixels_graph_classification_RingGNN_CIFAR10_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/superpixels_graph_classification_RingGNN_CIFAR10_L8_500k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/superpixels_graph_classification_RingGNN_CIFAR10_L8_500k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/superpixels_graph_classification_RingGNN_CIFAR10_L8_500k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/superpixels_graph_classification_RingGNN_CIFAR10_L8_500k.json' &
wait" C-m
tmux send-keys "tmux kill-session -t benchmark_CIFAR10" C-m








