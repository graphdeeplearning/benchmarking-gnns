#!/bin/bash



# bash script_main_TUs_graph_classification_100k_seed2.sh


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
# ENZYMES & DD & PROTEINS_full
############
seed=95
code=main_TUs_graph_classification.py 
tmux new -s benchmark_TUs_graph_classification -d
tmux send-keys "source activate benchmark_gnn" C-m
dataset=ENZYMES
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed --config 'configs/TUs_graph_classification_GatedGCN_ENZYMES_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed --config 'configs/TUs_graph_classification_GCN_ENZYMES_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed --config 'configs/TUs_graph_classification_GraphSage_ENZYMES_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed --config 'configs/TUs_graph_classification_MLP_ENZYMES_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed --config 'configs/TUs_graph_classification_GIN_ENZYMES_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed --config 'configs/TUs_graph_classification_MoNet_ENZYMES_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed --config 'configs/TUs_graph_classification_GAT_ENZYMES_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed --config 'configs/TUs_graph_classification_3WLGNN_ENZYMES_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed --config 'configs/TUs_graph_classification_RingGNN_ENZYMES_100k.json' &
wait" C-m
dataset=DD
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed --config 'configs/TUs_graph_classification_GatedGCN_DD_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed --config 'configs/TUs_graph_classification_GCN_DD_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed --config 'configs/TUs_graph_classification_GraphSage_DD_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed --config 'configs/TUs_graph_classification_MLP_DD_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed --config 'configs/TUs_graph_classification_GIN_DD_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed --config 'configs/TUs_graph_classification_MoNet_DD_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed --config 'configs/TUs_graph_classification_GAT_DD_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed --config 'configs/TUs_graph_classification_3WLGNN_DD_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed --config 'configs/TUs_graph_classification_RingGNN_DD_100k.json' &
wait" C-m
dataset=PROTEINS_full
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed --config 'configs/TUs_graph_classification_GatedGCN_PROTEINS_full_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed --config 'configs/TUs_graph_classification_GCN_PROTEINS_full_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed --config 'configs/TUs_graph_classification_GraphSage_PROTEINS_full_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed --config 'configs/TUs_graph_classification_MLP_PROTEINS_full_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed --config 'configs/TUs_graph_classification_GIN_PROTEINS_full_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed --config 'configs/TUs_graph_classification_MoNet_PROTEINS_full_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed --config 'configs/TUs_graph_classification_GAT_PROTEINS_full_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed --config 'configs/TUs_graph_classification_3WLGNN_PROTEINS_full_100k.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed --config 'configs/TUs_graph_classification_RingGNN_PROTEINS_full_100k.json' &
wait" C-m
tmux send-keys "tmux kill-session -t benchmark_TUs_graph_classification" C-m



