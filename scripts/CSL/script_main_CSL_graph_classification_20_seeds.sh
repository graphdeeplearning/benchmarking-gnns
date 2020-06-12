#!/bin/bash


############
# Usage
############

# bash script_main_CSL_graph_classification_20_seeds.sh



############
# GNNs
############

#20 seeds for

#MLP
#GCN
#GraphSage
#GatedGCN
#GAT
#MoNet
#GIN
#3WLGNN
#RingGNN

# with Positional Encoding (P.E.)


############
# CSL 
############

code=main_CSL_graph_classification.py 
dataset=CSL

dir_MLP='out/CSL/MLP/'
dir_GCN='out/CSL/GCN/'
dir_GraphSage='out/CSL/GraphSage/'
dir_GatedGCN='out/CSL/GatedGCN/'
dir_GAT='out/CSL/GAT/'
dir_MoNet='out/CSL/MoNet/'
dir_GIN='out/CSL/GIN/'

dir_RingGNN_small='out/CSL/RingGNN_small/'
dir_3WLGNN_small='out/CSL/3WLGNN_small/'
dir_RingGNN_large='out/CSL/RingGNN_large/'
dir_3WLGNN_large='out/CSL/3WLGNN_large/'
tmux new -s benchmark_CSL_20_seeds -d
tmux send-keys "source activate benchmark_gnn" C-m
all_seeds=(12 32 52 82 92)
# above are starting seeds; from each value, 4 seeds are generated, see below,
# therefore, 5 x 4 = 20 seeds


for seed in ${all_seeds[@]}; do
    tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $((seed+2)) --out_dir $dir_MLP --config 'configs/CSL_graph_classification_MLP_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $((seed+3)) --out_dir $dir_MLP --config 'configs/CSL_graph_classification_MLP_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $((seed+6)) --out_dir $dir_MLP --config 'configs/CSL_graph_classification_MLP_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $((seed+7)) --out_dir $dir_MLP --config 'configs/CSL_graph_classification_MLP_CSL_100k.json' &
wait" C-m
done

for seed in ${all_seeds[@]}; do
    tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $((seed+2)) --out_dir $dir_GCN --config 'configs/CSL_graph_classification_GCN_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $((seed+3)) --out_dir $dir_GCN --config 'configs/CSL_graph_classification_GCN_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $((seed+6)) --out_dir $dir_GCN --config 'configs/CSL_graph_classification_GCN_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $((seed+7)) --out_dir $dir_GCN --config 'configs/CSL_graph_classification_GCN_CSL_100k.json' &
wait" C-m
done

for seed in ${all_seeds[@]}; do
    tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $((seed+2)) --out_dir $dir_GraphSage --config 'configs/CSL_graph_classification_GraphSage_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $((seed+3)) --out_dir $dir_GraphSage --config 'configs/CSL_graph_classification_GraphSage_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $((seed+6)) --out_dir $dir_GraphSage --config 'configs/CSL_graph_classification_GraphSage_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $((seed+7)) --out_dir $dir_GraphSage --config 'configs/CSL_graph_classification_GraphSage_CSL_100k.json' &
wait" C-m
done

for seed in ${all_seeds[@]}; do
    tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $((seed+2)) --out_dir $dir_GatedGCN --config 'configs/CSL_graph_classification_GatedGCN_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $((seed+3)) --out_dir $dir_GatedGCN --config 'configs/CSL_graph_classification_GatedGCN_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $((seed+6)) --out_dir $dir_GatedGCN --config 'configs/CSL_graph_classification_GatedGCN_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $((seed+7)) --out_dir $dir_GatedGCN --config 'configs/CSL_graph_classification_GatedGCN_CSL_100k.json' &
wait" C-m
done

for seed in ${all_seeds[@]}; do
    tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $((seed+2)) --out_dir $dir_GAT --config 'configs/CSL_graph_classification_GAT_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $((seed+3)) --out_dir $dir_GAT --config 'configs/CSL_graph_classification_GAT_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $((seed+6)) --out_dir $dir_GAT --config 'configs/CSL_graph_classification_GAT_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $((seed+7)) --out_dir $dir_GAT --config 'configs/CSL_graph_classification_GAT_CSL_100k.json' &
wait" C-m
done

for seed in ${all_seeds[@]}; do
    tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $((seed+2)) --out_dir $dir_MoNet --config 'configs/CSL_graph_classification_MoNet_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $((seed+3)) --out_dir $dir_MoNet --config 'configs/CSL_graph_classification_MoNet_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $((seed+6)) --out_dir $dir_MoNet --config 'configs/CSL_graph_classification_MoNet_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $((seed+7)) --out_dir $dir_MoNet --config 'configs/CSL_graph_classification_MoNet_CSL_100k.json' &
wait" C-m
done

for seed in ${all_seeds[@]}; do
    tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $((seed+2)) --out_dir $dir_GIN --config 'configs/CSL_graph_classification_GIN_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $((seed+3)) --out_dir $dir_GIN --config 'configs/CSL_graph_classification_GIN_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $((seed+6)) --out_dir $dir_GIN --config 'configs/CSL_graph_classification_GIN_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $((seed+7)) --out_dir $dir_GIN --config 'configs/CSL_graph_classification_GIN_CSL_100k.json' &
wait" C-m
done

for seed in ${all_seeds[@]}; do
    tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $((seed+2)) --out_dir $dir_3WLGNN_small --config 'configs/CSL_graph_classification_3WLGNN_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $((seed+3)) --out_dir $dir_3WLGNN_small --config 'configs/CSL_graph_classification_3WLGNN_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $((seed+6)) --out_dir $dir_3WLGNN_small --config 'configs/CSL_graph_classification_3WLGNN_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $((seed+7)) --out_dir $dir_3WLGNN_small --config 'configs/CSL_graph_classification_3WLGNN_CSL_100k.json' &
wait" C-m
done

for seed in ${all_seeds[@]}; do
    tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $((seed+2)) --out_dir $dir_RingGNN_small --config 'configs/CSL_graph_classification_RingGNN_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $((seed+3)) --out_dir $dir_RingGNN_small --config 'configs/CSL_graph_classification_RingGNN_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $((seed+6)) --out_dir $dir_RingGNN_small --config 'configs/CSL_graph_classification_RingGNN_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $((seed+7)) --out_dir $dir_RingGNN_small --config 'configs/CSL_graph_classification_RingGNN_CSL_100k.json' &
wait" C-m
done

for seed in ${all_seeds[@]}; do
    tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $((seed+2)) --out_dir $dir_3WLGNN_large --hidden_dim 181 --config 'configs/CSL_graph_classification_3WLGNN_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $((seed+3)) --out_dir $dir_3WLGNN_large --hidden_dim 181 --config 'configs/CSL_graph_classification_3WLGNN_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $((seed+6)) --out_dir $dir_3WLGNN_large --hidden_dim 181 --config 'configs/CSL_graph_classification_3WLGNN_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $((seed+7)) --out_dir $dir_3WLGNN_large --hidden_dim 181 --config 'configs/CSL_graph_classification_3WLGNN_CSL_100k.json' &
wait" C-m
done

for seed in ${all_seeds[@]}; do
    tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $((seed+2)) --out_dir $dir_RingGNN_large --hidden_dim 102 --config 'configs/CSL_graph_classification_RingGNN_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 1 --seed $((seed+3)) --out_dir $dir_RingGNN_large --hidden_dim 102 --config 'configs/CSL_graph_classification_RingGNN_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 2 --seed $((seed+6)) --out_dir $dir_RingGNN_large --hidden_dim 102 --config 'configs/CSL_graph_classification_RingGNN_CSL_100k.json' &
python $code --dataset $dataset --gpu_id 3 --seed $((seed+7)) --out_dir $dir_RingGNN_large --hidden_dim 102 --config 'configs/CSL_graph_classification_RingGNN_CSL_100k.json' &
wait" C-m
done

tmux send-keys "tmux kill-session -t benchmark_CSL_20_seeds" C-m





