#!/bin/bash

# check : 
# bash script.sh
# tmux attach -t script_cit_graph
# tmux detach
# pkill python

# bash script_main_CitationGraphs_node_classification.sh


############
# GNNs
############

#GatedGCN TODO
#GCN
#GraphSage
#MLP
#MLP_GATED
#GIN
#GAT

code=main_CitationGraphs_node_classification.py 
tmux new -s benchmark_CitationGraphs_node_classification -d
tmux send-keys "conda activate benchmark_gnn" C-m

datasets=(CORA CITESEER PUBMED)
nets=(GCN GraphSage MLP MLP_GATED GIN GAT)
for dataset in ${datasets[@]}; do
    for net in ${nets[@]}; do
        tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --config 'configs/CitationGraphs_node_classification_$net.json' &
wait" C-m
    done
done
tmux send-keys "tmux kill-session -t benchmark_CitationGraphs_node_classification" C-m
