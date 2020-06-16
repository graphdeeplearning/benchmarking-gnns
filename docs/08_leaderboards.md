# Leaderboards

The leaderboard includes the best performing GNN models on each datasets, _in order_, with their scores and the number of trainable parameters. The **small** parameter models have 100k trainable parameters and the **large** parameter models have 500k trainable parameters.

## 1. PATTERN - Node Classification

**Models with small configs, _i.e._ 100k trainable parameters**   

|Rank|Model | #Params | Test Acc  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
|1| RingGNN | 105206 | 86.245 &plusmn; 0.013 | [Link](https://papers.nips.cc/paper/9718-on-the-equivalence-between-graph-isomorphism-testing-and-function-approximation-with-gnns) |
|2| 3WLGNN | 103572 | 85.661 &plusmn; 0.353 | [Link](https://arxiv.org/abs/1905.11136) |
|3| GIN | 100884 | 85.590 &plusmn; 0.011 | [Link](https://arxiv.org/abs/1810.00826)|
|4| MoNet | 103775 | 85.482 &plusmn; 0.037 | [Link](https://arxiv.org/abs/1611.08402) |
|5| GatedGCN  | 104003 | 84.480 &plusmn; 0.122 | [Link](https://arxiv.org/abs/1711.07553) |
|6| GAT | 109936 | 75.824 &plusmn; 1.823 | [Link](https://arxiv.org/abs/1710.10903) |
|7| GCN  | 100923 | 63.880 &plusmn; 0.074 | [Link](https://arxiv.org/abs/1609.02907) |
|8| GraphSage | 101739 | 50.516 &plusmn; 0.001 | [Link](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) |

**Models with large configs, _i.e._ 500k trainable parameters**   

|Rank|Model | #Params | Test Acc  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
|1|GatedGCN-PE | 505421 | 86.363 &plusmn; 0.127| [Link](https://arxiv.org/abs/2003.00982) |
|2|RingGNN | 504766 | 86.244 &plusmn; 0.025 |[Link](https://papers.nips.cc/paper/9718-on-the-equivalence-between-graph-isomorphism-testing-and-function-approximation-with-gnns) |
|3|MoNet | 511487 | 85.582 &plusmn; 0.038 | [Link](https://arxiv.org/abs/1611.08402) |
|4|GatedGCN | 502223 | 85.568 &plusmn; 0.088 | [Link](https://arxiv.org/abs/1711.07553) |
|5|GIN | 508574 | 85.387  &plusmn; 0.136 | [Link](https://arxiv.org/abs/1810.00826)|
|6|3WLGNN | 502872 | 85.341 &plusmn; 0.207 | [Link](https://arxiv.org/abs/1905.11136) |
|7|GAT | 526990 | 78.271 &plusmn; 0.186 | [Link](https://arxiv.org/abs/1710.10903) |
|8|GCN | 500823 | 71.892 &plusmn; 0.334 | [Link](https://arxiv.org/abs/1609.02907) |
|9|GraphSage | 502842 | 50.492 &plusmn; 0.001 | [Link](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) |

<br>

## 2. CLUSTER - Node Classification

**Models with small configs, _i.e._ 100k trainable parameters**   

|Rank|Model | #Params | Test Acc  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
|1|GatedGCN|104355|60.404 &plusmn; 0.419|[Link](https://bit.ly/gatedgcn-paper) |
|2|GIN|103544|58.384 &plusmn; 0.236|[Link](https://bit.ly/gin-paper) |
|3|MoNet|104227|58.064 &plusmn; 0.131| [Link](https://bit.ly/monet-paper) |
|4|GAT|110700|57.732 &plusmn; 0.323|[Link](https://bit.ly/gat-paper) |
|5|3WLGNN|105552|57.130 &plusmn; 6.539|[Link](https://bit.ly/3wlgnn-paper) |
|6|GCN| 101655| 53.445 &plusmn; 2.029 | [Link](https://bit.ly/gcn-paper) |
|7|GraphSage|102187|50.454 &plusmn; 0.145|[Link](https://stanford.io/graphsage-paper) |
|8|RingGNN|104746|42.418 &plusmn; 20.063|[Link](https://bit.ly/ring-gnn-paper) |


**Models with large configs, _i.e._ 500k trainable parameters**   


|Rank|Model | #Params | Test Acc  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
|1|GatedGCN-PE|503473|74.088 &plusmn; 0.344|[Link](https://bit.ly/gatedgcn-pe-paper) |
|2|GatedGCN|502615|73.840 &plusmn; 0.326|[Link](https://bit.ly/gatedgcn-paper) |
|3|GAT|527874|70.587 &plusmn; 0.447|[Link](https://bit.ly/gat-paper) |
|4|GCN|501687|68.498 &plusmn; 0.976|[Link](https://bit.ly/gcn-paper) |
|5|MoNet|511999|66.407 &plusmn; 0.540|[Link](https://bit.ly/monet-paper) |
|6|GIN|517570|64.716 &plusmn; 1.553|[Link](https://bit.ly/gin-paper) |
|7|GraphSage|503350|63.844 &plusmn; 0.110|[Link](https://stanford.io/graphsage-paper) |
|8|3WLGNN|507252|55.489 &plusmn; 7.863|[Link](https://bit.ly/3wlgnn-paper) |
|9|RingGNN|524202|22.340 &plusmn; 0.000|[Link](https://bit.ly/ring-gnn-paper) |



<br>

## 3. ZINC - Graph Regression

**Models with small configs, _i.e._ 100k trainable parameters**   

|Rank|Model | #Params | Test MAE  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|


**Models with large configs, _i.e._ 500k trainable parameters**   


|Rank|Model | #Params | Test MAE  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|



<br>

## 4. MNIST - Graph Classification

**Models with small configs, _i.e._ 100k trainable parameters**   

|Rank|Model | #Params | Test Acc  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|


**Models with large configs, _i.e._ 500k trainable parameters**   


|Rank|Model | #Params | Test Acc  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|



<br>

## 5. CIFAR10 - Graph Classification

**Models with small configs, _i.e._ 100k trainable parameters**   

|Rank|Model | #Params | Test Acc  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|


**Models with large configs, _i.e._ 500k trainable parameters**   


|Rank|Model | #Params | Test Acc  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|



<br>

## 6. TSP - Edge Classification/Link Prediction

**Models with small configs, _i.e._ 100k trainable parameters**   

|Rank|Model | #Params | Test F1  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|


**Models with large configs, _i.e._ 500k trainable parameters**   


|Rank|Model | #Params | Test F1  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|



<br>

## 7. OGBL-COLLAB - Edge Classification/Link Prediction

**Models with small configs, _i.e._ 100k trainable parameters**   

|Rank|Model | #Params | Test Hits@50  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|


**Models with large configs, _i.e._ 500k trainable parameters**   


|Rank|Model | #Params | Test Hits@50  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|






<br><br><br>