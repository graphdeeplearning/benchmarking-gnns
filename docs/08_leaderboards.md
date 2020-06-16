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
|7|GraphSage|102187|50.454 &plusmn; 0.145|[Link](https://bit.ly/graphsage-paper) |
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
|7|GraphSage|503350|63.844 &plusmn; 0.110|[Link](https://bit.ly/graphsage-paper) |
|8|3WLGNN|507252|55.489 &plusmn; 7.863|[Link](https://bit.ly/3wlgnn-paper) |
|9|RingGNN|524202|22.340 &plusmn; 0.000|[Link](https://bit.ly/ring-gnn-paper) |



<br>

## 3. ZINC - Graph Regression

**Models with small configs, _i.e._ 100k trainable parameters**   

|Rank|Model | #Params | Test MAE  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
|1|3WLGNN-E|103098| 0.256 &plusmn; 0.054|[Link](https://bit.ly/3wlgnn-paper) |
|2|RingGNN-E|104403 |0.363 &plusmn; 0.026|[Link](https://bit.ly/ring-gnn-paper) |
|3|GatedGCN-E|105875|0.375 &plusmn; 0.003|[Link](https://bit.ly/gatedgcn-pe-paper) |
|4|GIN|103079| 0.387 &plusmn; 0.015|[Link](https://bit.ly/gin-paper) |
|5|MoNet|106002|0.397 &plusmn; 0.010|[Link](https://bit.ly/monet-paper) |
|6|3WLGNN|102150 |0.407 &plusmn; 0.028|[Link](https://bit.ly/3wlgnn-paper) |
|7|GatedGCN|105735|0.435 &plusmn; 0.011|[Link](https://bit.ly/gatedgcn-paper) |
|8|GCN|103077| 0.459 &plusmn; 0.006|[Link](https://bit.ly/gcn-paper) |
|9|GraphSage|94977|0.468 &plusmn; 0.003|[Link](https://bit.ly/graphsage-paper) |
|10|GAT|102385|0.475 &plusmn; 0.007|[Link](https://bit.ly/gat-paper) |
|11|RingGNN|97978 |0.512 &plusmn; 0.023|[Link](https://bit.ly/ring-gnn-paper) |


**Models with large configs, _i.e._ 500k trainable parameters**   


|Rank|Model | #Params | Test MAE  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
|1|GatedGCN-PE|505011 |0.214 &plusmn; 0.006|[Link](https://bit.ly/gatedgcn-pe-paper) |
|2|GatedGCN-E|504309| 0.282 &plusmn; 0.015|[Link](https://bit.ly/gatedgcn-pe-paper) |
|3|MoNet|504013 |0.292 &plusmn; 0.006|[Link](https://bit.ly/monet-paper) |
|4|3WLGNN-E|507603|0.303 &plusmn; 0.068|[Link](https://bit.ly/3wlgnn-paper) |
|5|RingGNN-E|527283| 0.353 &plusmn; 0.019|[Link](https://bit.ly/ring-gnn-paper) |
|6|GCN|505079| 0.367 &plusmn; 0.011|[Link](https://bit.ly/gcn-paper) |
|7|GAT|531345|0.384 &plusmn; 0.007|[Link](https://bit.ly/gat-paper) |
|8|GraphSage|505341 |0.398 &plusmn; 0.002|[Link](https://bit.ly/graphsage-paper) |
|9|GIN|509549| 0.526 &plusmn; 0.051|[Link](https://bit.ly/gin-paper) |



<br>

## 4. MNIST - Graph Classification

**Models with small configs, _i.e._ 100k trainable parameters**   

|Rank|Model | #Params | Test Acc  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
|1|GatedGCN|104217 |97.340 &plusmn; 0.143|[Link](https://bit.ly/gatedgcn-paper) |
|2|GraphSage|104337 |97.312 &plusmn; 0.097|[Link](https://bit.ly/graphsage-paper) |
|3|GIN|105434 |96.485 &plusmn; 0.252|[Link](https://bit.ly/gin-paper) |
|4|GAT|110400| 95.535 &plusmn; 0.205|[Link](https://bit.ly/gat-paper) |
|5|3WLGNN|108024 |95.075 &plusmn; 0.961|[Link](https://bit.ly/3wlgnn-paper) |
|6|MoNet|104049 |90.805 &plusmn; 0.032|[Link](https://bit.ly/monet-paper) |
|7|GCN|101365 |90.705 &plusmn; 0.218|[Link](https://bit.ly/gcn-paper) |
|8|RingGNN|105398| 11.350 &plusmn; 0.000|[Link](https://bit.ly/ring-gnn-paper) |


**Models with large configs, _i.e._ 500k trainable parameters for 3WLGNN and RingGNN**   


|Rank|Model | #Params | Test Acc  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
|1|3WLGNN|501690|95.002 &plusmn; 0.419|[Link](https://bit.ly/3wlgnn-paper) |
|2|RingGNN|505182| 91.860 &plusmn; 0.449|[Link](https://bit.ly/ring-gnn-paper) |



<br>

## 5. CIFAR10 - Graph Classification

**Models with small configs, _i.e._ 100k trainable parameters**   

|Rank|Model | #Params | Test Acc  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
|1|GatedGCN|104357|67.312 &plusmn; 0.311|[Link](https://bit.ly/gatedgcn-paper) |
|2|GraphSage|104517|65.767 &plusmn; 0.308|[Link](https://bit.ly/graphsage-paper) |
|3|GAT|110704|64.223 &plusmn; 0.455|[Link](https://bit.ly/gat-paper) |
|4|3WLGNN|108516|59.175 &plusmn; 1.593|[Link](https://bit.ly/3wlgnn-paper) |
|5|GCN|101657|55.710 &plusmn; 0.381|[Link](https://bit.ly/gcn-paper) |
|6|GIN|105654|55.255 &plusmn; 1.527|[Link](https://bit.ly/gin-paper) |
|7|MoNet|104229|54.655 &plusmn; 0.518|[Link](https://bit.ly/monet-paper) |
|8|RingGNN|105165|19.300 &plusmn; 16.108|[Link](https://bit.ly/ring-gnn-paper) |


**Models with large configs, _i.e._ 500k trainable parameters for 3WLGNN and RingGNN**   


|Rank|Model | #Params | Test Acc  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
|1|3WLGNN|502770|58.043 &plusmn; 2.512|[Link](https://bit.ly/3wlgnn-paper) |
|2|RingGNN|504949| 39.165 &plusmn; 17.114|[Link](https://bit.ly/ring-gnn-paper) |



<br>

## 6. TSP - Edge Classification/Link Prediction

**Models with small configs, _i.e._ 100k trainable parameters**   

|Rank|Model | #Params | Test F1  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
|1|GatedGCN-E|97858 |0.808 &plusmn; 0.003|[Link](https://bit.ly/gatedgcn-pe-paper) |
|2|GatedGCN|97858 |0.791 &plusmn; 0.003|[Link](https://bit.ly/gatedgcn-paper) |
|3|3WLGNN-E|106366 |0.694 &plusmn; 0.073|[Link](https://bit.ly/3wlgnn-paper) |
|4|k-NN baseline|NA(k=2)|0.693 &plusmn; 0.000|[Link](https://bit.ly/gatedgcn-pe-paper) |
|5|GAT|96182| 0.671 &plusmn; 0.002|[Link](https://bit.ly/gat-paper) |
|6|GraphSage|99263 |0.665 &plusmn; 0.003|[Link](https://bit.ly/graphsage-paper) |
|7|GIN|99002 |0.656 &plusmn; 0.003|[Link](https://bit.ly/gin-paper) |
|8|RingGNN-E|106862 |0.643 &plusmn; 0.024|[Link](https://bit.ly/ring-gnn-paper) |
|9|MoNet|99007| 0.641 &plusmn; 0.002|[Link](https://bit.ly/monet-paper) |
|10|GCN|95702 |0.630 &plusmn; 0.001|[Link](https://bit.ly/gcn-paper) |


**Models with large configs, _i.e._ 500k trainable parameters**   


|Rank|Model | #Params | Test F1  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
|1|GatedGCN-E|500770 |0.838 &plusmn; 0.002|[Link](https://bit.ly/gatedgcn-pe-paper) |
|2|RingGNN-E|507938| 0.704 &plusmn; 0.003|[Link](https://bit.ly/ring-gnn-paper) |
|3|k-NN baseline|NA(k=2)|0.693 &plusmn; 0.000|[Link](https://bit.ly/gatedgcn-pe-paper) |
|4|3WLGNN-E|506681|0.288 &plusmn; 0.311|[Link](https://bit.ly/3wlgnn-paper) |



<br>

## 7. OGBL-COLLAB - Edge Classification/Link Prediction

**Models with configs having 40k trainable parameters**   

|Rank|Model | #Params | Test Hits@50  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
|1|GatedGCN|40965|52.816 &plusmn; 1.303|[Link](https://bit.ly/gatedgcn-paper) |
|2|GatedGCN-PE|42769|52.018 &plusmn; 1.178|[Link](https://bit.ly/gatedgcn-pe-paper) |
|3|GraphSage|39856|51.618 &plusmn; 0.690|[Link](https://bit.ly/graphsage-paper) |
|4|GAT|42637|51.501 &plusmn; 0.962|[Link](https://bit.ly/gat-paper) |
|5|GCN|40479|50.422 &plusmn; 1.131|[Link](https://bit.ly/gcn-paper) |
|6|GatedGCN-E|40965|49.212 &plusmn; 1.560|[Link](https://bit.ly/gatedgcn-pe-paper) |
|7|MatrixFact baseline|-|44.206 &plusmn; 0.452|[Link](https://arxiv.org/abs/2005.00687)|
|8|GIN|39544|41.730 &plusmn; 2.284|[Link](https://bit.ly/gin-paper) |
|9|MoNet|39751|36.144 &plusmn; 2.191|[Link](https://bit.ly/monet-paper) |



**Note for OGBL-COLLAB** 
- 40k params is the highest we could fit the single OGBL-COLLAB graph on GPU for fair comparisons.   
- RingGNN and 3WLGNN rely on dense tensors which leads to OOM on both GPU and CPU memory.





<br><br><br>