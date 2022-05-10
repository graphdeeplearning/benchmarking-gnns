# Leaderboards

The leaderboard includes the best performing GNN models on each datasets, _in order_, with their scores and the number of trainable parameters. 
<!-- The **small** parameter models have 100k trainable parameters and the **large** parameter models have 500k trainable parameters. -->

## 1. PATTERN - Node Classification


**Models with configs having 500k trainable parameters**   

|Rank|Model | #Params | Test Acc  &plusmn; s.d. | Links |
|----| ---------- |------------:| :--------:|:-------:|
|1|GatedGCN-PE | 505421 | 86.363 &plusmn; 0.127| [Paper](https://arxiv.org/abs/2003.00982) |
|2|RingGNN | 504766 | 86.244 &plusmn; 0.025 |[Paper](https://papers.nips.cc/paper/9718-on-the-equivalence-between-graph-isomorphism-testing-and-function-approximation-with-gnns) |
|3|MoNet | 511487 | 85.582 &plusmn; 0.038 | [Paper](https://arxiv.org/abs/1611.08402) |
|4|GatedGCN | 502223 | 85.568 &plusmn; 0.088 | [Paper](https://arxiv.org/abs/1711.07553) |
|5|GIN | 508574 | 85.387  &plusmn; 0.136 | [Paper](https://arxiv.org/abs/1810.00826)|
|6|3WLGNN | 502872 | 85.341 &plusmn; 0.207 | [Paper](https://arxiv.org/abs/1905.11136) |
|7|GAT | 526990 | 78.271 &plusmn; 0.186 | [Paper](https://arxiv.org/abs/1710.10903) |
|8|GCN | 500823 | 71.892 &plusmn; 0.334 | [Paper](https://arxiv.org/abs/1609.02907) |
|9|GraphSage | 502842 | 50.492 &plusmn; 0.001 | [Paper](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) |

<br>

## 2. CLUSTER - Node Classification


**Models with configs having 500k trainable parameters**   


|Rank|Model | #Params | Test Acc  &plusmn; s.d. | Links |
|----| ---------- |------------:| :--------:|:-------:|
|1|GatedGCN-PE|503473|74.088 &plusmn; 0.344|[Paper](https://bit.ly/gatedgcn-pe-paper) |
|2|GatedGCN|502615|73.840 &plusmn; 0.326|[Paper](https://bit.ly/gatedgcn-paper) |
|3|GAT|527874|70.587 &plusmn; 0.447|[Paper](https://bit.ly/gat-paper) |
|4|GCN|501687|68.498 &plusmn; 0.976|[Paper](https://bit.ly/gcn-paper) |
|5|MoNet|511999|66.407 &plusmn; 0.540|[Paper](https://bit.ly/monet-paper) |
|6|GIN|517570|64.716 &plusmn; 1.553|[Paper](https://bit.ly/gin-paper) |
|7|GraphSage|503350|63.844 &plusmn; 0.110|[Paper](https://bit.ly/graphsage-paper) |
|8|3WLGNN|507252|55.489 &plusmn; 7.863|[Paper](https://bit.ly/3wlgnn-paper) |
|9|RingGNN|524202|22.340 &plusmn; 0.000|[Paper](https://bit.ly/ring-gnn-paper) |



<br>

## 3. ZINC - Graph Regression


**Models with configs having 500k trainable parameters**   


|Rank|Model | #Params | Test MAE  &plusmn; s.d. | Links |
|----| ---------- |------------:| :--------:|:-------:|
|1|PNA|387155 |0.142 &plusmn; 0.010|[Paper](https://bit.ly/pna-paper), [Code](https://bit.ly/pna-code) |
|2|MPNN (sum)|480805 |0.145 &plusmn; 0.007|[Paper](https://bit.ly/pna-paper), [Code](https://bit.ly/pna-code) |
|3|GatedGCN-PE|505011 |0.214 &plusmn; 0.006|[Paper](https://bit.ly/gatedgcn-pe-paper) |
|4|MPNN (max)|480805 |0.252 &plusmn; 0.009|[Paper](https://bit.ly/pna-paper), [Code](https://bit.ly/pna-code) |
|5|GatedGCN-E|504309| 0.282 &plusmn; 0.015|[Paper](https://bit.ly/gatedgcn-pe-paper) |
|6|MoNet|504013 |0.292 &plusmn; 0.006|[Paper](https://bit.ly/monet-paper) |
|7|3WLGNN-E|507603|0.303 &plusmn; 0.068|[Paper](https://bit.ly/3wlgnn-paper) |
|8|RingGNN-E|527283| 0.353 &plusmn; 0.019|[Paper](https://bit.ly/ring-gnn-paper) |
|9|GCN|505079| 0.367 &plusmn; 0.011|[Paper](https://bit.ly/gcn-paper) |
|10|GAT|531345|0.384 &plusmn; 0.007|[Paper](https://bit.ly/gat-paper) |
|11|GraphSage|505341 |0.398 &plusmn; 0.002|[Paper](https://bit.ly/graphsage-paper) |
|12|GIN|509549| 0.526 &plusmn; 0.051|[Paper](https://bit.ly/gin-paper) |



<br>

## 4. MNIST - Graph Classification

**Models with configs having 100k trainable parameters**   

|Rank|Model | #Params | Test Acc  &plusmn; s.d. | Links |
|----| ---------- |------------:| :--------:|:-------:|
|1|PNA|119812 |97.940 &plusmn; 0.120|[Paper](https://bit.ly/pna-paper), [Code](https://bit.ly/pna-code) |
|2|MPNN (max)|109057 |97.690 &plusmn; 0.220|[Paper](https://bit.ly/pna-paper), [Code](https://bit.ly/pna-code) |
|3|GatedGCN|104217 |97.340 &plusmn; 0.143|[Paper](https://bit.ly/gatedgcn-paper) |
|4|GraphSage|104337 |97.312 &plusmn; 0.097|[Paper](https://bit.ly/graphsage-paper) |
|5|MPNN (sum)|109057 |96.900 &plusmn; 0.150|[Paper](https://bit.ly/pna-paper), [Code](https://bit.ly/pna-code) |
|6|GIN|105434 |96.485 &plusmn; 0.252|[Paper](https://bit.ly/gin-paper) |
|7|GAT|110400| 95.535 &plusmn; 0.205|[Paper](https://bit.ly/gat-paper) |
|8|3WLGNN|108024 |95.075 &plusmn; 0.961|[Paper](https://bit.ly/3wlgnn-paper) |
|9|MoNet|104049 |90.805 &plusmn; 0.032|[Paper](https://bit.ly/monet-paper) |
|10|GCN|101365 |90.705 &plusmn; 0.218|[Paper](https://bit.ly/gcn-paper) |
|11|RingGNN|105398| 11.350 &plusmn; 0.000|[Paper](https://bit.ly/ring-gnn-paper) |


**Models with configs having 500k trainable parameters for 3WLGNN and RingGNN**   


|Rank|Model | #Params | Test Acc  &plusmn; s.d. | Links |
|----| ---------- |------------:| :--------:|:-------:|
|1|3WLGNN|501690|95.002 &plusmn; 0.419|[Paper](https://bit.ly/3wlgnn-paper) |
|2|RingGNN|505182| 91.860 &plusmn; 0.449|[Paper](https://bit.ly/ring-gnn-paper) |



<br>

## 5. CIFAR10 - Graph Classification

**Models with configs having 100k trainable parameters**   

|Rank|Model | #Params | Test Acc  &plusmn; s.d. | Links |
|----| ---------- |------------:| :--------:|:-------:|
|1|MPNN (max)|109277 |70.860 &plusmn; 0.270 |[Paper](https://bit.ly/pna-paper), [Code](https://bit.ly/pna-code) |
|2|PNA|113472 |70.350 &plusmn; 0.630 |[Paper](https://bit.ly/pna-paper), [Code](https://bit.ly/pna-code) |
|3|GatedGCN|104357|67.312 &plusmn; 0.311|[Paper](https://bit.ly/gatedgcn-paper) |
|4|GraphSage|104517|65.767 &plusmn; 0.308|[Paper](https://bit.ly/graphsage-paper) |
|5|MPNN (sum)|109277 |65.610 &plusmn; 0.300 |[Paper](https://bit.ly/pna-paper), [Code](https://bit.ly/pna-code) |
|6|GAT|110704|64.223 &plusmn; 0.455|[Paper](https://bit.ly/gat-paper) |
|7|3WLGNN|108516|59.175 &plusmn; 1.593|[Paper](https://bit.ly/3wlgnn-paper) |
|8|GCN|101657|55.710 &plusmn; 0.381|[Paper](https://bit.ly/gcn-paper) |
|9|GIN|105654|55.255 &plusmn; 1.527|[Paper](https://bit.ly/gin-paper) |
|10|MoNet|104229|54.655 &plusmn; 0.518|[Paper](https://bit.ly/monet-paper) |
|11|RingGNN|105165|19.300 &plusmn; 16.108|[Paper](https://bit.ly/ring-gnn-paper) |


**Models with configs having 500k trainable parameters for 3WLGNN and RingGNN**   


|Rank|Model | #Params | Test Acc  &plusmn; s.d. | Links |
|----| ---------- |------------:| :--------:|:-------:|
|1|3WLGNN|502770|58.043 &plusmn; 2.512|[Paper](https://bit.ly/3wlgnn-paper) |
|2|RingGNN|504949| 39.165 &plusmn; 17.114|[Paper](https://bit.ly/ring-gnn-paper) |



<br>

## 6. TSP - Edge Classification/Link Prediction

**Models with configs having 100k trainable parameters**   

|Rank|Model | #Params | Test F1  &plusmn; s.d. | Links |
|----| ---------- |------------:| :--------:|:-------:|
|1|GatedGCN-E|97858 |0.808 &plusmn; 0.003|[Paper](https://bit.ly/gatedgcn-pe-paper) |
|2|GatedGCN|97858 |0.791 &plusmn; 0.003|[Paper](https://bit.ly/gatedgcn-paper) |
|3|3WLGNN-E|106366 |0.694 &plusmn; 0.073|[Paper](https://bit.ly/3wlgnn-paper) |
|4|k-NN baseline|NA(k=2)|0.693 &plusmn; 0.000|[Paper](https://bit.ly/gatedgcn-pe-paper) |
|5|GAT|96182| 0.671 &plusmn; 0.002|[Paper](https://bit.ly/gat-paper) |
|6|GraphSage|99263 |0.665 &plusmn; 0.003|[Paper](https://bit.ly/graphsage-paper) |
|7|GIN|99002 |0.656 &plusmn; 0.003|[Paper](https://bit.ly/gin-paper) |
|8|RingGNN-E|106862 |0.643 &plusmn; 0.024|[Paper](https://bit.ly/ring-gnn-paper) |
|9|MoNet|99007| 0.641 &plusmn; 0.002|[Paper](https://bit.ly/monet-paper) |
|10|GCN|95702 |0.630 &plusmn; 0.001|[Paper](https://bit.ly/gcn-paper) |


**Models with configs having 500k trainable parameters**   


|Rank|Model | #Params | Test F1  &plusmn; s.d. | Links |
|----| ---------- |------------:| :--------:|:-------:|
|1|GatedGCN-E|500770 |0.838 &plusmn; 0.002|[Paper](https://bit.ly/gatedgcn-pe-paper) |
|2|RingGNN-E|507938| 0.704 &plusmn; 0.003|[Paper](https://bit.ly/ring-gnn-paper) |
|3|k-NN baseline|NA(k=2)|0.693 &plusmn; 0.000|[Paper](https://bit.ly/gatedgcn-pe-paper) |
|4|3WLGNN-E|506681|0.288 &plusmn; 0.311|[Paper](https://bit.ly/3wlgnn-paper) |



<br>

## 7. OGBL-COLLAB - Edge Classification/Link Prediction

**Models with configs having 40k trainable parameters**   

|Rank|Model | #Params | Test Hits@50  &plusmn; s.d. | Links |
|----| ---------- |------------:| :--------:|:-------:|
|1|GatedGCN|40965|52.816 &plusmn; 1.303|[Paper](https://bit.ly/gatedgcn-paper) |
|2|GatedGCN-PE|42769|52.018 &plusmn; 1.178|[Paper](https://bit.ly/gatedgcn-pe-paper) |
|3|GraphSage|39856|51.618 &plusmn; 0.690|[Paper](https://bit.ly/graphsage-paper) |
|4|GAT|42637|51.501 &plusmn; 0.962|[Paper](https://bit.ly/gat-paper) |
|5|GCN|40479|50.422 &plusmn; 1.131|[Paper](https://bit.ly/gcn-paper) |
|6|GatedGCN-E|40965|49.212 &plusmn; 1.560|[Paper](https://bit.ly/gatedgcn-pe-paper) |
|7|MatrixFact baseline|-|44.206 &plusmn; 0.452|[Paper](https://arxiv.org/abs/2005.00687)|
|8|GIN|39544|41.730 &plusmn; 2.284|[Paper](https://bit.ly/gin-paper) |
|9|MoNet|39751|36.144 &plusmn; 2.191|[Paper](https://bit.ly/monet-paper) |



**Note for OGBL-COLLAB** 
- 40k params is the highest we could fit the single OGBL-COLLAB graph on GPU for fair comparisons.   
- RingGNN and 3WLGNN rely on dense tensors which leads to OOM on both GPU and CPU memory.





<br><br><br>