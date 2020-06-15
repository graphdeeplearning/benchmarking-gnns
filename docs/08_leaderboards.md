# Leaderboards

The leaderboard includes the best performing GNN models on each datasets, _in order_, with their scores and the number of trainable parameters. The **small** parameter models have 100k trainable parameters and the **large** parameter models have 500k trainable parameters. For each dataset, the GNN model with the best performance score is is highlighted as **bold**.

## 1. PATTERN - Node Classification

### 1.a Models with small configs, _i.e._ 100k trainable parameters

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

### 1.b Models with large configs, _i.e._ 500k trainable parameters   

|Rank|Model | #Params | Test Acc  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
|1|**GatedGCN-PE** | **505421** | **86.363 &plusmn; 0.127**| [Link](https://arxiv.org/abs/2003.00982) |
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
| | | | | | |
**Models with large configs, _i.e._ 500k trainable parameters**   


|Rank|Model | #Params | Test Acc  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
| | | | | | 


<br>

## 3. ZINC - Graph Regression

**Models with small configs, _i.e._ 100k trainable parameters**   

|Rank|Model | #Params | Test MAE  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
| | | | | | |
**Models with large configs, _i.e._ 500k trainable parameters**   


|Rank|Model | #Params | Test MAE  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
| | | | | | 

<br>

## 4. MNIST - Graph Classification

**Models with small configs, _i.e._ 100k trainable parameters**   

|Rank|Model | #Params | Test Acc  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
| | | | | | |
**Models with large configs, _i.e._ 500k trainable parameters**   


|Rank|Model | #Params | Test Acc  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
| | | | | | 

<br>

## 5. CIFAR10 - Graph Classification

**Models with small configs, _i.e._ 100k trainable parameters**   

|Rank|Model | #Params | Test Acc  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
| | | | | | |
**Models with large configs, _i.e._ 500k trainable parameters**   


|Rank|Model | #Params | Test Acc  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
| | | | | | 

<br>

## 6. TSP - Edge Classification/Link Prediction

**Models with small configs, _i.e._ 100k trainable parameters**   

|Rank|Model | #Params | Test F1  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
| | | | | | |
**Models with large configs, _i.e._ 500k trainable parameters**   


|Rank|Model | #Params | Test F1  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
| | | | | | 

<br>

## 7. OGBL-COLLAB - Edge Classification/Link Prediction

**Models with small configs, _i.e._ 100k trainable parameters**   

|Rank|Model | #Params | Test Hits@50  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
| | | | | | |
**Models with large configs, _i.e._ 500k trainable parameters**   


|Rank|Model | #Params | Test Hits@50  &plusmn; s.d. | Paper |
|----| ---------- |------------:| :--------:|:-------:|
| | | | | | 




<br><br><br>