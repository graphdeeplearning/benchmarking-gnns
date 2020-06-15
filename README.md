## Updates

**Jun 11, 2020**: 
- We released a new version of the [paper](https://arxiv.org/abs/2003.00982v2) which incorporates experimental pipeline for WL-GNNs operating on dense rank-2 tensors. For the code corresponding to an [earlier version](https://arxiv.org/abs/2003.00982v1) of this project, please check [this branch](https://github.com/graphdeeplearning/benchmarking-gnns/tree/arXivV1).  
- We also include a [LEADERBOARD](#8-leaderboard) for all datasets, which will be updated regularly.

---

# Benchmarking Graph Neural Networks

<img src="./docs/gnns.jpg" align="right" width="350"/>

<br>

## 1. Benchmark installation

[Follow these instructions](./docs/01_benchmark_installation.md) to install the benchmark and setup the environment.


<br>

## 2. Download datasets

[Proceed as follows](./docs/02_download_datasets.md) to download the benchmark datasets.


<br>

## 3. Reproducibility 

[Use this page](./docs/03_run_codes.md) to run the codes and reproduce the published results.


<br>

## 4. Adding a new dataset 

[Instructions](./docs/04_add_dataset.md) to add a dataset to the benchmark.


<br>

## 5. Adding a Message-passing GCN

[Step-by-step directions](./docs/05_add_mpgcn.md) to add a MP-GCN to the benchmark.


<br>

## 6. Adding a Weisfeiler-Lehman GNN

[Step-by-step directions](./docs/06_add_wlgnn.md) to add a WL-GNN to the benchmark.



<br>

## 7. Reference 

```
@article{dwivedi2020benchmarkgnns,
  title={Benchmarking Graph Neural Networks},
  author={Dwivedi, Vijay Prakash and Joshi, Chaitanya K and Laurent, Thomas and Bengio, Yoshua and Bresson, Xavier},
  journal={arXiv preprint arXiv:2003.00982},
  year={2020}
}
```

---

## 8. Leaderboard

The leaderboard includes the best performing GNN models on each datasets, _in order_, with their scores and the number of trainable parameters. The **small** parameter models have 100k trainable parameters and the **large** parameter models have 500k trainable parameters.

### 8.1 PATTERN - Node Classification


<!-- || Small |  | | Large  |  |
| -------- |------------:| :--------: |------------| --------: |:------------:|
|**Model** | **#Params** | **Test Acc  &plusmn; s.d.** | **Model** | **#Params** | **Test Acc  &plusmn; s.d.** |
| GatedGCN  | 104003 | 84.480 &plusmn; 0.122 | GatedGCN | 502223 | 85.568 &plusmn; 0.088 |
| GCN  | 100923 | 63.880 &plusmn; 0.074 | GCN | 500823 | 71.892 &plusmn; 0.334 |
| GraphSage | 101739 | 50.516 &plusmn; 0.001 | GraphSage | 502842 | 50.492 &plusmn; 0.001 |
| MoNet | 103775 | 85.482 &plusmn; 0.037 | MoNet | 511487 | 85.582 &plusmn; 0.038 |
| GAT | 109936 | 75.824 &plusmn; 1.823 | GAT | 526990 | 78.271 &plusmn; 0.186 | 
| GIN | 100884 | 85.590 &plusmn; 0.011 | GIN | 508574 | 85.387  &plusmn; 0.136 |
| RingGNN | 105206 | 86.245 &plusmn; 0.013 | RingGNN | 504766 | 86.244 &plusmn; 0.025 |
| 3WLGNN | 103572 | 85.661 &plusmn; 0.353 | 3WLGNN | 502872 | 85.341 &plusmn; 0.207 |
| | | |GatedGCN-PE | 505421 | 86.363 &plusmn; 0.127 -->


|| Small |  | | Large  |  |
| -------- |------------:| :--------: |------------| --------: |:------------:|
|**Model** | **#Params** | **Test Acc  &plusmn; s.d.** | **Model** | **#Params** | **Test Acc  &plusmn; s.d.** |
| RingGNN | 105206 | 86.245 &plusmn; 0.013 |GatedGCN-PE | 505421 | 86.363 &plusmn; 0.127
| 3WLGNN | 103572 | 85.661 &plusmn; 0.353 |RingGNN | 504766 | 86.244 &plusmn; 0.025 |
| GIN | 100884 | 85.590 &plusmn; 0.011 |MoNet | 511487 | 85.582 &plusmn; 0.038 |
| MoNet | 103775 | 85.482 &plusmn; 0.037 |GatedGCN | 502223 | 85.568 &plusmn; 0.088 |
| GatedGCN  | 104003 | 84.480 &plusmn; 0.122 |GIN | 508574 | 85.387  &plusmn; 0.136 |
| GAT | 109936 | 75.824 &plusmn; 1.823 | 3WLGNN | 502872 | 85.341 &plusmn; 0.207 |
| GCN  | 100923 | 63.880 &plusmn; 0.074 |GAT | 526990 | 78.271 &plusmn; 0.186 | 
| GraphSage | 101739 | 50.516 &plusmn; 0.001 |GCN | 500823 | 71.892 &plusmn; 0.334 |
||||GraphSage | 502842 | 50.492 &plusmn; 0.001 |


### 8.2 CLUSTER - Node Classification

### 8.3 ZINC - Graph Regression

### 8.4 MNIST - Graph Classification

### 8.5 CIFAR10 - Graph Classification

### 8.6 TSP - Edge Classification/Link Prediction

### 8.7 OGBL-COLLAB - Edge Classification/Link Prediction


<br><br><br>

