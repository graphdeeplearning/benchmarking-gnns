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

The leaderboard includes the best-5 performing GNN models on each datasets, with their scores and the number of trainable parameters.

### 8.1 PATTERN - Node Classification

|Model|#Params (small)|Test Acc  &plusmn; s.d.|#Params (large)|Test Acc  &plusmn; s.d.| 
| -------- |:------------:|:---------------:|:-----------:|:------------------------:|
| GatedGCN  | 104003 | 84.480 &plusmn; 0.122 | 502223 | 85.568 &plusmn; 0.088 |
| GCN  | 100923 | 63.880 &plusmn; 0.074 | 500823 | 71.892 &plusmn; 0.334 |

### 8.2 CLUSTER - Node Classification

### 8.3 ZINC - Graph Regression

### 8.4 MNIST - Graph Classification

### 8.5 CIFAR10 - Graph Classification

### 8.6 TSP - Edge Classification/Link Prediction

### 8.7 OGBL-COLLAB - Edge Classification/Link Prediction


<br><br><br>

