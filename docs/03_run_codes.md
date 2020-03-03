# Reproducibility


<br>

## 1. Usage


<br>

### 1.1 In terminal

```
# Run the main file (at the root of the project)
python main_molecules_graph_regression.py --dataset ZINC --config 'configs/molecules_graph_regression_GatedGCN_ZINC.json' # for CPU
python main_molecules_graph_regression.py --dataset ZINC --gpu_id 0 --config 'configs/molecules_graph_regression_GatedGCN_ZINC.json' # for GPU
```
The training and network parameters for each dataset and network is stored in a json file in the [`configs/`](../configs) directory.












<br>

### 1.2 In jupyter notebook
```
# Run the notebook file (at the root of the project)
conda activate benchmark_gnn 
jupyter notebook
```
Use [`main_molecules_graph_regression.ipynb`](../main_molecules_graph_regression.ipynb) notebook to explore the code and do the training interactively.




<br>

## 2. Output, checkpoints and visualizations

Output results are located in the folder defined by the variable `out_dir` in the corresponding config file (eg. [`configs/molecules_graph_regression_GatedGCN_ZINC.json`](../configs/molecules_graph_regression_GatedGCN_ZINC.json) file).  

If `out_dir = 'out/molecules_graph_regression/'`, then 

#### 2.1 To see checkpoints and results
1. Go to`out/molecules_graph_regression/results` to view all result text files.
2. Directory `out/molecules_graph_regression/checkpoints` contains model checkpoints.

#### 2.2 To see the training logs in Tensorboard
1. Go to the logs directory, i.e. `out/molecules_graph_regression/logs/`
2. Run the command `tensorboard --logdir='./'`
3. Open `http://localhost:6006` in your browser. Note that the port information (here 6006) appears on the terminal immediately after running Step 2.




<br>

## 3. Reproduce results

<br>

### 3.1 Results (1 run)

```
# At the root of the project
bash script_one_code_to_rull_them_all.sh # run all datasets and all GNNs
```

See script [script_one_code_to_rull_them_all.sh](../script_one_code_to_rull_them_all.sh). 




<br>

### 3.2 Results (4 runs, except TSP)

```
# At the root of the project
bash script_main_TUs_graph_classification.sh # run TU datasets
bash script_main_superpixels_graph_classification_MNIST.sh # run MNIST dataset
bash script_main_superpixels_graph_classification_CIFAR10.sh # run CIFAR10 dataset
bash script_main_molecules_graph_regression_ZINC.sh # run ZINC dataset
bash script_main_SBMs_node_classification_PATTERN.sh # run PATTERN dataset
bash script_main_SBMs_node_classification_CLUSTER.sh # run CLUSTER dataset
bash script_main_TSP_edge_classification.sh # run TSP dataset
```

Scripts are [located](../../../) at the root of the repository.

 


















<br><br><br>