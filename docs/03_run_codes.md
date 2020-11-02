# Reproducibility


<br>

## 1. Usage


<br>

### 1.1 In terminal

```
# Run the main file (at the root of the project)
python main_molecules_graph_regression.py --dataset ZINC --config 'configs/molecules_graph_regression_GatedGCN_ZINC_100k.json' # for CPU
python main_molecules_graph_regression.py --dataset ZINC --gpu_id 0 --config 'configs/molecules_graph_regression_GatedGCN_ZINC_100k.json' # for GPU
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

Output results are located in the folder defined by the variable `out_dir` in the corresponding config file (eg. [`configs/molecules_graph_regression_GatedGCN_ZINC_100k.json`](../configs/molecules_graph_regression_GatedGCN_ZINC_100k.json) file).  

If `out_dir = 'out/molecules_graph_regression/'`, then 

#### 2.1 To see checkpoints and results
1. Go to`out/molecules_graph_regression/results` to view all result text files.
2. Directory `out/molecules_graph_regression/checkpoints` contains model checkpoints.

#### 2.2 To see the training logs in Tensorboard on local machine
1. Go to the logs directory, i.e. `out/molecules_graph_regression/logs/`.
2. Run the commands
```
source activate benchmark_gnn
tensorboard --logdir='./' --port 6006
```
3. Open `http://localhost:6006` in your browser. Note that the port information (here 6006 but it may change) appears on the terminal immediately after starting tensorboard.


#### 2.3 To see the training logs in Tensorboard on remote machine
1. Move this [script](../scripts/TensorBoard/script_tensorboard.sh) to the root of the repository, i.e. benchmarking-gnns/.
2. Run the script `bash script_tensorboard.sh`.
3. On your local machine, run the command `ssh -N -f -L localhost:6006:localhost:6006 user@xx.xx.xx.xx`.
4. Open `http://localhost:6006` in your browser. Note that `user@xx.xx.xx.xx` corresponds to your user login and the IP of the remote machine.


<br>

## 3. Reproduce results (4 runs on all, except CSL and TUs)


```
# At the root of the project 
bash scripts/SuperPixels/script_main_superpixels_graph_classification_MNIST_100k.sh # run MNIST dataset for 100k params
bash scripts/SuperPixels/script_main_superpixels_graph_classification_MNIST_500k.sh # run MNIST dataset for 500k params; WL-GNNs
bash scripts/SuperPixels/script_main_superpixels_graph_classification_CIFAR10_100k.sh # run CIFAR10 dataset for 100k params
bash scripts/SuperPixels/script_main_superpixels_graph_classification_CIFAR10_500k.sh # run CIFAR10 dataset for 500k params; WL-GNNs

bash scripts/ZINC/script_main_molecules_graph_regression_ZINC_100k.sh # run ZINC dataset for 100k params
bash scripts/ZINC/script_main_molecules_graph_regression_ZINC_500k.sh # run ZINC dataset for 500k params
bash scripts/ZINC/script_main_molecules_graph_regression_ZINC_PE_GatedGCN_500k.sh # run ZINC dataset with PE for GatedGCN

bash scripts/SBMs/script_main_SBMs_node_classification_PATTERN_100k.sh # run PATTERN dataset for 100k params
bash scripts/SBMs/script_main_SBMs_node_classification_PATTERN_500k.sh # run PATTERN dataset for 500k params
bash scripts/SBMs/script_main_SBMs_node_classification_PATTERN_PE_GatedGCN_500k.sh # run PATTERN dataset with PE for GatedGCN
bash scripts/SBMs/script_main_SBMs_node_classification_CLUSTER_100k.sh # run CLUSTER dataset for 100k params
bash scripts/SBMs/script_main_SBMs_node_classification_CLUSTER_500k.sh # run CLUSTER dataset for 500k params
bash scripts/SBMs/script_main_SBMs_node_classification_CLUSTER_PE_GatedGCN_500k.sh # run CLUSTER dataset with PE for GatedGCN

bash scripts/TSP/script_main_TSP_edge_classification_100k.sh # run TSP dataset for 100k params
bash scripts/TSP/script_main_TSP_edge_classification_edge_feature_analysis.sh # run TSP dataset for edge feature analysis 

bash scripts/COLLAB/script_main_COLLAB_edge_classification_40k.sh # run OGBL-COLLAB dataset for 40k params
bash scripts/COLLAB/script_main_COLLAB_edge_classification_edge_feature_analysis.sh # run OGBL-COLLAB dataset for edge feature analysis 
bash scripts/COLLAB/script_main_COLLAB_edge_classification_PE_GatedGCN_40k.sh # run OGBL-COLLAB dataset with PE for GatedGCN

bash scripts/CSL/script_main_CSL_graph_classification_20_seeds.sh # run CSL dataset without node features on 20 seeds
bash scripts/CSL/script_main_CSL_graph_classification_PE_20_seeds.sh # run CSL dataset with PE on 20 seeds

bash scripts/TU/script_main_TUs_graph_classification_100k_seed1.sh # run TU datasets for 100k params on seed1
bash scripts/TU/script_main_TUs_graph_classification_100k_seed2.sh # run TU datasets for 100k params on seed2
```

Scripts are [located](../scripts/) at the `scripts/` directory of the repository.

 

 <br>

## 4. Generate statistics obtained over mulitple runs (except CSL and TUs)
After running a script, statistics (mean and standard variation) can be generated from a notebook. For example, after running the script `scripts/ZINC/script_main_molecules_graph_regression_ZINC_100k.sh`, go to the results folder `out/molecules_graph_regression/results/`, and run the [notebook](../scripts/StatisticalResults/generate_statistics_molecules_graph_regression_ZINC_100k.ipynb) `scripts/StatisticalResults/generate_statistics_molecules_graph_regression_ZINC_100k.ipynb` to generate the statistics.


















<br><br><br>