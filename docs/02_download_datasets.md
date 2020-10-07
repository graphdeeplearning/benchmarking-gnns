# Download datasets

The following sections except the last one only work for versions before DGL 0.5.  If you have DGL 0.5 or later, please refer to the last section.

<br>

## 1. TU datasets

Nothing to do. The TU datasets are automatically downloaded.



<br>

## 2. MNIST/CIFAR10 super-pixel datasets
MNIST size is 1.39GB and CIFAR10 size is 2.51GB.

```
# At the root of the project
cd data/ 
bash script_download_superpixels.sh
```
Script [script_download_superpixels.sh](../data/script_download_superpixels.sh) is located here. Codes to reproduce the datasets for [MNIST](../data/superpixels/prepare_superpixels_MNIST.ipynb) and for [CIFAR10](../data/superpixels/prepare_superpixels_CIFAR.ipynb).





<br>

## 3. ZINC molecular dataset
ZINC size is 58.9MB.

```
# At the root of the project
cd data/ 
bash script_download_molecules.sh
```
Script [script_download_molecules.sh](../data/script_download_molecules.sh) is located here. Code to reproduce the ZINC dataset is [here](../data/molecules/prepare_molecules.ipynb).


<br>

## 4. PATTERN/CLUSTER SBM datasets
PATTERN size is 1.98GB and CLUSTER size is 1.26GB.

```
# At the root of the project
cd data/ 
bash script_download_SBMs.sh
```
Script [script_download_SBMs.sh](../data/script_download_SBMs.sh) is located here. Codes to reproduce the datasets for [PATTERN](../data/SBMs/generate_SBM_PATTERN.ipynb) and for [CLUSTER](../data/SBMs/generate_SBM_CLUSTER.ipynb).

<br>

## 5. TSP dataset
TSP size is 1.87GB.

```
# At the root of the project
cd data/ 
bash script_download_TSP.sh
```
Script [script_download_TSP.sh](../data/script_download_TSP.sh) is located here. Codes to reproduce the TSP dataset is [here](../data/TSP/prepare_TSP.ipynb).

<br>

## 6. CSL dataset
CSL size is 27KB.

```
# At the root of the project
cd data/ 
bash script_download_CSL.sh
```
Script [script_download_CSL.sh](../data/script_download_CSL.sh) is located here. 

<br>

## 7. COLLAB dataset
COLLAB size is 360MB.

No script to run. The COLLAB dataset files will be automatically downloaded from OGB when running the experiment files for COLLAB.


<br>

## 8. All datasets

```
# At the root of the project
cd data/ 
bash script_download_all_datasets.sh
```

Script [script_download_all_datasets.sh](../data/script_download_all_datasets.sh) is located here. 

## 9. All datasets with DGL 0.5 or later

```
# At the root of the project
cd data/
bash script_download_all_datasets_dgl0.5.sh
```

Script [script_download_all_datasets_dgl0.5.sh](../data/script_download_all_datasets_dgl0.5.sh) is located here. 

If you previously had DGL 0.4.3post2 or earlier versions, you will need to remove the downloaded pickle files and run the script above to redownload the datasets to migrate to DGL 0.5.

<br><br><br>
