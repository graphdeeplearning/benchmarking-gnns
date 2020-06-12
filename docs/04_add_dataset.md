# Adding a new dataset 


<br>

## 1. Organization of data/ directory 

Each dataset in the [data/](../data/) directory is pre-processed and prepared in a specific folder, for example [data/molecules/](../data/molecules) for the ZINC dataset. The graph dataset is prepared and saved in the DGL format (compatible with PyTorch), see e.g. [`prepare_molecules.ipynb`](../data/molecules/prepare_molecules.ipynb) where the graph dataset is saved in *ZINC.pkl* file.

File [`data.py`](../data/data.py) contains function `LoadData()` that loads any dataset by calling a specific data function, for instance `MoleculeDataset()` that is defined in [molecules.py](../data/molecules.py). 





<br>

## 2. How to add a new dataset?

<br>


### 2.1 Prepare your dataset

The user will first prepare for each graph, the adjacency matrix, the node feature and the edge feature (if any). 
See  [prepare_molecules.ipynb](../data/molecules/prepare_molecules.ipynb) that calls class *MoleculeDatasetDGL()* defined in file [molecules.py](../data/molecules.py).





<br>

### 2.2 Save your data in DGL format


Then, the user will convert the graph into the DGL format. See class *MoleculeDGL()* in file [molecules.py](../data/molecules.py). User will have to complete the `_prepare()` method for the new dataset. A standard code is 
```
class NewDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name, **kwargs):
        # other useful parameters, if needed
        
        self.graph_labels = []
        self.graph_lists = []
        self._prepare()
    
    def _prepare(self):
        # write here the code for preparation
        # of the new graph classification data
        
        # Steps
        # S1: initilize a dgl graph g = dgl.DGLGraph()
        # S2: add nodes using g.add_nodes()
        # S3: add edges using g.add_edges()
        # S4: add node feat by assigning a torch tensor to g.ndata['feat'] 
        # S5: add edge feat by assigning a torch tensor to g.edata['feat']
        # S6: Append the dgl graph to self.graph_lists
        
        # Repeat Steps S1 to S6 for 'n_samples' number of times
        
        # See data/molecules.py file for example, or the following link in dgl docs:
        # https://docs.dgl.ai/en/latest/_modules/dgl/data/minigc.html#MiniGCDataset
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]
```


### 2.3 Load your dataset

At the next step, the user will define a class `NewDataset()` that loads the DGL dataset and define a `collate()` module to create mini-batches of graphs.  Note that `collate()` function is for the MP-GCNs which use batches of sparse graphs, and `collate_dense_gnn()` is for the WL-GNNs which use dense graphs, with no batching of multiple graphs in one tensor.
```
class NewDataset(torch.utils.data.Dataset):
    def __init__(self, name):
        with open(name+'.pkl',"rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]
    
    def collate(self, samples):
    	graphs, labels = map(list, zip(*samples))
    	batched_graph = dgl.batch(graphs)
        return batched_graph, labels
        
    def collate_dense_gnn(self, samples):
        """
            we have the adjacency matrix in R^{n x n}, the node features in R^{d_n} and edge features R^{d_e}.
            Then we build a zero-initialized tensor, say X, in R^{(1 + d_n + d_e) x n x n}. X[0, :, :] is the adjacency matrix.
            The diagonal X[1:1+d_n, i, i], i = 0 to n-1, store the node feature of node i. 
            The off diagonal X[1+d_n:, i, j] store edge features of edge(i, j).
        """
        # prepare one dense tensor using above instruction
        # store as x_will_all_info
        
        return x_with_all_info, labels
```


### 2.4 Load your dataset with a name

The user will upgrade `LoadData(DATASET_NAME)` in `data.py` with the name of the new dataset and will return the dataset class `NewDataset()`. 
```
def LoadData(DATASET_NAME):
    if DATASET_NAME == 'NEW_DATA':
        return NewDataset(DATASET_NAME)
```





### 2.5 Create mini-batches for MP-GCNs

Eventually, the user will call function `LoadData(DATASET_NAME)` to load the dataset and function `DataLoader()` to create mini-batch of graphs. For example, this code loads the ZINC dataset and prepares mini-batch of 128 train graphs:
```
from data.data import LoadData
from data.molecules import MoleculeDataset
from torch.utils.data import DataLoader

DATASET_NAME = 'ZINC'
dataset = LoadData(DATASET_NAME)
train_loader = DataLoader(dataset.train, batch_size=128, shuffle=True, collate_fn=MoleculeDataset.collate)
```

**Note** that the batching approach for MP-GCNs is not applicable for WL-GNNs which operate on dense tensors. Therefore, we simply have the following code for WL-GNNs.

```
train_loader = DataLoader(dataset.train, shuffle=True, collate_fn=MoleculeDataset.collate_dense_gnn)
```



<br>

## 3. Dataset split

A data split for the TU dataset that preserves the class distribution across train-validation-test sets was prepared. The splits are stored in the [TUs/](../data/TUs) folder. Similarly, the split indices for CSL are stored in the [CSL/](../data/CSL) folder. We also store the split for the ZINC dataset in the [molecules/](../data/molecules) folder. For COLLAB, the dataset splits are automatically fetched from the OGB library.
















<br><br><br>