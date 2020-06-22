# Contributing to Leaderboards

<br>

## 1. Adding new results

For fair comparisons, please follow our experimental settings to add new model’s performances to leaderboards of the datasets. In particular, all GNN models reported used either a small budget of 100k trainable parameters or a large budget of 500k parameters. Besides, the performance scores must use at least four runs with four different seeds (you may use the same seeds from our [scripts](https://github.com/graphdeeplearning/benchmarking-gnns/tree/master/scripts)).




<br>

## 2. Reproducibility

To ensure reproducibility of reported results, prepare executable scripts as instructed [here](https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/docs/03_run_codes.md) and make them available in your GitHub repo.




<br>

## 3. Results required for leaderboards

Please report the following fields.

- **Model**: Name of the model
- **#Params**: Number of trainable parameters
- **Test Perf ± s.d.**: Average test performance (Acc/MAE/F1/Hits@50) and the standard deviation of the results on 4 seeds. 
- **Links**: The hyperlinks for references to the paper and accompanying code repository.


For a demo script to generate summary statistics of results, refer [here](https://github.com/graphdeeplearning/benchmarking-gnns/tree/master/scripts/StatisticalResults).




<br>

## 4. Submitting the results

Two options are available for submission:

a. Create a Pull Request (PR) in our GitHub repo to make the required change in the leaderboard file, [here](https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/docs/07_leaderboards.md). Note to place your result on the appropriate row and update the ranks of the GNN models as necessary. Our team will merge the PR after validating the score and consequent ranks.

b. Fill in the Google Form, [here](https://forms.gle/c1vk4hyBDGAtjz9R6).




<br>


# Adding new GNN codes to the benchmark repo
- We have decided not to merge newly proposed GNN models in this repo. We strive to keep the benchmarking code infrastructure as modular and minimal as possible.
- For any new GNN model added to leaderboards, a hyperlink to the original code implementation, which builds on top of this benchmark, will be added.



<br>

We welcome contributions and suggestions to improve diverse aspects of the benchmarking framework.


<br><br><br>