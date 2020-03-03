### Details:
- Split total number of graphs into 3 (train, val and test) in 80:10:10
- Stratified split proportionate to original distribution of data with respect to classes
- Using sklearn to perform the split and then save the indexes
- Preparing 10 such combinations of indexes split to be used in Graph NNs
- As with KFold, each of the 10 fold have unique test set.