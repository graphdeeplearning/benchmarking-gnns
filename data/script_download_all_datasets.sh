

# Command to download dataset:
#   bash script_download_all_datasets.sh



############
# ZINC
############

DIR=molecules/
cd $DIR

FILE=ZINC.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://data.dgl.ai/dataset/benchmarking-gnns/ZINC.pkl -o ZINC.pkl -J -L -k
fi

FILE=ZINC-full.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://data.dgl.ai/dataset/benchmarking-gnns/ZINC-full.pkl -o ZINC-full.pkl -J -L -k
fi

cd ..


############
# MNIST and CIFAR10
############

DIR=superpixels/
cd $DIR

FILE=MNIST.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://data.dgl.ai/dataset/benchmarking-gnns/MNIST.pkl -o MNIST.pkl -J -L -k
fi

FILE=CIFAR10.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://data.dgl.ai/dataset/benchmarking-gnns/CIFAR10.pkl -o CIFAR10.pkl -J -L -k
fi

cd ..


############
# PATTERN and CLUSTER 
############

DIR=SBMs/
cd $DIR

FILE=SBM_CLUSTER.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://data.dgl.ai/dataset/benchmarking-gnns/SBM_CLUSTER.pkl -o SBM_CLUSTER.pkl -J -L -k
fi

FILE=SBM_PATTERN.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://data.dgl.ai/dataset/benchmarking-gnns/SBM_PATTERN.pkl -o SBM_PATTERN.pkl -J -L -k
fi

cd ..


############
# TSP 
############

DIR=TSP/
cd $DIR

FILE=TSP.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://data.dgl.ai/dataset/benchmarking-gnns/TSP.pkl -o TSP.pkl -J -L -k
fi

cd ..


############
# CSL 
############

DIR=CSL/
cd $DIR

FILE=CSL.zip
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/rnbkp5ubgk82ocu/CSL.zip?dl=1 -o CSL.zip -J -L -k
	unzip CSL.zip -d ./
	rm -r __MACOSX/
fi

cd ..


############
# GraphTheoryProp 
############

DIR=graphtheoryprop/
cd $DIR

FILE=GraphTheoryProp.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/sat1tj9lzvljtpe/GraphTheoryProp.pkl?dl=1 -o GraphTheoryProp.pkl -J -L -k
fi

cd ..



############
# CYCLES 
############

DIR=cycles/
cd $DIR

FILE=CYCLES_6_56.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/9fs9aqfp10q9wue/CYCLES_6_56.pkl?dl=1 -o CYCLES_6_56.pkl -J -L -k
fi

cd ..






