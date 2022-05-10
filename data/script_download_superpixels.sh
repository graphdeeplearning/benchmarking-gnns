

# Command to download dataset:
#   bash script_download_superpixels.sh


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






