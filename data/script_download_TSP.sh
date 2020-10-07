# Command to download dataset:
#   bash script_download_TSP.sh

DIR=TSP/
cd $DIR

FILE=TSP.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://data.dgl.ai/dataset/benchmarking-gnns/TSP.pkl -o TSP.pkl -J -L -k
fi
