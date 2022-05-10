

# Command to download dataset:
#   bash script_download_molecules.sh


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

FILE=AQSOL.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/x1ej77z6peqpk2t/AQSOL.pkl?dl=1 -o AQSOL.pkl -J -L -k
fi

