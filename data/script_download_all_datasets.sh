

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
	curl https://www.dropbox.com/s/bhimk9p1xst6dvo/ZINC.pkl?dl=1 -o ZINC.pkl -J -L -k
fi

FILE=ZINC-full.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/2m4iywux4debbvy/ZINC-full.pkl?dl=1 -o ZINC-full.pkl -J -L -k
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
	curl https://www.dropbox.com/s/wcfmo4yvnylceaz/MNIST.pkl?dl=1 -o MNIST.pkl -J -L -k
fi

FILE=CIFAR10.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/agocm8pxg5u8yb5/CIFAR10.pkl?dl=1 -o CIFAR10.pkl -J -L -k
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
	curl https://www.dropbox.com/s/edpjywwexztxann/SBM_CLUSTER.pkl?dl=1 -o SBM_CLUSTER.pkl -J -L -k
fi

FILE=SBM_PATTERN.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/9h6crgk4argc89o/SBM_PATTERN.pkl?dl=1 -o SBM_PATTERN.pkl -J -L -k
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
	curl https://www.dropbox.com/s/qga6q0gxx3wb8k0/TSP.pkl?dl=1 -o TSP.pkl -J -L -k
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











