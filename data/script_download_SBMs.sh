

# Command to download dataset:
#   bash script_download_SBMs.sh


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


