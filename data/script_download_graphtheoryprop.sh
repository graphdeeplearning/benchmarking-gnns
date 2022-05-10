# Command to download dataset:
#   bash script_download_graphtheoryprop.sh

DIR=graphtheoryprop/
cd $DIR

FILE=GraphTheoryProp.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/sat1tj9lzvljtpe/GraphTheoryProp.pkl?dl=1 -o GraphTheoryProp.pkl -J -L -k
fi
