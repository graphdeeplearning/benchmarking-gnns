# Command to download dataset:
#   bash script_download_cycles.sh

DIR=cycles/
cd $DIR

FILE=CYCLES_6_56.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/9fs9aqfp10q9wue/CYCLES_6_56.pkl?dl=1 -o CYCLES_6_56.pkl -J -L -k
fi
