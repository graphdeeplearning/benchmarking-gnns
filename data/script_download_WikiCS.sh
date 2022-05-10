

# Command to download dataset:
#   bash script_download_WikiCS.sh


DIR=WikiCS/
mkdir $DIR
cd $DIR

FILE=data.json
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://github.com/pmernyei/wiki-cs-dataset/raw/master/dataset/data.json -o data.json -J -L -k
fi




