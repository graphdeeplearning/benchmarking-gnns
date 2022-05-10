

# Command to download dataset:
#   bash script_download_CSL.sh


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




