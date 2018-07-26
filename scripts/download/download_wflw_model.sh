# Download models
FILE=$1
TARGET_DIR=./models/$FILE/
mkdir -p $TARGET_DIR
URL_MODEL=https://wywu.github.io/projects/LAB/support/${FILE}_final.tar.gz
TAR_FILE_MODEL=./models/$FILE/${FILE}_final.tar.gz
wget -N $URL_MODEL -O $TAR_FILE_MODEL
tar -zxvf $TAR_FILE_MODEL -C ./models/$FILE/
rm $TAR_FILE_MODEL

URL_MODEL=https://wywu.github.io/projects/LAB/support/${FILE}_wo_mp.tar.gz
TAR_FILE_MODEL=./models/$FILE/${FILE}_wo_mp.tar.gz
wget -N $URL_MODEL -O $TAR_FILE_MODEL
tar -zxvf $TAR_FILE_MODEL -C ./models/$FILE/
rm $TAR_FILE_MODEL