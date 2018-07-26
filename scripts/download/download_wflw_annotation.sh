# Download annotations
FILE=$1
TARGET_DIR=./datasets/$FILE/
mkdir -p $TARGET_DIR
URL_ANNO=https://wywu.github.io/projects/LAB/support/${FILE}_annotations.tar.gz
TAR_FILE_ANNO=./datasets/$FILE/${FILE}_annotations.tar.gz
wget -N $URL_ANNO -O $TAR_FILE_ANNO
tar -zxvf $TAR_FILE_ANNO -C ./datasets/$FILE/
rm $TAR_FILE_ANNO