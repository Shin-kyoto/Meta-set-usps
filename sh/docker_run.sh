#!/bin/bash

mkdir {extra_data,dataset,libs,results}

# $1 : path/to/COCO
# $2 : path/to/mnist
# $3 : path/to/usps
ln -s $1 ./extra_data
ln -s $2 ./dataset/mnist
ln -s $3 ./dataset/usps

cd ./libs
git clone https://github.com/cocodataset/cocoapi.git
cd ../

# $4 : docker image name
docker build -t $4 .

# $5 : docker container name
# $6 : dataset_root
# $7 : work directory
docker run -it --name $5 --runtime nvidia --ipc host -v $6:$6 -v $7:/Meta-set  $4