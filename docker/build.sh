#!/usr/bin/env bash

BASE_IMAGE=$1
NEMO_VERSION="1.0.0rc1"

# find container tag from os version
source docker/tag.sh

if [ $ARCH = "aarch64" ]; then
	if [ -z $BASE_IMAGE ]; then
		if [ $L4T_VERSION = "32.5.1" ]; then
			BASE_IMAGE="nvcr.io/nvidia/l4t-ml:r32.5.0-py3"
		elif [ $L4T_VERSION = "32.5.0" ]; then
			BASE_IMAGE="nvcr.io/nvidia/l4t-ml:r32.5.0-py3"
		elif [ $L4T_VERSION = "32.4.4" ]; then
			BASE_IMAGE="nvcr.io/nvidia/l4t-ml:r32.4.4-py3"
		elif [ $L4T_VERSION = "32.4.3" ]; then
			BASE_IMAGE="nvcr.io/nvidia/l4t-ml:r32.4.3-py3"
		elif [ $L4T_VERSION = "32.4.2" ]; then
			BASE_IMAGE="nvcr.io/nvidia/l4t-ml:r32.4.2-py3"
		else
			echo "cannot build jetson-voice docker container for L4T R$L4T_VERSION"
			echo "please upgrade to the latest JetPack, or build jetson-voice natively"
			exit 1
		fi
	fi
elif [ $ARCH = "x86_64" ]; then
	BASE_IMAGE=${BASE_IMAGE:-"nvcr.io/nvidia/nemo:$NEMO_VERSION"}
fi

echo "BASE_IMAGE=$BASE_IMAGE"
echo "TAG=jetson-voice:$TAG"

# build the container
sudo docker build -t jetson-voice:$TAG -f Dockerfile.$ARCH \
          --build-arg BASE_IMAGE=$BASE_IMAGE \
		--build-arg NEMO_VERSION=$NEMO_VERSION \
		.

