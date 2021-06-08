#!/usr/bin/env bash

# find OS version
source scripts/os_version.sh

if [ $ARCH = "aarch64" ]; then
	TAG="r$L4T_VERSION"
elif [ $ARCH = "x86_64" ]; then
	TAG="$ARCH"
else
	echo "unsupported architecture:  $ARCH"
	exit 1
fi

CONTAINER_NAME="jetson-voice"
CONTAINER_IMAGE="$CONTAINER_NAME:$TAG"
CONTAINER_REMOTE_IMAGE="dustynv/$CONTAINER_IMAGE"

# check for local image
if [[ "$(sudo docker images -q $CONTAINER_IMAGE 2> /dev/null)" == "" ]]; then
	CONTAINER_IMAGE=$CONTAINER_REMOTE_IMAGE
fi

