#!/usr/bin/env bash

BASE_IMAGE=$1

# find OS version
source scripts/os_version.sh

if [ $ARCH = "aarch64" ]; then
	TAG="r$L4T_VERSION"
elif [ $ARCH = "x86_64" ]; then
	TAG=$ARCH
else
	echo "unsupported architecture:  $ARCH"
	exit 1
fi

