#!/usr/bin/env bash

# find OS version
source scripts/os_version.sh

if [ $ARCH = "aarch64" ]; then
	TAG="r$L4T_VERSION"
	
	if [ $L4T_VERSION = "32.5.1" ] || [ $L4T_VERSION = "32.5.2" ]; then
		TAG="r32.5.0"
	fi	
elif [ $ARCH = "x86_64" ]; then
	TAG="$ARCH"
else
	echo "unsupported architecture:  $ARCH"
	exit 1
fi

CONTAINER_NAME="jetson-voice"


