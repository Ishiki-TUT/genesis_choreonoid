#!/bin/bash

# --build-system
# --build-version

set -x

IRSL_TAG=_new_rsl_rl
# IRSL_TAG=_latest
# IRSL_TAG=''

_UBUNTU_VER=22.04
_ROS_DISTRO=one
REPO=repo.irsl.eiiris.tut.ac.jp/
IRSL_SYSTEM_IMAGE_NAME=${REPO}irsl_system:22.04_one
# IRSL_SYSTEM_IMAGE_NAME=${REPO}irsl_system:24.04_one
OUTPUT_IMAGE_NAME=${REPO}genesis_with_irsl${IRSL_TAG}:22.04_one
# OUTPUT_IMAGE_NAME=${REPO}genesis_with_irsl${IRSL_TAG}:24.04_one

DOCKER_OPT='--progress plain'

# make irsl_system with cuda
FIND_IMAGE=$(docker image ls --format '{{.Repository}}:{{.Tag}}' | grep ${IRSL_SYSTEM_IMAGE_NAME})
if [ -n "${BUILD_SYSTEM}" ]; then
    FIND_IMAGE=
else
    docker pull ${IRSL_SYSTEM_IMAGE_NAME}
fi
if [ -z "${FIND_IMAGE}" ]; then
    cd .irsl_docker_irsl_system/
    BUILD_ROS=${_ROS_DISTRO} BUILD_UBUNTU=${_UBUNTU_VER} ./build.sh ${IRSL_SYSTEM_IMAGE_NAME}
    cd ..
fi

docker build . ${DOCKER_OPT} --build-arg BASE_IMAGE=${IRSL_SYSTEM_IMAGE_NAME} -f Dockerfile.add_genesis${IRSL_TAG} -t ${OUTPUT_IMAGE_NAME}
