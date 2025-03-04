#!/bin/bash
set -e
BUILD_PATH="build"


if [ -z "${SPARSITY_IMAGES}" ]; then
  SPARSITY_IMAGES="."
fi


if [ -z "${SINGULARITY_TMPDIR}" ]; then
  echo -e "\033[38;5;208mBuilding singularity takes a lot of space. Setting SINGULARITY_TMPDIR environment variable  might be a good option to point to a folder with substantial amount of space.\033[0m"
fi


if [ ! -d ${BUILD_PATH} ]; then
    mkdir -p ${BUILD_PATH}
fi

if [ ! -f "${BUILD_PATH}/sparsity-base.sif" ]; then
    echo "Building sparsity-base.sif from sparsity-base.def..."
    singularity build --fakeroot ${BUILD_PATH}/sparsity-base.sif singularity_defs/sparsity-base.def
else
    echo "sparsity-base.sif already exists."
fi

timestamp=$(date +'%Y.%m.%d_%H.%M.%S')
output_image="sparsity_${timestamp}.sif"
echo "Building ${output_image} from sparsity-head.def..."
singularity build --fakeroot "${SPARSITY_IMAGES}/${output_image}" singularity_defs/sparsity-head.def

echo "Build process completed."