#!/bin/bash
# use either `apptainer` or `singularity` - should be mostly compatible
COMMAND=$(command -v apptainer >/dev/null && echo apptainer || echo singularity)
$COMMAND build --fakeroot sparsity-base-lumi.sif sparsity-base-lumi.def
$COMMAND build --fakeroot sparsity-lumi-`date +'%Y.%m.%d_%H.%M.%S'`.sif sparsity-head-lumi.def