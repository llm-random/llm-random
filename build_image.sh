#!/bin/bash
COMMAND=$(command -v apptainer >/dev/null && echo apptainer || echo singularity)
# $COMMAND build --fakeroot sparsity-base.sif sparsity-base.def
$COMMAND build --fakeroot sparsity_`date +'%Y.%m.%d_%H.%M.%S'`.sif sparsity-head.def