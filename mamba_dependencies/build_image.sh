#!/bin/bash
singularity build --fakeroot sparsity-base.sif sparsity-base.def
singularity build --fakeroot sparsity_`date +'%Y.%m.%d_%H.%M.%S'`.sif sparsity-head.def