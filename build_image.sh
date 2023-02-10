#!/bin/bash
sudo singularity build sparsity-base.sif sparsity-base.def
sudo singularity build sparsity_`date +'%Y.%m.%d_%H.%M.%S'`.sif sparsity-head.def