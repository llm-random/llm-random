#!/bin/bash

cat requirements.txt

python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
python3 -m pip install -U multiprocess
if ! command -v nvcc &> /dev/null; then
    echo "CUDA Toolkit is not installed. Skipping installation of mamba dependencies."
else
    python3 -m pip install --no-build-isolation -r mamba_requirements.txt
fi
