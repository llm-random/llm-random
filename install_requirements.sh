#!/bin/bash

cat requirements.txt

python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
python3 -m pip install -U multiprocess

