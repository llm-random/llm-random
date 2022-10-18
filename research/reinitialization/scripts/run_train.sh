#!/bin/bash

source venv/bin/activate
python3 -m research.reinitialization.reinit_train NAME=pruning-U
