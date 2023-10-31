#!/bin/bash

module CUDA/11.7.0
echo "Will run the following command:"
echo "$@"
echo "==============================="
env
echo "==============================="
env | grep NEPTUNE
exit
$@