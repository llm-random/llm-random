#!/bin/bash

find $(dirname "$0")/../../research/conditional/train/configs/ -name '*.yaml' -type f -exec md5sum {} +
