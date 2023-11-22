#!/bin/bash

find research/conditional/train/configs/ -name '*.yaml' -type f -exec md5sum {} +
