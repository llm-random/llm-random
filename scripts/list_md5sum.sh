#!/bin/bash

find configs -name '*.yaml' -type f -exec md5sum {} +
