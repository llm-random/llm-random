#!/bin/sh

# setup git hooks
git config --local core.hooksPath .githooks

# setup the virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
. venv/bin/activate
python3 -m pip install -r requirements.txt