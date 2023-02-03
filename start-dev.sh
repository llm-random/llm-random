#!/bin/sh

# setup git hooks
git config --local core.hooksPath .githooks

# setup the virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    . venv/bin/activate
    python3 -m pip install -U pip
    python3 -m pip install -r requirements.txt
    pip install --upgrade --force-reinstall datasets==2.9.0
else
    echo "[start-dev.sh] I don't know Walt, seems kind of sus to me. You've already got a venv folder, do you really want to make another one? Try again."
fi
