#!/bin/sh

# setup git hooks
git config --local core.hooksPath .githooks

# setup the virtual environment
if [ ! -d "venv" ]; then
    python3.10 -m venv venv
    . venv/bin/activate
    ./install_requirements.sh
else
    echo "[start-dev.sh] I don't know Walt, seems kind of sus to me. You've already got a venv folder, do you really want to make another one? Try again."
fi
