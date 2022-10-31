#!/bin/bash

sudo rm -f /var/lib/apt/lists/lock || true
sudo apt -y update
sudo apt -y upgrade python3.9 python3.9-venv
                
sudo pip uninstall jax jaxlib libtpu-nightly libtpu -y
pip uninstall jax jaxlib libtpu-nightly libtpu -y

python3.9 -m pip install --user virtualenv
python3.9 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

git clone https://github.com/dashstander/geomstats.git

GEOMSTATS_BACKEND=jax pip install ./geomstats