#!/bin/bash
# setup_llama_env.sh — load Python 3.10 & your pip libs

module purge
module load horovod/python3.10_pytorch   # provides python3.10 + CUDA + pip
module load git-lfs                       # for huggingface-lfs clones

# Upgrade pip into your home
python -m pip install --user --upgrade pip

# Install all your deps
python -m pip install --user \
    transformers accelerate sentencepiece pandas openpyxl

# Make sure your ~/.local/bin is in front
export PATH="$HOME/.local/bin:$PATH"
