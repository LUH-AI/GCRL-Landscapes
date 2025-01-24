#!/usr/bin/env bash

if [ -z "$1" ]
  then
    echo "Agent empty"
    exit 1
fi
if [ -z "$2" ]
  then
    echo "Dataset empty"
    exit 1
fi

# for running on cluster with nvidia gpus, use egl backend
export MUJOCO_GL=egl

# cache autotuning results to not having to recompile them on every run
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache"

python src/gcrl_landscapes/main.py --agent "$1" --dataset "$2" --n_configurations 32 --phase_steps 50000 100000 150000 --eval_steps 10000 50000 100000 150000 200000 --eval_episodes 10
