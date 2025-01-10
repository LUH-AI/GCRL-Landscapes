#!/usr/bin/env bash

# for running on cluster with nvidia gpus, use egl backend
export MUJOCO_GL=egl

# cache autotuning results to not having to recompile them on every run
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache"

python src/gcrl_landscapes/main.py --agent CRL --dataset antmaze-medium-navigate-v0 --n_configurations 32 --phase_steps 50000 100000 200000 --eval_steps 10000 50000 100000 200000 500000 --eval_episodes 10
