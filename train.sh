#!/usr/bin/env bash

# for running on cluster with nvidia gpus, use egl backend
export MUJOCO_GL=egl

# cache autotuning results to not having to recompile them on every run
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache"

python src/gcrl_landscapes/main.py
