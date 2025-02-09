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
if [ -z "$3" ]
  then
    echo "no hyperparameters given"
    exit 1
fi


# for running on cluster with nvidia gpus, use egl backend
export MUJOCO_GL=egl

# cache autotuning results to not having to recompile them on every run
export JAX_COMPILATION_CACHE_DIR="${BIGWORK}/jax_cache"

python -m gcrl_landscapes.main setup --agent "$1" --dataset "$2" --n_configurations 32 --phases 50000 100000 150000 --eval_steps 10000 50000 100000 150000 200000 --eval_episodes 10 --hyperparameters $3 --logdir "logs/2025-02-08-${1}-${2}-seeds_5-${3}-firsttry"
python -m gcrl_landscapes.main submit --logdir --logdir "logs/2025-02-08-${1}-${2}-seeds_5-${3}-firsttry" --phase 0 --n_seeds 5
