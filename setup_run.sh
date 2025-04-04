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

if [ -z "$4" ]
  then
    echo "no logdir given"
    exit 1
fi


# cache autotuning results to not having to recompile them on every run
export JAX_COMPILATION_CACHE_DIR="${BIGWORK}/jax_cache"

python -m gcrl_landscapes.main setup --agent "$1" --dataset "$2" --n_configurations 32 --phases 50000 100000 200000 --eval_steps 50000 100000 200000 250000 --eval_episodes 50 --hyperparameters $3 --logdir "$4"
