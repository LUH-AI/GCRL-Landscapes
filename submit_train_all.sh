#!/usr/bin/env bash
# Submit train_all.sh to Slurm with cluster-appropriate partition/reservation.
# Usage: [CLUSTER=pc2] bash submit_train_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/configs/cluster/${CLUSTER:-luh}.sh"

sbatch \
  "$@" \
  --partition="${CLUSTER_SETUP_PARTITION}" \
  ${CLUSTER_RESERVATION:+--reservation="${CLUSTER_RESERVATION}"} \
  --export=ALL,CLUSTER="${CLUSTER:-luh}" \
  "${SCRIPT_DIR}/train_all.sh"
