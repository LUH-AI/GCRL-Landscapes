#!/usr/bin/env bash
# Additional-experiments campaign (PAPER/rebuttal/additional-exp): the arms that
# need no repo change and can be submitted immediately.
#
#   A6a  GCBC proposal policies      -> unblocks the SfBC rejection-sampling
#                                       re-extraction (A6); GCBC-alone success
#                                       is itself a quotable anchor number.
#   A1   Straight-through control    -> 7VBe W1, the in-thread promise: one
#                                       uninterrupted run at the published total
#                                       budget, no phase resets.
#
# A3 (DDPG+BC on Cube) is deliberately absent: Malte is running that arm.
#
# A1 needs no --single_phase flag: phases are computed as
# int(pct/100 * final_phase) in submission.py, so --phase_percentages 100 with
# --final_step_is_phase is exactly one phase covering the whole budget. The
# published per-phase steps are passed as --extra_eval_steps so the learning
# curves stay comparable to the phased campaign.
#
# Usage:
#   bash rebuttal_additional_exp.sh --dry-run
#   bash rebuttal_additional_exp.sh --arms a6a
#   bash rebuttal_additional_exp.sh --arms a3
#
# Env overrides: CLUSTER, PARTITION, RESERVATION, NSEEDS, LOGROOT, PYTHON

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck disable=SC1090
source "${SCRIPT_DIR}/configs/cluster/${CLUSTER:-luh}.sh"

PYTHON="${PYTHON:-${SCRIPT_DIR}/.venv/bin/python}"
LOGROOT="${LOGROOT:-${SCRIPT_DIR}/logs-additional-exp}"
NSEEDS="${NSEEDS:-3}"
EVAL_EPISODES="10"

TASKS_PER_NODE_TOTAL="${TASKS_PER_NODE_TOTAL:-4}"
TASKS_PER_NODE_PARALLEL="${TASKS_PER_NODE_PARALLEL:-4}"
MEM_PER_CPU="${MEM_PER_CPU:-3G}"

PARTITION="${PARTITION:-$CLUSTER_PARTITION}"
RESERVATION="${RESERVATION:-$CLUSTER_RESERVATION}"
[[ "$RESERVATION" == "none" ]] && RESERVATION=""

# ---------------------------------------------------------------------------
# Per-agent final-phase step budgets, read from the published campaigns'
# info.toml under $REFROOT. Matching the budget is what makes the ceiling
# comparison against the paper's AWR numbers meaningful.
#
#   cube-single-play-v0      GCIQL 949583  CRL 315517  QRL 809000  GCIVL  46888
#   antmaze-medium-navigate  GCIQL 820277  CRL 159090  QRL 310750  GCIVL 865793
# ---------------------------------------------------------------------------
REFROOT="${REFROOT:-/project/NHWP25179/SSRL-Landscapes}"

# name|group|agent|dataset|actor_loss|final_phase|nconfigs|min_per_mill|extra_eval_steps
#
# actor_loss "-" means "leave the agent default" (GCBC).
# extra_eval_steps "-" means none.
declare -a -r ARMS=(
  # --- A6a: BC proposal policies. 1 config x 3 seeds, supervised, hours. -----
  # 1e6 steps is OGBench's own GCBC protocol, so the N=1 column can be sanity
  # checked against their published GCBC baselines.
  "a6a-gcbc-antmaze-medium|a6a|GCBC|antmaze-medium-navigate-v0|-|1000000|1|300|-"
  "a6a-gcbc-cube|a6a|GCBC|cube-single-play-v0|-|1000000|1|300|-"

  # --- A3 (DDPG+BC on Cube) is NOT in this table on purpose: Malte is already
  # running it. Do not add it here — a second submission would burn a full
  # cube-budget arm per agent and produce a conflicting design.

  # --- A1: straight-through, no phase resets. Paired to the published --------
  # design via inject_reference_configs.py so configuration_i is the same
  # (lr, alpha) point in both protocols -> the rank correlation is meaningful.
  # GCIQL (narrow near peak) and QRL (broad) for maximal contrast.
  "a1-straight-gciql-antmaze-medium|a1|GCIQL|antmaze-medium-navigate-v0|awr|820277|64|300|205069 410138 615208"
  "a1-straight-qrl-antmaze-medium|a1|QRL|antmaze-medium-navigate-v0|awr|310750|64|300|77687 155375 233062"
)

# Reference trees for the paired A1 injection (A3/A6a are unpaired by design).
declare -A -r REF_TREE=(
  ["a1-straight-gciql-antmaze-medium"]="logs-advantage-antmaze-medium-fixed-batch/GCIQL_antmaze-medium-navigate-v0_64c_lr-alpha"
  ["a1-straight-qrl-antmaze-medium"]="logs-advantage-antmaze-medium-fixed-batch/QRL_antmaze-medium-navigate-v0_64c_lr-alpha"
)

DRY_RUN=0
SELECTED=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --arms) SELECTED="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

run() {
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '  %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

[[ $DRY_RUN -eq 1 ]] || cluster_load_env 2>/dev/null || true

mkdir -p "$LOGROOT"
MANIFEST="${LOGROOT}/campaign_manifest.tsv"
if [[ $DRY_RUN -eq 0 && ! -f "$MANIFEST" ]]; then
  printf 'arm\tgroup\tagent\tdataset\tactor_loss\tfinal_phase\tnconfigs\tlogdir\tsubmitted_at\n' > "$MANIFEST"
fi

echo "=== additional-experiments campaign ==="
echo "partition   : ${PARTITION}"
echo "reservation : ${RESERVATION:-<none>}"
echo "seeds       : ${NSEEDS}"
echo "logroot     : ${LOGROOT}"
echo

for arm_spec in "${ARMS[@]}"; do
  IFS='|' read -r name group agent dataset actor_loss final_phase nconfigs min_per_mill extra_eval <<<"$arm_spec"

  if [[ -n "$SELECTED" ]] && [[ ",${SELECTED}," != *",${group},"* ]] && [[ ",${SELECTED}," != *",${name},"* ]]; then
    continue
  fi

  arm_root="${LOGROOT}/${name}"
  hp_suffix="lr-alpha"
  HYPERPARAMETERS="lr alpha"
  if [[ "$agent" == "GCBC" ]]; then
    # GCBC has no alpha: its only swept axis is lr, and with n_configurations=1
    # run_setup takes the adapted default config anyway.
    HYPERPARAMETERS="lr"
    hp_suffix="lr"
  fi
  full_log_dir="${arm_root}/${agent}_${dataset}_${nconfigs}c_${hp_suffix}"

  echo "--- ${name}"
  echo "    agent=${agent} dataset=${dataset} actor_loss=${actor_loss} steps=${final_phase}"
  echo "    grid=${nconfigs} cfg x ${NSEEDS} seeds x 1 phase"
  echo "    logdir=${full_log_dir}"

  if [[ -d "$full_log_dir" && "${SUBMIT_ONLY:-0}" != "1" ]]; then
    echo "    SKIP: logdir exists (SUBMIT_ONLY=1 to re-submit without regenerating configs)"
    continue
  fi

  if [[ ! -d "$full_log_dir" ]]; then
    setup_args=(
      -m gcrl_landscapes.main setup
      --agent "$agent"
      --datasets "$dataset"
      --n_configurations "$nconfigs"
      --final_phase "$final_phase"
      --phase_percentages 100
      --eval_episodes "$EVAL_EPISODES"
      --hyperparameters $HYPERPARAMETERS
      --logdir "$full_log_dir"
      --final_step_is_phase
    )
    [[ "$actor_loss" != "-" ]] && setup_args+=(--actor_loss "$actor_loss")
    [[ "$extra_eval" != "-" ]] && setup_args+=(--extra_eval_steps $extra_eval)
    run "$PYTHON" "${setup_args[@]}"

    ref_rel="${REF_TREE[$name]:-}"
    if [[ -n "$ref_rel" ]]; then
      if [[ -d "${REFROOT}/${ref_rel}/configurations" ]]; then
        run "$PYTHON" "${SCRIPT_DIR}/analysis/inject_reference_configs.py" \
          --target "$full_log_dir" \
          --reference "${REFROOT}/${ref_rel}" \
          --fields lr alpha
      else
        echo "    ERROR: reference configs missing at ${REFROOT}/${ref_rel}" >&2
        echo "           A1 is only meaningful as a PAIRED design; not submitting." >&2
        exit 1
      fi
    fi
  fi

  run "$PYTHON" -m gcrl_landscapes.main submit \
    --logdir "$full_log_dir" \
    --n_seeds "$NSEEDS" \
    --tasks_per_node_total "$TASKS_PER_NODE_TOTAL" \
    --tasks_per_node_parallel "$TASKS_PER_NODE_PARALLEL" \
    --mem_per_cpu "$MEM_PER_CPU" \
    --jobname "${name}" \
    --partition "$PARTITION" \
    ${RESERVATION:+--reservation "$RESERVATION"} \
    ${CLUSTER_GRES:+--gres "$CLUSTER_GRES"} \
    --min_per_mill_steps "$min_per_mill"

  if [[ $DRY_RUN -eq 0 ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$name" "$group" "$agent" "$dataset" "$actor_loss" "$final_phase" \
      "$nconfigs" "$full_log_dir" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"
  fi
  echo
done

echo "=== done ==="
[[ $DRY_RUN -eq 1 ]] && echo "(dry run - nothing submitted)"
exit 0
