#!/usr/bin/env bash
# Reproduce every numeric table in the current paper draft.
#
# Each script in scripts/ corresponds to one paper table; outputs land as
# CSVs in outputs/<table_id>.csv.  See README.md for the table-to-script
# mapping and the data prerequisites.
#
# Order matters in two places:
#   - t2 reads outputs/t1_final_landscape_ci.csv (for the Max Success column),
#     so t1 must run first.
#   - t3 and t4 share an on-disk merged-DataFrame cache built by the first
#     of them, so running them back-to-back is cheap.
#
# Usage:
#   cd paper-metrics
#   ./reproduce_all.sh           # all 10 scripts
#   ./reproduce_all.sh main      # only main-paper tables (t1, t2, t3, t4)
#   ./reproduce_all.sh appendix  # only robustness-appendix tables (a2..a7)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Path to the project venv. Override via PAPER_METRICS_PYTHON if needed.
PYTHON="${PAPER_METRICS_PYTHON:-/Users/adityamohan/git/GCRL/GCRL-Landscapes/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
    echo "Python interpreter not found at ${PYTHON}." >&2
    echo "Set PAPER_METRICS_PYTHON to a venv that has the gcrl_landscapes package + pyarrow + sklearn + scipy." >&2
    exit 1
fi

mode="${1:-all}"

run() {
    local label="$1"; local script="$2"
    echo
    echo "▶  ${label}  →  scripts/${script}"
    "${PYTHON}" "scripts/${script}"
}

case "${mode}" in
    main|all)
        run "T1  tab:final_landscape_ci"  t1_final_landscape_ci.py
        run "T2  tab:compact_diagnostics" t2_compact_diagnostics.py
        run "T3  tab:antmaze_ess_phase4 / tab:antmaze_ess_full" t3_antmaze_ess.py
        run "T4  tab:adv_corr"            t4_adv_corr.py
        ;;
esac

case "${mode}" in
    appendix|all)
        run "A2  tab:seed_variance_summary"            a2_seed_variance.py
        run "A3  tab:phase_mobility"                   a3_phase_mobility.py
        run "A4  tab:subsample_stability_close_pairs"  a4_subsample_stability.py
        run "A5  tab:wmax_sensitivity"                 a5_wmax_sensitivity.py
        run "A6  tab:adv_norm_sensitivity"             a6_adv_norm_sensitivity.py
        run "A7  tab:sensitivity_summary"              a7_sensitivity_summary.py
        ;;
esac

case "${mode}" in
    main|appendix|all) ;;
    *) echo "Unknown mode '${mode}'. Use 'main', 'appendix', or 'all'." >&2; exit 1 ;;
esac

echo
echo "Done. Outputs in $(pwd)/outputs/:"
ls -1 outputs/*.csv 2>/dev/null | sed 's|^|  |'
