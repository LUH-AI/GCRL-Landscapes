#!/usr/bin/env python3
"""Export top-k config IDs per agent to JSON.

Run this first, then pass the JSON to AFR and advantage scripts for filtered analysis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import parse_args, get_top_k_configs
from gcrl_landscapes.util.data import load_or_compute
from gcrl_landscapes.evaluation.tabular import compute_merged_df


def main() -> None:
    zipfiles, _, top_k = parse_args()

    if top_k is None:
        raise ValueError("--top-k required (e.g., --top-k 5)")

    merged_results_df, _ = load_or_compute(zipfiles, compute_merged_df)
    top_configs = get_top_k_configs(merged_results_df, k=top_k)

    # Serialize config_index as int for clean JSON
    top_k_json = {
        agent: [int(c) for c in configs] for agent, configs in top_configs.items()
    }

    # Output next to the zip file
    output = zipfiles[0].parent / f"top_k_{top_k}_{zipfiles[0].stem}.json"
    output.write_text(json.dumps(top_k_json, indent=2))
    print(f"Exported top-k config IDs → {output}")


if __name__ == "__main__":
    main()
