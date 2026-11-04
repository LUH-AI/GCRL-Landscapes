"""Export final-phase per-configuration success for the raw dot figure.

Seeds are aggregated with the IQM, which reproduces the paper's Table 1.

Usage: python export_configs.py <GCRL-Landscapes>/paper-metrics/data/eval_stats/antmaze-medium.csv
"""
import collections
import csv
import json
import statistics
import sys


def iqm(values):
    values = sorted(values)
    k = len(values) // 4
    return statistics.mean(values[k : len(values) - k])


seeds = collections.defaultdict(list)
hps = {}
for row in csv.DictReader(open(sys.argv[1])):
    if row["phase"] != "4" or row["algo"] not in ("GCIQL", "GCIVL"):
        continue
    key = (row["algo"], int(row["config"]))
    seeds[key].append(float(row["success"]))
    hps[key] = (float(row["lr"]), float(row["alpha"]))

out = collections.defaultdict(list)
for key in sorted(seeds):
    lr, alpha = hps[key]
    out[key[0]].append([float(f"{lr:.3g}"), round(alpha, 2), round(iqm(seeds[key]), 3)])

for algo, rows in out.items():
    scores = [r[2] for r in rows]
    print(algo, len(rows), "max", max(scores), "at_least_0.5", sum(s >= 0.5 for s in scores),
          file=sys.stderr)
print(json.dumps(out, separators=(",", ":")))
