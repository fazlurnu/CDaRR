"""
Compute summary statistics from all results/*fulldpsi* CPA JSON files.

For each (folder, file, crossing_angle) entry:
  - mean_ipr_via_overall    : the overall_ipr value (already mean of ipr list)
  - mean_ipr_via_individual : mean of individual ipr values
  - lowest_ipr              : min of individual ipr values
  - lowest_worst_cpa        : lowest per-sample worst CPA across all samples
  - highest_worst_cpa       : highest per-sample worst CPA across all samples
  - median_worst_cpa        : median of per-sample worst CPAs

Per-sample worst CPA = min(worst_cpa[i]) — the closest any pair came in that sample.

Saves results/summary.csv (one row per entry + an OVERALL aggregate row).
"""

import json
import os
import glob
import csv

import numpy as np

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_CSV = os.path.join(RESULTS_ROOT, "summary.csv")

CROSSING_ANGLES = list(range(2, 43, 2)) + list(range(45, 181, 5))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def per_sample_worst(worst_cpa_list):
    """For each sample, get the minimum CPA across all pairs."""
    return np.array([np.min(pairs) for pairs in worst_cpa_list])


def stats_for_entry(data, angle):
    key = str(angle)
    if key not in data:
        return None
    entry = data[key]

    ipr_list = np.array(entry["ipr"])
    wcpa_per_sample = per_sample_worst(entry["worst_cpa"])

    return {
        "crossing_angle": angle,
        "mean_ipr_via_overall":    entry["overall_ipr"],
        "mean_ipr_via_individual": float(np.mean(ipr_list)),
        "lowest_ipr":              float(np.min(ipr_list)),
        "lowest_worst_cpa":        float(np.min(wcpa_per_sample)),
        "highest_worst_cpa":       float(np.max(wcpa_per_sample)),
        "median_worst_cpa":        float(np.median(wcpa_per_sample)),
    }


def main():
    # Find all *fulldpsi* folders and their *_cpa.json files
    pattern = os.path.join(RESULTS_ROOT, "*fulldpsi*", "*_cpa.json")
    cpa_files = sorted(glob.glob(pattern))

    if not cpa_files:
        print(f"No files found matching: {pattern}")
        return

    rows = []

    # Accumulators for the overall aggregate row
    all_overall_ipr = []
    all_individual_ipr = []
    all_worst_cpa = []

    for fpath in cpa_files:
        folder = os.path.basename(os.path.dirname(fpath))
        fname  = os.path.basename(fpath)
        data   = load_json(fpath)

        for angle in CROSSING_ANGLES:
            s = stats_for_entry(data, angle)
            if s is None:
                continue

            rows.append({
                "folder":                  folder,
                "file":                    fname,
                "crossing_angle":          s["crossing_angle"],
                "mean_ipr_via_overall":    round(s["mean_ipr_via_overall"],    6),
                "mean_ipr_via_individual": round(s["mean_ipr_via_individual"], 6),
                "lowest_ipr":              round(s["lowest_ipr"],              6),
                "lowest_worst_cpa":        round(s["lowest_worst_cpa"],        4),
                "highest_worst_cpa":       round(s["highest_worst_cpa"],       4),
                "median_worst_cpa":        round(s["median_worst_cpa"],        4),
            })

            # Accumulate for global aggregate
            all_overall_ipr.append(s["mean_ipr_via_overall"])

            entry = data[str(angle)]
            all_individual_ipr.extend(entry["ipr"])
            all_worst_cpa.extend(
                [np.min(pairs) for pairs in entry["worst_cpa"]]
            )

    if not rows:
        print("No data found.")
        return

    # Build the OVERALL aggregate row
    all_worst_cpa_arr = np.array(all_worst_cpa)
    overall_row = {
        "folder":                  "OVERALL",
        "file":                    "ALL",
        "crossing_angle":          "ALL",
        "mean_ipr_via_overall":    round(float(np.mean(all_overall_ipr)),    6),
        "mean_ipr_via_individual": round(float(np.mean(all_individual_ipr)), 6),
        "lowest_ipr":              round(float(np.min(all_individual_ipr)),  6),
        "lowest_worst_cpa":        round(float(np.min(all_worst_cpa_arr)),   4),
        "highest_worst_cpa":       round(float(np.max(all_worst_cpa_arr)),   4),
        "median_worst_cpa":        round(float(np.median(all_worst_cpa_arr)),4),
    }

    fieldnames = [
        "folder", "file", "crossing_angle",
        "mean_ipr_via_overall", "mean_ipr_via_individual", "lowest_ipr",
        "lowest_worst_cpa", "highest_worst_cpa", "median_worst_cpa",
    ]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(overall_row)

    print(f"Saved {len(rows)} rows + 1 OVERALL row to: {OUTPUT_CSV}")
    print("\n--- OVERALL ---")
    for k, v in overall_row.items():
        if k not in ("folder", "file", "crossing_angle"):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
