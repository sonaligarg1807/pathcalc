#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import time
import multiprocessing as mp

print("Appending path: /home/sgarg/pathcalc/")
sys.path.append("/home/sgarg/pathcalc/")

from pathcalc import gro, a_series_path, resid

# ------------------ Global parameters ------------------
MAX_PATH_LEN = 25
USE_RANDOM_RESIDS = True
N_PROCESSES = 32

# Bias axis
BIAS_AXIS = "y"
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}[BIAS_AXIS]

# ------------------ Cutoffs ------------------
NORMAL_CUTOFF = 0.68
RESCUE_SEARCH_RADIUS = 1.5
RESCUE_TOO_FAR_CUTOFF = 1.2

# Rescue behavior
RESCUE_TOP_K = 10
RESCUE_FORWARD_EPS = 0.5
RESCUE_MAX_NEIGHBORS = 28


def axis_range_from_resids(resid_list, coms_dict, axis_index):
    coords = [coms_dict[r][axis_index] for r in resid_list]
    return min(coords), max(coords)


def process_source_resid(
    source_resid,
    groFilePath,
    topFilePath,
    gmxPath,
    mdpFilePath,
    axis_ranges,
    normal_cutoff,
    rescue_search_radius,
    rescue_too_far_cutoff,
    rescue_max_neighbors,
    root_dir,
    ham_file,
    all_coms,
    all_resids,
):
    source_resid = int(source_resid)
    subdir = os.path.join(root_dir, f"SR_{source_resid}")
    os.makedirs(subdir, exist_ok=True)

    prev_dir = os.getcwd()
    os.chdir(subdir)

    try:
        print(f"\n=== Source resid {source_resid} ===")

        pen_gro_local = gro(groFilePath)

        pathsample = a_series_path.PathFinder(
            pen_gro_local,
            all_coms,
            all_resids,
            ham_file,
            topFilePath,
            gmxPath,
            mdpFilePath,
            axis_ranges,
            normal_cutoff,
            bias_axis=BIAS_AXIS,
            rescue_radius=rescue_search_radius,
            rescue_cutoff=rescue_too_far_cutoff,
            rescue_top_k=RESCUE_TOP_K,
            rescue_forward_eps=RESCUE_FORWARD_EPS,
            rescue_max_neighbors=rescue_max_neighbors,
            slab_normal_axis="z",
        )

        first_source_resid = source_resid
        sampled_paths = [first_source_resid]
        visited_resids = {first_source_resid}

        summary_rows = []
        step_idx = 0

        first_range = pathsample.check_axis_range(first_source_resid)
        if first_range == "out_of_range":
            print(f"Residue {first_source_resid} out of range. Skipping.")
            return

        print(f"Starting walk from resid {first_source_resid} in {first_range}")

        crossed = False

        while True:
            cpl_values = pathsample.avg_cpl(
                ham_file,
                source_resid,
                topFilePath,
                gmxPath,
                mdpFilePath,
            )

            probabilities = pathsample.probabilities(cpl_values)

            result = pathsample.select_next_neighbor(
                probabilities,
                source_resid,
                first_source_resid,
                visited_resids,
                sampled_paths,
            )

            if result is None:
                print("No valid next residue found. Ending walk.")
                break

            selected_neighbor, step_info = result

            step_idx += 1
            step_info["step"] = step_idx
            summary_rows.append(step_info)

            visited_resids.add(selected_neighbor)
            sampled_paths.append(selected_neighbor)

            selected_range = pathsample.check_axis_range(selected_neighbor)
            if not crossed:
                if (
                    (first_range == "in_range_1" and selected_range == "in_range_2")
                    or (first_range == "in_range_2" and selected_range == "in_range_1")
                ):
                    crossed = True
                    print(f"Residue {selected_neighbor} crossed GB.")

            if len(sampled_paths) >= MAX_PATH_LEN:
                print(f"Reached max path length {MAX_PATH_LEN}.")
                break

            source_resid = selected_neighbor

        # -------- Save path --------
        with open(f"final_sampled_paths_{first_source_resid}.txt", "w") as f:
            for r in sampled_paths:
                f.write(f"{r}\n")

        # -------- Save summary --------
        summary_file = f"path_summary_{first_source_resid}.csv"
        header = [
            "step",
            "mode",
            "source_resid",
            "target_resid",
            "distance_nm",
            "axis_progress_nm",
            "source_grain",
            "target_grain",
            "crossed",
        ]

        with open(summary_file, "w") as f:
            f.write(",".join(header) + "\n")
            for row in summary_rows:
                f.write(",".join(str(row[h]) for h in header) + "\n")

        print(f"Saved path + summary for SR_{first_source_resid}")

    finally:
        os.chdir(prev_dir)


def main():
    topFilePath = "/data/sgarg/pentacene/gb_pen_schellhammer/final_working_str/calculations/topo_qm/pen-esp.top"
    groFilePath = "/data/sgarg/pentacene/gb_pen_schellhammer/final_working_str/calculations/a/a90/qm_paths/a_series_path_test/slab_renumber.gro"
    mdpFilePath = "/data/sgarg/pentacene/pathcalc/inps/namd-qmmm.mdp"
    gmxPath = "/home/fghalami/GMX/Gromacs-SH/build_test/src/kernel"
    ham_file = "TB_HAMILTONIAN.xvg"

    g1_trimmed_gro = "/data/sgarg/pentacene/gb_pen_schellhammer/final_working_str/calculations/a/a90/qm_paths/a_series_path_test/mapped_grains/slab_a90_seg01_local112.07_g2.gro_trimmed.gro"
    g2_trimmed_gro = "/data/sgarg/pentacene/gb_pen_schellhammer/final_working_str/calculations/a/a90/qm_paths/a_series_path_test/mapped_grains/slab_a90_seg02_local131.96_g1.gro_trimmed.gro"

    pen_gro_full = gro(groFilePath)
    MP_coms = pen_gro_full.MP_resCOMs
    all_resids = [int(r) for r in pen_gro_full.allRes]
    coms_dict = {r: MP_coms[i] for i, r in enumerate(all_resids)}

    g1_resids = sorted({int(r) for r in gro(g1_trimmed_gro).allRes})
    g2_resids = sorted({int(r) for r in gro(g2_trimmed_gro).allRes})

    g1_min, g1_max = axis_range_from_resids(g1_resids, coms_dict, AXIS_INDEX)
    g2_min, g2_max = axis_range_from_resids(g2_resids, coms_dict, AXIS_INDEX)

    if (g1_min + g1_max) <= (g2_min + g2_max):
        axis_ranges = ([g1_min, g1_max], [g2_min, g2_max])
    else:
        axis_ranges = ([g2_min, g2_max], [g1_min, g1_max])

    root_dir = os.getcwd()

    if USE_RANDOM_RESIDS:
        random_file = os.path.join(root_dir, "random_resids.txt")
        with open(random_file) as f:
            source_resids = [int(l.split()[0]) for l in f if not l.startswith("#")]
    else:
        source_resids = sorted(set(g1_resids) | set(g2_resids))

    args = [
        (
            r,
            groFilePath,
            topFilePath,
            gmxPath,
            mdpFilePath,
            axis_ranges,
            NORMAL_CUTOFF,
            RESCUE_SEARCH_RADIUS,
            RESCUE_TOO_FAR_CUTOFF,
            RESCUE_MAX_NEIGHBORS,
            root_dir,
            ham_file,
            MP_coms,
            all_resids,
        )
        for r in source_resids
    ]

    with mp.Pool(processes=N_PROCESSES) as pool:
        pool.starmap(process_source_resid, args)


if __name__ == "__main__":
    main()