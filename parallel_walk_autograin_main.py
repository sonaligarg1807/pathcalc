#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Parallel biased random walk; grain y-ranges are auto-detected from grain .gro files instead of hardcoded."""

import os
import time
import multiprocessing as mp

from pathcalc import gro, path, resid, mapped_resids

# ------------------ Global parameters ------------------
MAX_PATH_LEN = 25          # maximum number of residues in one sampled path
USE_RANDOM_RESIDS = True   # True -> use random_resids.txt, False -> use mapped g1+g2 resids
N_PROCESSES = 32           # number of parallel workers


def process_source_resid(
    source_resid: int,
    groFilePath: str,
    topFilePath: str,
    gmxPath: str,
    mdpFilePath: str,
    y_ranges,
    cutoff: float,
    root_dir: str,
    ham_file: str,
    all_coms,          # precomputed in parent
    all_resids,        # precomputed in parent
):
    source_resid = int(source_resid)
    subdir = os.path.join(root_dir, f"SR_{source_resid}")
    os.makedirs(subdir, exist_ok=True)
    os.chdir(subdir)

    print(f"\n=== Source resid {source_resid} ===")
    print(f"Subdirectory for source resid {source_resid} generated: {subdir}")

    # fresh gro object per worker
    pen_gro_local = gro(groFilePath)

    pathsample = path.PathFinder(
        pen_gro_local,
        all_coms,
        all_resids,
        ham_file,
        topFilePath,
        gmxPath,
        mdpFilePath,
        y_ranges,
        cutoff,
    )

    first_source_resid = source_resid
    sampled_paths = [first_source_resid]
    visited_resids = [first_source_resid]

    first_range = pathsample.check_y_range(first_source_resid)
    if first_range == "out_of_range":
        print(f"Residue {first_source_resid} is out of range. Skipping.")
        os.chdir(root_dir)
        return

    print(f"Starting biased walk from resid {first_source_resid} in {first_range}")
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

        selected_neighbor = pathsample.select_next_neighbor(
            probabilities,
            source_resid,
            first_source_resid,
            visited_resids,
            sampled_paths,
        )

        if selected_neighbor is None:
            print("No valid next residue found. Ending walk.")
            break

        visited_resids.append(selected_neighbor)
        sampled_paths.append(selected_neighbor)

        selected_range = pathsample.check_y_range(selected_neighbor)

        if not crossed:
            if (
                (first_range == "in_range_1" and selected_range == "in_range_2")
                or (first_range == "in_range_2" and selected_range == "in_range_1")
            ):
                crossed = True
                print(
                    f"Residue {selected_neighbor} has crossed from "
                    f"{first_range} to {selected_range}."
                )

        if len(sampled_paths) >= MAX_PATH_LEN:
            print(f"Reached max path length {MAX_PATH_LEN}. Ending walk.")
            break

        source_resid = selected_neighbor

    output_file = f"final_sampled_paths_{first_source_resid}.txt"
    with open(output_file, "w") as f:
        for resid_id in sampled_paths:
            f.write(f"{resid_id}\n")

    print(f"Final sampled paths saved to {output_file}")
    print(f"Sampled path for source_resid {first_source_resid}: {sampled_paths}")

    os.chdir(root_dir)


def main():
    topFilePath = (
        "/data/sgarg/pentacene/gb_pen_schellhammer/"
        "final_working_str/calculations/topo_qm/pen-esp.top"
    )
    groFilePath = (
        "/data/sgarg/pentacene/gb_pen_schellhammer/"
        "final_working_str/calculations/b/b45/slab_eq/nvt.gro"
    )
    mdpFilePath = "/data/sgarg/pentacene/pathcalc/inps/namd-qmmm.mdp"
    gmxPath = "/home/fghalami/GMX/Gromacs-SH/build_test/src/kernel"
    ham_file = "TB_HAMILTONIAN.xvg"

    g1FilePath_old = (
        "/data/sgarg/pentacene/gb_pen_schellhammer/final_working_str/"
        "structural_analysis_GB/b45/slab_01/slab_01_seg02_y67.57_g1.gro"
    )
    g2FilePath_old = (
        "/data/sgarg/pentacene/gb_pen_schellhammer/final_working_str/"
        "structural_analysis_GB/b45/slab_01/slab_01_seg02_y67.57_g2.gro"
    )

    mapped_out_dir = "./mapped_nvt"
    cutoff = 0.60
    mapped_cutoff = 0.3

    pen_gro_full = gro(groFilePath)

    start = time.time()
    MP_coms = pen_gro_full.MP_resCOMs
    all_resids = [int(r) for r in pen_gro_full.allRes]
    print(
        f"Calculated center of mass for {len(MP_coms)} residues "
        f"in {time.time() - start:.3f} seconds"
    )

    coms_dict = {r: MP_coms[i] for i, r in enumerate(all_resids)}

    g1_resids_nvt, g2_resids_nvt = mapped_resids.map_and_write_grains_to_nvt(
        g1FilePath_old,
        g2FilePath_old,
        groFilePath,
        mapped_out_dir,
        cutoff=mapped_cutoff,
        ref_resids=all_resids,
        ref_coms=MP_coms,
    )

    g1_ymin, g1_ymax = mapped_resids.y_range_from_resids(g1_resids_nvt, coms_dict)
    g2_ymin, g2_ymax = mapped_resids.y_range_from_resids(g2_resids_nvt, coms_dict)

    # ====== IMPORTANT FIX: enforce ordering so range_1 is lower-y grain ======
    g1_center = 0.5 * (g1_ymin + g1_ymax)
    g2_center = 0.5 * (g2_ymin + g2_ymax)

    print(f"Raw grain1 y-range: [{g1_ymin:.3f}, {g1_ymax:.3f}] (center {g1_center:.3f})")
    print(f"Raw grain2 y-range: [{g2_ymin:.3f}, {g2_ymax:.3f}] (center {g2_center:.3f})")

    if g1_center <= g2_center:
        # grain1 is lower (or equal) in y → becomes range_1
        y_ranges = ([g1_ymin, g1_ymax], [g2_ymin, g2_ymax])
        print("Assigned: range_1 = lower-y grain (g1), range_2 = upper-y grain (g2).")
    else:
        # grain2 is lower in y → becomes range_1
        y_ranges = ([g2_ymin, g2_ymax], [g1_ymin, g1_ymax])
        print("Assigned: range_1 = lower-y grain (g2), range_2 = upper-y grain (g1).")

    print(f"Final range_1 (lower) y-range: {y_ranges[0]}")
    print(f"Final range_2 (upper) y-range: {y_ranges[1]}")
    # ========================================================================

    root_dir = os.getcwd()

    if USE_RANDOM_RESIDS:
        print("\n[MODE] Using random_resids.txt as source_resids")

        natoms = 36

        if os.path.exists("random_resids.txt"):
            print("random_resids.txt already exists — skipping extraction.")
        else:
            print("random_resids.txt not found — extracting random source residues.")
            selector = resid.ResidExtractor(pen_gro_full, natoms, all_resids, MP_coms)
            selector.extract_source_resids(
                axes_count=3,
                axes="x,y,z",
                selection_choice="no",
                y_range_choice=2,
                x_limits=(2.0, 14.0),
                y_limits_range_1=(1.0, 3.0),
                y_limits_range_2=[(4.0, 5.5), (7.0, 8.5)],
                z_limits=(7.0, 8.5),
                num_to_select=2,
                output_file="random_resids.txt",
                gro_output_file="random_resids.gro",
            )

        with open("random_resids.txt", "r") as f:
            lines = f.readlines()
        source_resids = [
            int(line.split()[0]) for line in lines if not line.startswith("#")
        ]
        print(f"Total source_resids from random_resids.txt: {len(source_resids)}")

    else:
        print("\n[MODE] Using union of mapped g1+g2 resids as source_resids")
        source_resids = sorted(set(g1_resids_nvt) | set(g2_resids_nvt))
        print(f"Total source_resids from g1+g2 (NVT): {len(source_resids)}")

    args = [
        (
            r,
            groFilePath,
            topFilePath,
            gmxPath,
            mdpFilePath,
            y_ranges,
            cutoff,
            root_dir,
            ham_file,
            MP_coms,
            all_resids,
        )
        for r in source_resids
    ]

    print(f"\nStarting multiprocessing with {N_PROCESSES} processes...")
    with mp.Pool(processes=N_PROCESSES) as pool:
        pool.starmap(process_source_resid, args)

    print("\nAll source_resids processed.")


if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
    main()
