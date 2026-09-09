"""Single-threaded biased random walk for QM path sampling across a grain boundary."""

import os
import time
from pathcalc import gro, path, resid

topFilePath = "/data/sgarg/pentacene/sampling_qm_zone_algorithm/gb-pen/input_files/pen-esp.top"
groFilePath = "/data/sgarg/pentacene/sampling_qm_zone_algorithm/gb-pen/lattice_orientation/b/b30/trajectories_1000/traj.gro"
mdpFilePath = "/data/sgarg/pentacene/pathcalc/inps/namd-qmmm.mdp"
gmxPath = "/data/fghalami/gromacs-sh-old_Eik/test_plumed/gromacs-sh-old/COUPLED-DYNAMICS/build-tomas-jan2023/src/kernel"
ham_file = "TB_HAMILTONIAN.xvg"

# load structure and compute center of mass for every residue
pen_gro = gro(groFilePath)

start = time.time()
all_coms = pen_gro.MP_resCOMs
print(f"calculated center of mass for {len(all_coms)} residues in {time.time()-start} seconds")
all_resids = list(pen_gro.allRes)

# pick random source residues to start walks from
natoms = 36
start = time.time()
selector = resid.ResidExtractor(pen_gro, natoms, all_resids, all_coms)
selector.extract_source_resids(
    axes_count=3, axes="x,y,z", selection_choice="no", y_range_choice=2,
    x_limits=(2.0, 14.0), y_limits_range_1=(1.0, 3.0),
    y_limits_range_2=[(4.0, 5.5), (7.0, 8.5)], z_limits=(7.0, 8.5),
    num_to_select=2, output_file="random_resids.txt",
)
print(f"Time taken to select residues: {time.time() - start} seconds")

with open("random_resids.txt", "r") as f:
    lines = f.readlines()
source_resids = [int(line.split()[0]) for line in lines if not line.startswith("#")]

root_dir = os.getcwd()
pathsample = path.PathFinder(pen_gro, all_coms, all_resids, ham_file, topFilePath, gmxPath, mdpFilePath)

for source_resid in source_resids:
    source_resid = int(source_resid)

    subdir = f"SR_{source_resid}"
    os.makedirs(subdir, exist_ok=True)
    os.chdir(subdir)

    first_source_resid = source_resid
    sampled_paths = [first_source_resid]
    visited_resids = [first_source_resid]

    first_range = pathsample.check_y_range(first_source_resid)
    if first_range == "out_of_range":
        print(f"Residue {first_source_resid} is out of range. Skipping.")
        os.chdir(root_dir)
        continue

    while True:
        cpl_values = pathsample.avg_cpl(ham_file, source_resid, topFilePath, gmxPath, mdpFilePath)
        probabilities = pathsample.probabilities(cpl_values)

        selected_neighbor = pathsample.select_next_neighbor(
            probabilities, source_resid, first_source_resid, visited_resids, sampled_paths
        )
        if selected_neighbor is None:
            print("No valid next residue found. Ending walk.")
            break

        visited_resids.append(selected_neighbor)
        sampled_paths.append(selected_neighbor)

        # stop once the walk has crossed from one grain's y-range into the other
        selected_range = pathsample.check_y_range(selected_neighbor)
        if (first_range == "in_range_1" and selected_range == "in_range_2") or \
           (first_range == "in_range_2" and selected_range == "in_range_1"):
            print(f"Residue {selected_neighbor} has crossed the range. Ending walk.")
            break

        source_resid = selected_neighbor

    output_file = f"final_sampled_paths_{first_source_resid}.txt"
    with open(output_file, "w") as f:
        for r in sampled_paths:
            f.write(f"{r}\n")

    os.chdir(root_dir)
