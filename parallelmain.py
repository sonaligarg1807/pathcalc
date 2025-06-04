from pathcalc import gro, path, resid
import os
import time
import multiprocessing as mp
from collections import defaultdict

def process_source_resid(source_resid, pen_gro, all_coms, all_resids,
                        ham_file, topFilePath, gmxPath, mdpFilePath, y_ranges, root_dir):
    source_resid = int(source_resid)
    print(f"Starting biased random walk from source resid {source_resid}")

    # Create subdir and switch to it
    subdir = os.path.join(root_dir, f"SR_{source_resid}")
    os.makedirs(subdir, exist_ok=True)
    os.chdir(subdir)
    print(f"subdirectory for source resid {source_resid} generated")
    
    pathsample = path.PathFinder(pen_gro, all_coms, all_resids, ham_file, topFilePath, gmxPath, mdpFilePath, y_ranges)

    first_source_resid = source_resid
    sampled_paths = [first_source_resid]
    print(f"First source_resid: {first_source_resid}")
    print(f"sampled path: {sampled_paths}")
    visited_resids = []
    visited_resids.append(first_source_resid)
    print(f"visited residue is {visited_resids}")

    first_range = pathsample.check_y_range(first_source_resid)
    if first_range == "out_of_range":
        print(f"Residue {first_source_resid} is out of range. Skipping.")
        os.chdir(root_dir)
        return
    
    while True:
        extracted_cpl_values = defaultdict(list)
        cpl_values = pathsample.avg_cpl(ham_file, source_resid, topFilePath, gmxPath, mdpFilePath)
        if cpl_values:
            extracted_cpl_values[source_resid] = cpl_values
            print(f"extracted coupling values for source resid {source_resid}: {cpl_values}")
        else:
            extracted_cpl_values[source_resid] = None

        probabilities = pathsample.probabilities(cpl_values)
        print(f"calculated probabilities for source resid {source_resid}: {probabilities}")
        
        selected_neighbor = pathsample.select_next_neighbor(probabilities, source_resid, first_source_resid,
                                                            visited_resids, sampled_paths)
        print(f"selected neighbor is {selected_neighbor}")
        
        if selected_neighbor is None:
            print("No valid next residue found. Ending walk.")
            break

        visited_resids.append(selected_neighbor)
        sampled_paths.append(selected_neighbor)

        selected_range = pathsample.check_y_range(selected_neighbor)
        if (first_range == "in_range_1" and selected_range == 'in_range_2') or \
            (first_range == "in_range_2" and selected_range == 'in_range_1'):
            print(f"Residue {selected_neighbor} has crossed the range. Ending walk.")
            break
        
        # Continue iteration with selected_neighbor as the new source_resid
        print(f"Continuing with selected_neighbor {selected_neighbor} as the new source_resid.")
        source_resid = selected_neighbor

    #write the final sampled paths to a text file
    output_file = f"final_sampled_paths_{first_source_resid}.txt"
    with open(output_file, "w") as f:
        for resid in sampled_paths:
            f.write(f"{resid}\n")
    print (f"Final sampled paths saved to {output_file}")
    
    print(f"Sampled paths for source_resid {first_source_resid}: {sampled_paths}")

    os.chdir(root_dir)
print("\nAll source_resids processed.")
    
def main():
    # Loading files and COMs as before
    topFilePath = "/data/sgarg/pentacene/sampling_qm_zone_algorithm/gb-pen/input_files/pen-esp.top"
    groFilePath = "/data/sgarg/pentacene/sampling_qm_zone_algorithm/gb-pen/lattice_orientation/b/b30/trajectories_1000/traj.gro"
    mdpFilePath = "/data/sgarg/pentacene/pathcalc/inps/namd-qmmm.mdp"
    gmxPath = "/data/fghalami/gromacs-sh-old_Eik/test_plumed/gromacs-sh-old/COUPLED-DYNAMICS/build-tomas-jan2023/src/kernel"
    ham_file = "TB_HAMILTONIAN.xvg"
    y_ranges = ([2.5, 4.0] , [4.0, 5.5])
    
    #loading topology and gro file
    pen_gro = gro(groFilePath)
    print(f"loaded gro file: {pen_gro} with {len(pen_gro.allRes)} residues")
    
    #calculating COM for all resids
    start= time.time()
    all_coms = pen_gro.MP_resCOMs
    print(f"calculated center of mass for {len(all_coms)} residues in {time.time()-start} seconds")
    all_resids = list(pen_gro.allRes)

    #extracting random resids
    natoms = 36
    start = time.time()
    selector = resid.ResidExtractor(pen_gro, natoms, all_resids, all_coms)
    selected_resids = selector.extract_source_resids(
        axes_count=3, axes="x,y,z", selection_choice="no", y_range_choice=2,
        x_limits=(2.0, 14.0), y_limits_range_1=(1.0, 3.0),
        y_limits_range_2=[(4.0, 5.5), (7.0, 8.5)], z_limits=(7.0, 8.5),
        num_to_select=5, output_file="random_resids.txt"
    )
    print(f"Time taken to select residues: {time.time() - start} seconds")
    print(f"Selected {len(selected_resids)} residues based on the specified criteria.")
    
    #reading source_resid values from file
    with open("random_resids.txt", "r") as f:
        lines = f.readlines()
    source_resids = [int(line.split()[0]) for line in lines if not line.startswith("#")]

    root_dir = os.getcwd()
    
    args = [(resid, pen_gro, all_coms, all_resids, ham_file, topFilePath, gmxPath, mdpFilePath, y_ranges, root_dir)
            for resid in source_resids]

    with mp.Pool(processes=32) as pool:
        pool.starmap(process_source_resid, args)

    print("\nAll source_resids processed.")
    
if __name__ == "__main__":
    main()
