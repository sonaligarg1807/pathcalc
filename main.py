from tintcalc import top, gro, sann, integral, resid
import os 
import time
import mdtraj as md
import shutil


topFilePath = "/data/sgarg/pentacene/sampling_qm_zone_algorithm/gb-pen/input_files/pen-esp.top"
tprFilePath = "/data/sgarg/pentacene/sampling_qm_zone_algorithm/gb-pen/lattice_orientation/b/b30/trajectories_1000/traj.tpr"
trrFilePath = "/data/sgarg/pentacene/sampling_qm_zone_algorithm/gb-pen/lattice_orientation/b/b30/trajectories_1000/traj.trr"
groFilePath = "/data/sgarg/pentacene/sampling_qm_zone_algorithm/gb-pen/lattice_orientation/b/b30/trajectories_1000/traj.gro"
# groFilePath = os.path.join(os.getcwd(), 'inps', 'ham.gro')
# groFilePath = "/data/sgarg/pentacene/sampling_qm_zone_algorithm/gb-pen/lattice_orientation/b/b30/trajectories_1000/TRAJSS1.gro"
mdpFilePath = os.path.join(os.getcwd(), 'inps', 'namd-qmmm.mdp')
ham_file = "TB_HAMILTONIAN.xvg"
log_file = "ham.log"
grompp_path = "/data/fghalami/gromacs-sh-old_Eik/test_plumed/gromacs-sh-old/COUPLED-DYNAMICS/build-tomas-jan2023/src/kernel/grompp"
mdrun_path = "/data/fghalami/gromacs-sh-old_Eik/test_plumed/gromacs-sh-old/COUPLED-DYNAMICS/build-tomas-jan2023/src/kernel/mdrun"

natoms = 36
start = time.time()
selector = resid.ResidExtractor(groFilePath, natoms)
selected_resids = selector.extract_source_resids(axes_count=3, axes="x,y,z", selection_choice="yes", y_range_choice=1, x_limits=(2.0, 14.0), y_limits_range_1=(5.0, 15.0), y_limits_range_2=[(5.5, 6.5), (11.0, 12.0)], z_limits=(7.5, 9.0), num_to_select=5, output_file="random_resids.txt")
print(f"Time taken to select residues: {time.time() - start} seconds")
print(f"Selected {len(selected_resids)} residues based on the specified criteria.")

#Load trajectory using mdtraj
print("Loading trajectory...")
traj = md.load(trrFilePath, top=groFilePath)
print(f"Trajectory loaded with {traj.n_frames} frames.")

# define snapshot selection parameters
n_snapshots = 10
snapshots_interval = int(traj.n_frames / n_snapshots)
selected_frames = list(range(0, traj.n_frames, snapshots_interval))[:n_snapshots]
print(f"Selected {len(selected_frames)} snapshots every {snapshots_interval} frames")

#reading source_resid values from file
with open("random_resids.txt", "r") as f:
    lines = f.readlines()
source_resids = [int(line.split()[0]) for line in lines if not line.startswith("#")]

#source_resid = 3099

#calculate COMs once using a reference gro file. Use this when the system is large
# reference_gro = gro(groFilePath)
# print(f"Loaded reference gro file with {reference_gro.natoms} atoms and {reference_gro.nRes} residues.")
# all_coms = reference_gro.resCOMs
# all_resids = list(reference_gro.allRes)

for source_resid in source_resids:
    source_dir = os.path.join(os.getcwd(), "results", f"SR_{source_resid}")
    os.makedirs(source_dir, exist_ok=True)
    
    #process each selected frame
    snapshot_avg_list = []
    for i, frame_idx in enumerate(selected_frames):
        frame = traj[frame_idx]
        snapshot_gro_file = os.path.join(source_dir, f'snapshot_{i}.gro')
        
        frame.save(snapshot_gro_file)
        print(f"Snapshot {i} saved to {snapshot_gro_file}")
    
        pen_top = top(topFilePath)
        print(f"Loaded topology")
        pen_gro = gro(snapshot_gro_file)
        # pen_gro = gro(groFilePath)
        print(f"Loaded snapshot with {pen_gro.natoms} atoms and {pen_gro.nRes} residues.")
        
        #finding neighbors of source residue
        print(f"\nProcessing snapshot {i}, frame {frame_idx}")
        start = time.time()
        all_coms = pen_gro.MP_resCOMs
        print(f"Calculated COMs for {len(all_coms)} residues.")
        all_resids = list(pen_gro.allRes)
        print(f"Time taken: {time.time() - start} seconds")
        print(f"Found {len(all_resids)} residues in the snapshot.")
        start = time.time()
        nn = pen_gro.cutAroundRes(source_resid, [1., 1., 1.])
        print(f"Time taken: {time.time() - start} seconds")
        #pen_gro.showIt(nn)
        
        start = time.time()
        #Initialize firstNN with subset
        nn_finder = sann.FirstNN(all_coms, all_resids, subset_resids=nn)
        neighbors = nn_finder.nearest_neighbors_asann(source_resid)
        print(f"Neighbors of residue 115: {neighbors}")
        print(f"Time taken: {time.time() - start} seconds")
        calc = integral.CouplingCalculator(
            gro_file=snapshot_gro_file,
            top_file=topFilePath,
            mdp_file=mdpFilePath,
            ham_file=ham_file,
            log_file=log_file,
            sub_dir=os.path.join(source_dir, f'subdir_{source_resid}_{i}'),
            grompp_path=grompp_path,
            mdrun_path=mdrun_path)
        # 
        extracted_coupling_values = {f"average_coupling_values_{source_resid}": {}}
    # 
        for target_resid in neighbors:
            print(f"Calculating coupling for residue {target_resid}...")
            calc.calculate_integral(source_resid=source_resid, target_resid=target_resid, extracted_coupling_values=extracted_coupling_values)
            print(f"Coupling calculation for residue {target_resid} completed.")
            # 
        snapshot_avg = calc.compute_snapshot_average(source_resid=source_resid, extracted_coupling_values=extracted_coupling_values)
        snapshot_avg_list.append(snapshot_avg)
        print(f"Snapshot average for frame {frame_idx}: {snapshot_avg:.4f}")
        # 
    calc.finalize_average(source_resid=source_resid, snapshot_avgs=snapshot_avg_list, source_dir=source_dir)
    
    # Optional: clean up everything except final_average_coupling_values.txt
    for fname in os.listdir(source_dir):
        if fname != "final_average_coupling_values.txt":
            fpath = os.path.join(source_dir, fname)
            if os.path.isdir(fpath):
                shutil.rmtree(fpath)
            else:
                os.remove(fpath)
    