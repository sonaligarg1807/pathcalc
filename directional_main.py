#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Biased path sampling across grain boundaries using parallel processing.

Performs biased random walks from source residues, attempting to cross
from one grain to another along the grain–grain connecting vector.
"""

import sys
import os
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict

print("Appending path: /home/sgarg/pathcalc/")
sys.path.append("/home/sgarg/pathcalc/")

from pathcalc import gro, directional_bias_path, resid


class PathSamplerConfig:
    """Configuration for path sampling."""
    
    def __init__(self, args):
        # File paths
        self.top_file = "/data/sgarg/pentacene/gb_pen_schellhammer/final_working_str/calculations/topo_qm/pen-esp.top"
        self.gro_file = "/data/sgarg/pentacene/gb_pen_schellhammer/final_working_str/calculations/nbc/bc90/qm_paths/slab_renumber.gro"
        self.mdp_file = "/data/sgarg/pentacene/pathcalc/inps/namd-qmmm.mdp"
        self.gmx_path = "/home/fghalami/GMX/Gromacs-SH/build_test/src/kernel"
        self.ham_file = "TB_HAMILTONIAN.xvg"
        
        # Grain files
        self.g1_trimmed = "/data/sgarg/pentacene/gb_pen_schellhammer/final_working_str/calculations/nbc/bc90/qm_paths/mapped_grains/g1_trimmed.gro"
        self.g2_trimmed = "/data/sgarg/pentacene/gb_pen_schellhammer/final_working_str/calculations/nbc/bc90/qm_paths/mapped_grains/g2_trimmed.gro"
        
        # Parameters from arguments
        self.max_path_len = args.max_path_len
        self.use_random_resids = args.use_random_resids
        self.n_processes = args.n_processes
        self.cutoff = args.cutoff
        self.natoms = args.natoms
        self.num_random_resids = args.num_random_resids

        # NEW: directional penalty parameters
        self.min_cos_forward = args.min_cos_forward
        self.direction_alpha = args.direction_alpha

        self.root_dir = os.getcwd()


class GrainAnalyzer:
    """Analyzes grain boundaries and computes GB connecting vector."""
    
    def __init__(self, config, all_resids, coms_dict):
        self.config = config
        self.all_resids = all_resids
        self.coms_dict = coms_dict
        
    def load_trimmed_grains(self):
        """Load and validate trimmed grain files."""
        if not os.path.exists(self.config.g1_trimmed) or not os.path.exists(self.config.g2_trimmed):
            raise FileNotFoundError(
                f"Trimmed grain files not found:\n"
                f"  {self.config.g1_trimmed}\n"
                f"  {self.config.g2_trimmed}"
            )
        
        g1_gro = gro(self.config.g1_trimmed)
        g2_gro = gro(self.config.g2_trimmed)
        
        g1_resids = sorted({int(r) for r in g1_gro.allRes})
        g2_resids = sorted({int(r) for r in g2_gro.allRes})
        
        print(f"\nLoaded trimmed grains:")
        print(f"  g1_trimmed: {len(g1_resids)} residues")
        print(f"  g2_trimmed: {len(g2_resids)} residues")
        
        return g1_resids, g2_resids

    def _compute_center_3d(self, resid_list):
        """Compute 3D COM center for a list of residues."""
        xs, ys, zs = [], [], []
        for r in resid_list:
            x, y, z = self.coms_dict[r]
            xs.append(x); ys.append(y); zs.append(z)
        n = float(len(xs))
        return (sum(xs)/n, sum(ys)/n, sum(zs)/n)
    
    def compute_gb_vector(self, g1_resids, g2_resids):
        """
        Compute the 3D GB connecting vector from g1 center to g2 center.
        Direction is always from g1 -> g2 regardless of spatial arrangement.
        """
        g1_center_3d = self._compute_center_3d(g1_resids)
        g2_center_3d = self._compute_center_3d(g2_resids)
        
        gb_vector = (
            g2_center_3d[0] - g1_center_3d[0],
            g2_center_3d[1] - g1_center_3d[1],
            g2_center_3d[2] - g1_center_3d[2],
        )
        
        print(f"\nGrain 1 center: ({g1_center_3d[0]:.3f}, {g1_center_3d[1]:.3f}, {g1_center_3d[2]:.3f})")
        print(f"Grain 2 center: ({g2_center_3d[0]:.3f}, {g2_center_3d[1]:.3f}, {g2_center_3d[2]:.3f})")
        print(f"Grain-grain connecting vector (g1 -> g2): ({gb_vector[0]:.3f}, {gb_vector[1]:.3f}, {gb_vector[2]:.3f})")
        
        return gb_vector


class SourceResidSelector:
    """Handles selection of source residues for path sampling."""
    
    def __init__(self, config, pen_gro, all_resids, mp_coms):
        self.config = config
        self.pen_gro = pen_gro
        self.all_resids = all_resids
        self.mp_coms = mp_coms
        
    def get_source_resids(self, g1_resids, g2_resids):
        """Get source residues based on configuration."""
        if self.config.use_random_resids:
            return self._get_random_resids()
        else:
            return self._get_grain_union_resids(g1_resids, g2_resids)
    
    def _get_random_resids(self):
        """Extract random source residues or load from file."""
        print("\n[MODE] Using random_resids.txt as source_resids")
        
        random_file = "random_resids.txt"
        
        if os.path.exists(random_file):
            print(f"{random_file} already exists — skipping extraction.")
        else:
            print(f"{random_file} not found — extracting random source residues.")
            self._extract_random_resids(random_file)
        
        with open(random_file, "r") as f:
            lines = f.readlines()
        
        source_resids = [
            int(line.split()[0]) for line in lines if not line.startswith("#")
        ]
        
        print(f"Total source_resids from {random_file}: {len(source_resids)}")
        return source_resids
    
    def _extract_random_resids(self, output_file):
        """Extract random residues using ResidExtractor."""
        selector = resid.ResidExtractor(
            self.pen_gro, 
            self.config.natoms, 
            self.all_resids, 
            self.mp_coms
        )
        
        selector.extract_source_resids(
            axes_count=3,
            axes="x,y,z",
            selection_choice="no",
            y_range_choice=2,
            x_limits=(2.0, 14.0),
            y_limits_range_1=(1.0, 3.0),
            y_limits_range_2=[(4.0, 5.5), (7.0, 8.5)],
            z_limits=(7.0, 8.5),
            num_to_select=self.config.num_random_resids,
            output_file=output_file,
        )
    
    def _get_grain_union_resids(self, g1_resids, g2_resids):
        """Use union of trimmed grain residues."""
        print("\n[MODE] Using union of trimmed g1+g2 resids as source_resids")
        source_resids = sorted(set(g1_resids) | set(g2_resids))
        print(f"Total source_resids from g1_trimmed + g2_trimmed: {len(source_resids)}")
        return source_resids


def process_source_resid(
    source_resid: int,
    gro_file: str,
    top_file: str,
    gmx_path: str,
    mdp_file: str,
    gb_vector: tuple,
    g1_resids: list,
    g2_resids: list,
    cutoff: float,
    root_dir: str,
    ham_file: str,
    all_coms: list,
    all_resids: list,
    max_path_len: int,
    min_cos_forward: float,
    direction_alpha: float,
):
    """Process a single source residue to generate a biased path."""
    source_resid = int(source_resid)
    subdir = os.path.join(root_dir, f"SR_{source_resid}")
    os.makedirs(subdir, exist_ok=True)
    os.chdir(subdir)

    print(f"\n=== Source resid {source_resid} ===")
    print(f"Subdirectory: {subdir}")

    # Fresh gro object per worker
    pen_gro_local = gro(gro_file)

    # NEW DirectionalPathFinder: using gb_vector + g1/g2 membership, no bias_axis
    pathsample = directional_bias_path.DirectionalPathFinder(
        pen_gro_local,
        all_coms,
        all_resids,
        ham_file,
        top_file,
        gmx_path,
        mdp_file,
        gb_vector=gb_vector,
        g1_resids=g1_resids,
        g2_resids=g2_resids,
        cutoff=cutoff,
        min_cos_forward=min_cos_forward,
        direction_alpha=direction_alpha,
    )

    # Initialize path tracking
    sampled_paths = [source_resid]
    visited_resids = [source_resid]
    current_resid = source_resid

    # Check initial grain
    first_grain = pathsample.grain_of_resid(current_resid)
    if first_grain == "none":
        print(f"Residue {source_resid} not in g1 or g2. Skipping.")
        os.chdir(root_dir)
        return

    print(f"Starting biased walk from resid {source_resid} in grain {first_grain}")
    crossed = False

    # Biased random walk
    while len(sampled_paths) < max_path_len:
        # Compute coupling values
        cpl_values = pathsample.avg_cpl(
            ham_file,
            current_resid,
            top_file,
            gmx_path,
            mdp_file,
        )

        if not cpl_values:
            print(f"No coupling values for resid {current_resid}. Ending walk.")
            break

        # Calculate probabilities and select next neighbor
        coupling_probs = pathsample.probabilities(cpl_values)
        selected_neighbor = pathsample.select_next_neighbor(
            coupling_probs,
            current_resid,
            source_resid,
            visited_resids,
            sampled_paths,
        )

        if selected_neighbor is None:
            print("No valid next residue found. Ending walk.")
            break

        # Update path
        visited_resids.append(selected_neighbor)
        sampled_paths.append(selected_neighbor)

        # Check for grain crossing using membership
        selected_grain = pathsample.grain_of_resid(selected_neighbor)
        if not crossed:
            if (
                (first_grain == "g1" and selected_grain == "g2") or
                (first_grain == "g2" and selected_grain == "g1")
            ):
                crossed = True
                print(
                    f"Residue {selected_neighbor} has crossed from "
                    f"{first_grain} to {selected_grain} along GB direction."
                )

        current_resid = selected_neighbor

    # Save results
    output_file = f"final_sampled_paths_{source_resid}.txt"
    with open(output_file, "w") as f:
        for resid_id in sampled_paths:
            f.write(f"{resid_id}\n")

    print(f"Final sampled paths saved to {output_file}")
    print(f"Path length: {len(sampled_paths)} residues")
    print(f"Crossed GB? {'YES' if crossed else 'NO'}")

    os.chdir(root_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Biased path sampling across grain boundaries with parallel processing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Optional parameters
    parser.add_argument(
        "--max-path-len",
        type=int,
        default=25,
        help="Maximum number of residues in one sampled path"
    )
    parser.add_argument(
        "--use-random-resids",
        action="store_true",
        help="Use random_resids.txt; if False, use trimmed g1+g2 union"
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=32,
        help="Number of parallel worker processes"
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=0.60,
        help="Neighbor/path cutoff distance (nm)"
    )
    parser.add_argument(
        "--natoms",
        type=int,
        default=36,
        help="Number of atoms per molecule"
    )
    parser.add_argument(
        "--num-random-resids",
        type=int,
        default=2,
        help="Number of random residues to select per region"
    )

    # NEW: directional bias controls
    parser.add_argument(
        "--min-cos-forward",
        type=float,
        default=0.0,
        help="Minimum cos(theta) with GB direction for a step to be allowed (0=no backward)"
    )
    parser.add_argument(
        "--direction-alpha",
        type=float,
        default=4.0,
        help="Exponent for directional penalty (larger = stronger forward bias)"
    )
    
    args = parser.parse_args()
    config = PathSamplerConfig(args)
    
    print(f"\n{'='*60}")
    print(f"Biased Path Sampling using GB vector")
    print(f"{'='*60}")
    print(f"Max path length:      {config.max_path_len}")
    print(f"Number of processes:  {config.n_processes}")
    print(f"Cutoff:               {config.cutoff} nm")
    print(f"min_cos_forward:      {config.min_cos_forward}")
    print(f"direction_alpha:      {config.direction_alpha}")
    
    # Load NVT system and compute COMs
    print(f"\nLoading {config.gro_file}...")
    pen_gro_full = gro(config.gro_file)
    
    start = time.time()
    mp_coms = pen_gro_full.MP_resCOMs
    all_resids = [int(r) for r in pen_gro_full.allRes]
    print(
        f"Calculated COMs for {len(mp_coms)} residues "
        f"in {time.time() - start:.3f} seconds"
    )
    
    coms_dict = {resid: mp_coms[i] for i, resid in enumerate(all_resids)}
    
    # Analyze grains & GB vector
    grain_analyzer = GrainAnalyzer(config, all_resids, coms_dict)
    g1_resids, g2_resids = grain_analyzer.load_trimmed_grains()
    gb_vector = grain_analyzer.compute_gb_vector(g1_resids, g2_resids)
    
    # Select source residues
    selector = SourceResidSelector(config, pen_gro_full, all_resids, mp_coms)
    source_resids = selector.get_source_resids(g1_resids, g2_resids)
    
    # Prepare multiprocessing arguments
    process_args = [
        (
            r,
            config.gro_file,
            config.top_file,
            config.gmx_path,
            config.mdp_file,
            gb_vector,
            g1_resids,
            g2_resids,
            config.cutoff,
            config.root_dir,
            config.ham_file,
            mp_coms,
            all_resids,
            config.max_path_len,
            config.min_cos_forward,
            config.direction_alpha,
        )
        for r in source_resids
    ]
    
    # Run parallel processing
    print(f"\nStarting multiprocessing with {config.n_processes} processes...")
    print(f"Processing {len(source_resids)} source residues...")
    
    with mp.Pool(processes=config.n_processes) as pool:
        pool.starmap(process_source_resid, process_args)
    
    print(f"\n{'='*60}")
    print("All source residues processed.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
