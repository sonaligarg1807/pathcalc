import math
import random
import os
import shutil
import time
import glob
from pathcalc import top, gmx, inpman, asann


class DirectionalPathFinder:
    def __init__(
        self,
        gro_obj,
        all_coms: list,
        all_resids,
        ham_file,
        topFilePath,
        gmxPath,
        mdpFilePath,
        gb_vector,
        g1_resids,
        g2_resids,
        cutoff: float,
        min_cos_forward: float = 0.0,
        direction_alpha: float = 4.0,
    ):
        """
        Parameters
        ----------
        gro_obj : tintcalc.gro object
        all_coms : list of [x, y, z] COMs, same order as all_resids
        all_resids : list of residue IDs
        ham_file : str, name of TB_HAMILTONIAN.xvg
        topFilePath, gmxPath, mdpFilePath : paths for running QM/MM
        gb_vector : (gx, gy, gz) from grain1 center -> grain2 center
        g1_resids, g2_resids : lists/sets of residue IDs belonging to grain1 and grain2
        cutoff : distance cutoff (nm) for allowed steps
        min_cos_forward : minimum cosine with GB direction, below which a step is rejected
                          e.g. 0.0 = no backward steps; 0.3 ~ reject >72° off-axis.
        direction_alpha : strength of directional bias; larger = stronger penalty on off-axis
        """
        self.gro_obj = gro_obj
        self.all_coms = all_coms
        self.all_resids = all_resids
        self.coms = {resid: com for resid, com in zip(all_resids, all_coms)}
        self.ham_file = ham_file
        self.topFilePath = topFilePath
        self.mdpFilePath = mdpFilePath
        self.gmxPath = gmxPath
        self.cutoff = cutoff

        # Normalize grain–grain vector
        if gb_vector is None:
            raise ValueError("gb_vector must be provided.")
        gx, gy, gz = gb_vector
        norm = math.sqrt(gx * gx + gy * gy + gz * gz)
        if norm == 0:
            raise ValueError("gb_vector has zero length.")
        self.gb_vector = (gx / norm, gy / norm, gz / norm)

        # Grain membership
        self.g1 = set(int(r) for r in g1_resids)
        self.g2 = set(int(r) for r in g2_resids)

        # Directional penalty parameters
        self.min_cos_forward = float(min_cos_forward)
        self.direction_alpha = float(direction_alpha)

    # ------------------- Helpers -------------------

    def get_com(self, resid):
        com = self.coms.get(resid, None)
        if com is None:
            print(f"Resid {resid} not found in all_coms.")
        return com

    def grain_of_resid(self, resid):
        if resid in self.g1:
            return "g1"
        elif resid in self.g2:
            return "g2"
        else:
            return "none"

    def cleanup_specific_files(self, keep_extensions=[".txt"]):
        for filename in os.listdir():
            if os.path.isdir(filename) or any(filename.endswith(ext) for ext in keep_extensions):
                continue
            try:
                os.remove(filename)
            except Exception as e:
                print(f"Could not delete {filename}: {e}")

    # ------------------- Coupling + probabilities -------------------

    def avg_cpl(self, ham_file, source_resid, topFilePath, gmxPath, mdpFilePath):
        start = time.time()
        nn = self.gro_obj.cutAroundRes(source_resid, [1.0, 1.0, 1.0], allCOMs=self.all_coms)
        print(f"Time taken to cut around residue {source_resid}: {time.time() - start:.2f} s")
        if not nn:
            print(f"No neighbors found for residue {source_resid}. Ending walk.")
            return None

        nn_finder = asann.FirstNN(self.all_coms, self.all_resids, subset_resids=nn)
        neighbors = nn_finder.nearest_neighbors_asann(source_resid)
        if not neighbors:
            print(f"No neighbors found for residue {source_resid}. Ending walk.")
            return None

        resids_to_write = [source_resid] + neighbors
        self.gro_obj.write_gro("temporary.gro", resids_to_write)

        gmx_runner = gmx.gmx(exe="gmx", gro="temporary.gro")
        gmx_runner.renum()

        renumbered_resids = self.gro_obj.renumbered_resids("temporary.gro")
        resid_mapping = dict(zip(resids_to_write, renumbered_resids))

        # Topology
        new_top_path = "pen-esp.top"
        shutil.copy(topFilePath, new_top_path)
        itp_path = "/data/sgarg/pentacene/sampling_qm_zone_algorithm/gb-pen/input_files"
        itp_files = os.path.join(itp_path, "*.itp")
        for itp_file in glob.glob(itp_files):
            shutil.copy(itp_file, ".")
        pen_top = top(new_top_path)
        pen_top.update_molecule_count("molecule", len(resids_to_write))

        gmx_runner = gmx.gmx(
            exe=gmxPath, gro="temporary.gro",
            top=new_top_path, mdp=mdpFilePath, tpr="ham"
        )
        try:
            gmx_runner.grompp()
        except Exception as e:
            print(f"Error running grompp: {e}")

        if not os.path.exists("pen.spec"):
            spec_manager = inpman.inpManager("spec")
            spec_manager.update(
                natoms=36, nelectrons=102,
                nallorbitals=102, nfragorbs=1, fragorbs=51
            )
            spec_manager.save("pen.spec")
        else:
            print("spec file already exists.")

        renum_sites = [resid_mapping[r] for r in resids_to_write]
        ct_manager = inpman.inpManager("ct")
        ct_manager.update(
            sites=renum_sites,
            seed=random.randint(1, 100),
            chargecarrier="hole",
            atomindex=[1, 20, 29],
            typefiles="pen.spec",
            jobtype="NOM",
            internalrelax="onsite",
        )
        ct_manager.save("charge-transfer.dat")

        gmx_runner = gmx.gmx(exe=gmxPath, tpr="ham")
        gmx_runner.mdrun(nt=1)
        if not os.path.exists(ham_file):
            print(f"File {ham_file} not found.")
            return None

        extracted = {}
        with open(ham_file, "r") as file:
            for line in file:
                if line.strip() and not line.startswith(("#", "@")):
                    parts = line.split()
                    try:
                        values = [float(parts[i]) for i in range(2, 2 + len(neighbors))]
                        for neighbor_resid, cpl_value in zip(neighbors, values):
                            extracted[neighbor_resid] = abs(cpl_value) * 1000.0  # meV
                    except (IndexError, ValueError):
                        print(f"Error processing line: {line.strip()}")
                        continue

        for neigh, val in extracted.items():
            print(f"  Neighbor {neigh}: {val:.1f} meV")

        self.cleanup_specific_files()
        return extracted

    def probabilities(self, coupling_data, beta=(1 / 25.7)):
        """Boltzmann-like probabilities based on coupling magnitude."""
        if not coupling_data:
            return {}
        max_cpl = max(coupling_data.values())
        exp_weights = {
            resid: math.exp(beta * (cpl - max_cpl))
            for resid, cpl in coupling_data.items()
        }
        total = sum(exp_weights.values())
        if total <= 0:
            return {r: 0.0 for r in coupling_data}
        return {r: w / total for r, w in exp_weights.items()}

    # ------------------- Geometric checks -------------------

    def check_too_far(self, source_com, target_com):
        distance = math.dist(source_com, target_com)
        print(f"Distance SR–NR: {distance:.2f} nm")
        return distance > self.cutoff

    # Replace the _step_cos method and the selection logic to add better debugging:

    def _step_cos(self, source_resid, target_resid, forward_sign=1.0):
        """
        Cosine of angle between SR->NR step vector and (±)GB direction.
        Returns signed cos(theta) in [-1, 1] after applying forward_sign.
        """
        src = self.get_com(source_resid)
        tgt = self.get_com(target_resid)
        if src is None or tgt is None:
            return None

        step = (tgt[0] - src[0], tgt[1] - src[1], tgt[2] - src[2])
        step_norm = math.sqrt(step[0] ** 2 + step[1] ** 2 + step[2] ** 2)
        if step_norm == 0:
            return None

        dot = (
            step[0] * self.gb_vector[0] +
            step[1] * self.gb_vector[1] +
            step[2] * self.gb_vector[2]
        )
        cos_theta_raw = dot / step_norm
        cos_theta = cos_theta_raw * forward_sign
        
        # Debug output
        print(f"    Step vector: ({step[0]:.3f}, {step[1]:.3f}, {step[2]:.3f})")
        print(f"    GB vector: ({self.gb_vector[0]:.3f}, {self.gb_vector[1]:.3f}, {self.gb_vector[2]:.3f})")
        print(f"    Raw cos(theta) = {cos_theta_raw:.3f}, forward_sign = {forward_sign:.1f}, final cos(theta) = {cos_theta:.3f}")
        
        return cos_theta

    # ------------------- Neighbor selection -------------------

    def _forward_sign_for_path(self, first_source_resid):
        """
        Decide if forward is +gb_vector (from g1->g2) or -gb_vector (g2->g1),
        based on where the first source sits.
        """
        g = self.grain_of_resid(first_source_resid)
        if g == "g1":
            print(f"First source {first_source_resid} in g1: forward = +gb_vector")
            return 1.0
        elif g == "g2":
            print(f"First source {first_source_resid} in g2: forward = -gb_vector")
            return -1.0
        else:
            print(f"First source {first_source_resid} in neither grain: forward = +gb_vector (default)")
            return 1.0

    def select_next_neighbor(
        self,
        coupling_probs,
        source_resid,
        first_source_resid,
        visited_resids,
        sampled_paths,
    ):
        """
        Choose next neighbor using:
          - coupling_probs (from avg_cpl -> probabilities())
          - directional penalty via cos(theta)^direction_alpha
          - min_cos_forward threshold
          - distance cutoff
        """
        forward_sign = self._forward_sign_for_path(first_source_resid)

        while True:
            random_value = random.uniform(0, 1)
            print(f"Random number for source_resid {source_resid}: {random_value}")

            src_com = self.get_com(source_resid)
            if src_com is None:
                print("No COM for source_resid; cannot proceed.")
                return None

            # Build effective probabilities including directional weight
            eff_weights = {}
            for target_resid, p_cpl in coupling_probs.items():
                if target_resid in visited_resids:
                    continue

                cos_theta = self._step_cos(source_resid, target_resid, forward_sign)
                if cos_theta is None:
                    continue

                print(f"Resid {target_resid}: cos(theta) = {cos_theta:.3f}")

                # Hard filter: no backward / strongly sideways moves
                if cos_theta < self.min_cos_forward:
                    print(f"  -> below min_cos_forward={self.min_cos_forward}, reject.")
                    continue

                tgt_com = self.get_com(target_resid)
                if tgt_com is None:
                    continue
                if self.check_too_far(src_com, tgt_com):
                    print(f"  -> too far, reject.")
                    continue

                # Directional penalty: cos(theta)^alpha
                dir_factor = cos_theta ** self.direction_alpha
                eff = p_cpl * dir_factor
                print(f"  p_cpl={p_cpl:.3f}, dir_factor={dir_factor:.3f}, p_eff={eff:.3e}")
                if eff > 0.0:
                    eff_weights[target_resid] = eff

            if not eff_weights:
                # No valid neighbors from this residue — backtrack
                print("No valid neighbors (after directional + distance filters); backtracking...")
                while sampled_paths:
                    sampled_paths.pop()
                    if not sampled_paths:
                        print("No previous residues to backtrack to. Stopping.")
                        return None

                    fallback_resid = sampled_paths[-1]
                    selected = self.fallback_select_neighbor(
                        fallback_resid, visited_resids, forward_sign
                    )
                    print(f"Selected neighbor for fallback resid {fallback_resid}: {selected}")
                    if selected is not None:
                        print(f"Backtracking successful. New selected neighbor: {selected}")
                        return selected

                print("No valid neighbors found after backtracking. Ending.")
                return None

            # Normalize effective weights to probabilities
            total_eff = sum(eff_weights.values())
            eff_probs = {r: w / total_eff for r, w in eff_weights.items()}

            # Sample by approximate inverse-CDF using the "closest to random_value" trick
            sorted_neighbors = sorted(
                eff_probs.items(),
                key=lambda x: abs(x[1] - random_value),
            )

            chosen = sorted_neighbors[0][0]
            print(f"Selected neighbor {chosen} with eff_prob ≈ {sorted_neighbors[0][1]:.3f}")
            return chosen

    def fallback_select_neighbor(self, fallback_resid, visited_resids, forward_sign):
        """
        When stuck, recompute couplings around fallback_resid and use the same
        directional bias / distance cutoff logic.
        """
        avg_cpl_values = self.avg_cpl(
            self.ham_file,
            fallback_resid,
            self.topFilePath,
            self.gmxPath,
            self.mdpFilePath,
        )
        if not avg_cpl_values:
            print(f"No coupling values for fallback_resid {fallback_resid}.")
            return None

        coupling_probs = self.probabilities(avg_cpl_values)
        random_value = random.uniform(0, 1)
        print(f"Random number for fallback_resid {fallback_resid}: {random_value}")

        src_com = self.get_com(fallback_resid)
        if src_com is None:
            return None

        eff_weights = {}
        for target_resid, p_cpl in coupling_probs.items():
            if target_resid in visited_resids:
                continue

            cos_theta = self._step_cos(fallback_resid, target_resid, forward_sign)
            if cos_theta is None or cos_theta < self.min_cos_forward:
                continue

            tgt_com = self.get_com(target_resid)
            if tgt_com is None or self.check_too_far(src_com, tgt_com):
                continue

            dir_factor = cos_theta ** self.direction_alpha
            eff = p_cpl * dir_factor
            if eff > 0.0:
                eff_weights[target_resid] = eff

        if not eff_weights:
            print(f"No valid neighbors found for fallback_resid {fallback_resid}. Backtracking further...")
            return None

        total_eff = sum(eff_weights.values())
        eff_probs = {r: w / total_eff for r, w in eff_weights.items()}
        sorted_neighbors = sorted(
            eff_probs.items(), key=lambda x: abs(x[1] - random_value)
        )
        chosen = sorted_neighbors[0][0]
        print(f"Selected neighbor for fallback: {chosen} with eff_prob ≈ {sorted_neighbors[0][1]:.3f}")
        return chosen
