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
        min_forward_step: float = 0.05,
        # OPTIONAL PARAMETERS
        orientation_vecs=None,        # dict: resid -> unit vector (e.g. long axis)
        grain1_centroid=None,         # (x,y,z)
        grain2_centroid=None,         # (x,y,z)
        theta_boundary_deg: float = 15.0,
        gb_axis_index=None,           # main separation axis index (0:x,1:y,2:z)
        gb_mid_coord=None,            # coordinate of GB mid-plane along that axis
        boundary_width_nm: float = 0.7,
    ):
        """
        Biased path finder that tries to move from start grain -> boundary -> target grain.

        gb_vector : approximate forward vector from grain1 -> grain2
        g1_resids, g2_resids : residue IDs for each grain
        cutoff : base neighbor cutoff (nm)
        min_cos_forward, min_forward_step : initial forward constraints in start grain
        orientation_vecs : optional orientation vectors for boundary detection
        grain1_centroid, grain2_centroid : grain COMs
        gb_axis_index, gb_mid_coord : main axis and GB mid-plane coord (from GrainAnalyzer)
        boundary_width_nm : half-width around mid-plane treated as "GB region"
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

        # Directional penalty parameters (initial + dynamic)
        self.min_cos_forward_init = float(min_cos_forward)
        self.min_forward_step_init = float(min_forward_step)
        self.min_cos_forward_dyn = float(min_cos_forward)
        self.min_forward_step_dyn = float(min_forward_step)
        self.direction_alpha = float(direction_alpha)

        # Expanded-cutoff fallback
        self.cutoff_increments = [0.05, 0.10]

        # Optional orientation + centroids
        self.orientation_vecs = orientation_vecs or {}
        self.grain1_centroid = grain1_centroid
        self.grain2_centroid = grain2_centroid
        self.theta_boundary = math.radians(theta_boundary_deg)

        # GB mid-plane info
        self.gb_axis_index = gb_axis_index
        self.gb_mid_coord = gb_mid_coord
        self.boundary_width_nm = float(boundary_width_nm)

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

    def orientation_diff(self, resid_i, resid_j):
        """Misorientation angle (radians) between two residues, if orientation_vecs given."""
        vi = self.orientation_vecs.get(resid_i)
        vj = self.orientation_vecs.get(resid_j)
        if vi is None or vj is None:
            return None
        dot = vi[0]*vj[0] + vi[1]*vj[1] + vi[2]*vj[2]
        dot = max(-1.0, min(1.0, dot))
        return math.acos(dot)

    def closer_to_target_grain(self, resid, start_grain):
        """True if resid is closer to the *other* grain centroid (not used for mode trigger anymore)."""
        if self.grain1_centroid is None or self.grain2_centroid is None:
            return False
        com = self.get_com(resid)
        if com is None:
            return False
        d1 = math.dist(com, self.grain1_centroid)
        d2 = math.dist(com, self.grain2_centroid)
        if start_grain == "g1":
            return d2 < d1
        elif start_grain == "g2":
            return d1 < d2
        else:
            return False

    def distance_to_midplane(self, resid):
        """
        Signed distance (nm) along main axis from resid's COM to GB mid-plane.
        Sign tells which side of GB; |d| is distance from GB.
        """
        if self.gb_axis_index is None or self.gb_mid_coord is None:
            return None
        com = self.get_com(resid)
        if com is None:
            return None
        axis = self.gb_axis_index
        return com[axis] - self.gb_mid_coord

    # ------------------- Coupling + probabilities -------------------

    def _compute_couplings_for_neighbors(
        self,
        ham_file,
        source_resid,
        neighbors,
        topFilePath,
        gmxPath,
        mdpFilePath,
    ):
        """
        Core routine: given an explicit neighbor list, build temporary.gro,
        run QM/MM, and extract couplings.
        """
        if not neighbors:
            print(f"No neighbors passed to _compute_couplings_for_neighbors for resid {source_resid}.")
            return None

        resids_to_write = [source_resid] + neighbors
        self.gro_obj.write_gro("temporary.gro", resids_to_write)

        # Renumber
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

        # GROMPP
        gmx_runner = gmx.gmx(
            exe=gmxPath, gro="temporary.gro",
            top=new_top_path, mdp=mdpFilePath, tpr="ham"
        )
        try:
            gmx_runner.grompp()
        except Exception as e:
            print(f"Error running grompp: {e}")

        # spec file
        if not os.path.exists("pen.spec"):
            spec_manager = inpman.inpManager("spec")
            spec_manager.update(
                natoms=36, nelectrons=102,
                nallorbitals=102, nfragorbs=1, fragorbs=51
            )
            spec_manager.save("pen.spec")
        else:
            print("spec file already exists.")

        # charge-transfer.dat
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

        # mdrun
        gmx_runner = gmx.gmx(exe=gmxPath, tpr="ham")
        gmx_runner.mdrun(nt=1)
        if not os.path.exists(ham_file):
            print(f"File {ham_file} not found.")
            return None

        # Extract couplings
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

    def avg_cpl(self, ham_file, source_resid, topFilePath, gmxPath, mdpFilePath):
        """
        Base coupling calculation:
        - use cutAroundRes + FirstNN to pick neighbors,
        - then call the core coupling routine.
        """
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

        return self._compute_couplings_for_neighbors(
            ham_file,
            source_resid,
            neighbors,
            topFilePath,
            gmxPath,
            mdpFilePath,
        )

    def avg_cpl_geom_cutoff(self, source_resid, cutoff_ext):
        """
        Geometric neighbor list based purely on distance to source_resid,
        for a larger cutoff (fallback only; no FirstNN).
        """
        src_com = self.get_com(source_resid)
        if src_com is None:
            return None

        neighbors = []
        for resid in self.all_resids:
            if resid == source_resid:
                continue
            tgt_com = self.get_com(resid)
            if tgt_com is None:
                continue
            dist = math.dist(src_com, tgt_com)
            if dist <= cutoff_ext:
                neighbors.append(resid)

        print(f"[GEOM] For resid {source_resid} with cutoff {cutoff_ext:.3f} nm, "
              f"found {len(neighbors)} geometric neighbors.")

        if not neighbors:
            return None

        return self._compute_couplings_for_neighbors(
            self.ham_file,
            source_resid,
            neighbors,
            self.topFilePath,
            self.gmxPath,
            self.mdpFilePath,
        )

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

        print(f"    Step vector: ({step[0]:.3f}, {step[1]:.3f}, {step[2]:.3f})")
        print(f"    GB vector: ({self.gb_vector[0]:.3f}, {self.gb_vector[1]:.3f}, {self.gb_vector[2]:.3f})")
        print(f"    Raw cos(theta) = {cos_theta_raw:.3f}, forward_sign = {forward_sign:.1f}, final cos(theta) = {cos_theta:.3f}")

        return cos_theta

    def _step_projection(self, source_resid, target_resid, forward_sign=1.0):
        """
        Projection (in nm) of SR->NR step onto the (±)GB direction.
        Positive -> forward, negative -> backward.
        """
        src = self.get_com(source_resid)
        tgt = self.get_com(target_resid)
        if src is None or tgt is None:
            return None

        step = (tgt[0] - src[0], tgt[1] - src[1], tgt[2] - src[2])
        dot = (
            step[0] * self.gb_vector[0] +
            step[1] * self.gb_vector[1] +
            step[2] * self.gb_vector[2]
        )
        proj = dot * forward_sign
        return proj

    # ------------------- Neighbor selection helpers -------------------

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

    def _build_eff_weights(
        self,
        source_resid,
        coupling_probs,
        visited_resids,
        forward_sign,
        cutoff_eff,
        mode,          # "start", "boundary", "target"
        start_grain,   # "g1" / "g2" / "none"
        target_grain,  # "g1" / "g2" / None
    ):
        """
        Build effective weights for all neighbors using a given cutoff.

        mode:
          - "start": strong forward bias in start grain, drift to mid-plane
          - "boundary": relaxed direction, strong cross-grain preference
          - "target": avoid going back to start grain
        """
        src_com = self.get_com(source_resid)
        if src_com is None:
            return {}

        eff_weights = {}
        for target_resid, p_cpl in coupling_probs.items():
            if target_resid in visited_resids:
                continue

            cos_theta = self._step_cos(source_resid, target_resid, forward_sign)
            if cos_theta is None:
                continue

            proj = self._step_projection(source_resid, target_resid, forward_sign)
            if proj is None:
                continue

            tgt_com = self.get_com(target_resid)
            if tgt_com is None:
                continue
            distance = math.dist(src_com, tgt_com)

            grain_tgt = self.grain_of_resid(target_resid)

            # NEW: stay inside g1/g2 only (no wandering into "none")
            if grain_tgt == "none":
                continue

            print(
                f"Resid {target_resid}: cos(theta) = {cos_theta:.3f}, "
                f"proj = {proj:.3f} nm, dist = {distance:.3f} nm, cutoff = {cutoff_eff:.3f} nm"
            )

            # --- MODE-DEPENDENT FILTERS ---
            if mode == "start":
                if cos_theta < self.min_cos_forward_dyn:
                    print(f"  -> below min_cos_forward_dyn={self.min_cos_forward_dyn}, reject.")
                    continue
                if proj < self.min_forward_step_dyn:
                    print(f"  -> proj {proj:.3f} < min_forward_step_dyn {self.min_forward_step_dyn:.3f}, reject.")
                    continue
                if distance > cutoff_eff:
                    print(f"  -> distance {distance:.3f} > cutoff {cutoff_eff:.3f}, reject.")
                    continue

            elif mode == "boundary":
                if distance > cutoff_eff:
                    print(f"  -> distance {distance:.3f} > cutoff {cutoff_eff:.3f}, reject.")
                    continue
                if cos_theta < -0.5:
                    print("  -> strongly backwards at boundary, reject.")
                    continue

            elif mode == "target":
                if distance > cutoff_eff:
                    print(f"  -> distance {distance:.3f} > cutoff {cutoff_eff:.3f}, reject.")
                    continue
                if cos_theta < -0.2:
                    print("  -> backwards in target grain, reject.")
                    continue

            # --- GRAIN-BASED BONUS / PENALTY ---
            grain_bonus = 1.0
            if target_grain is not None:
                if mode in ("start", "boundary") and grain_tgt == target_grain:
                    grain_bonus *= 10.0   # strong push across GB
                if mode == "target" and grain_tgt == start_grain:
                    grain_bonus *= 0.1    # penalize going back

            # --- GB mid-plane drift bonus (STRONG now) ---
            d_src = self.distance_to_midplane(source_resid)
            d_tgt = self.distance_to_midplane(target_resid)
            if d_src is not None and d_tgt is not None:
                if abs(d_tgt) < abs(d_src):
                    grain_bonus *= 5.0    # strongly favour moves towards GB
                else:
                    grain_bonus *= 0.2    # strongly penalise moves away from GB

            # Directional factor
            if cos_theta > 0.0:
                dir_factor = cos_theta ** self.direction_alpha
            else:
                dir_factor = max(0.0, cos_theta + 1.0) ** self.direction_alpha

            eff = p_cpl * dir_factor * grain_bonus
            print(f"  p_cpl={p_cpl:.3f}, dir_factor={dir_factor:.3f}, grain_bonus={grain_bonus:.2f}, p_eff={eff:.3e}")
            if eff > 0.0:
                eff_weights[target_resid] = eff

        return eff_weights

    # ------------------- Neighbor selection -------------------

    def select_next_neighbor(
        self,
        coupling_probs,
        source_resid,
        first_source_resid,
        visited_resids,
        sampled_paths,
        mode,          # "start", "boundary", "target"
        boundary_flag, # bool
    ):
        """
        Returns
        -------
        chosen_resid, new_mode, new_boundary_flag
        """
        forward_sign = self._forward_sign_for_path(first_source_resid)
        random_value = random.uniform(0, 1)
        print(f"Random number for source_resid {source_resid}: {random_value}")

        start_grain = self.grain_of_resid(first_source_resid)
        if start_grain == "g1":
            target_grain = "g2"
        elif start_grain == "g2":
            target_grain = "g1"
        else:
            target_grain = None

        # --- Potential boundary detection when still in start mode ---
        if mode == "start" and target_grain is not None:
            boundary_trigger = False

            # 1) neighbor already in target grain
            for tr in coupling_probs.keys():
                if self.grain_of_resid(tr) == target_grain:
                    boundary_trigger = True
                    break

            # 2) orientation-based (if provided)
            if not boundary_trigger and self.orientation_vecs:
                for tr in coupling_probs.keys():
                    dtheta = self.orientation_diff(source_resid, tr)
                    if dtheta is not None and dtheta > self.theta_boundary:
                        boundary_trigger = True
                        break

            # 3) near GB mid-plane (main trigger now)
            d_mid = self.distance_to_midplane(source_resid)
            if not boundary_trigger and d_mid is not None:
                if abs(d_mid) < self.boundary_width_nm:
                    print(f"{source_resid} within {self.boundary_width_nm:.2f} nm of GB mid-plane -> boundary mode")
                    boundary_trigger = True

            # (removed centroid-based trigger – it was firing too early)

            if boundary_trigger:
                print(f"Entering boundary mode from {source_resid}")
                mode = "boundary"
                boundary_flag = True
                self.min_cos_forward_dyn = 0.0
                self.min_forward_step_dyn = 0.0

        # 1) Base cutoff
        eff_weights = self._build_eff_weights(
            source_resid,
            coupling_probs,
            visited_resids,
            forward_sign,
            cutoff_eff=self.cutoff,
            mode=mode,
            start_grain=start_grain,
            target_grain=target_grain,
        )

        # 1b) if in start mode and nothing, relax thresholds once
        if mode == "start" and not eff_weights:
            old_cos = self.min_cos_forward_dyn
            old_proj = self.min_forward_step_dyn
            self.min_cos_forward_dyn = max(0.0, self.min_cos_forward_dyn * 0.8)
            self.min_forward_step_dyn = max(0.0, self.min_forward_step_dyn * 0.5)
            print(f"No candidates in start mode. Relax thresholds: "
                  f"min_cos {old_cos:.3f}->{self.min_cos_forward_dyn:.3f}, "
                  f"min_forward {old_proj:.3f}->{self.min_forward_step_dyn:.3f}")
            eff_weights = self._build_eff_weights(
                source_resid,
                coupling_probs,
                visited_resids,
                forward_sign,
                cutoff_eff=self.cutoff,
                mode=mode,
                start_grain=start_grain,
                target_grain=target_grain,
            )

        # 2) Geometric expanded cutoffs
        if not eff_weights:
            print("No valid neighbors with base cutoff; trying expanded geometric cutoffs...")
            for inc in self.cutoff_increments:
                new_cutoff = self.cutoff + inc
                print(f"  Geometric fallback: cutoff = {new_cutoff:.3f} nm (base + {inc:.3f})")
                geom_cpl = self.avg_cpl_geom_cutoff(source_resid, new_cutoff)
                if not geom_cpl:
                    print("  -> No couplings from geometric neighbor set.")
                    continue

                geom_probs = self.probabilities(geom_cpl)
                eff_weights = self._build_eff_weights(
                    source_resid,
                    geom_probs,
                    visited_resids,
                    forward_sign,
                    cutoff_eff=new_cutoff,
                    mode=mode,
                    start_grain=start_grain,
                    target_grain=target_grain,
                )
                if eff_weights:
                    print(f"  Found valid neighbors with geometric cutoff {new_cutoff:.3f} nm")
                    break

        # 3) Backtracking with boundary-like fallback
        if not eff_weights:
            print("No valid neighbors even after geometric cutoffs; backtracking...")
            while sampled_paths:
                sampled_paths.pop()
                if not sampled_paths:
                    print("No previous residues to backtrack to. Stopping.")
                    return None, mode, boundary_flag

                fallback_resid = sampled_paths[-1]
                selected = self.fallback_select_neighbor(
                    fallback_resid, visited_resids, forward_sign, start_grain, target_grain
                )
                print(f"Selected neighbor for fallback resid {fallback_resid}: {selected}")
                if selected is not None:
                    print(f"Backtracking successful. New selected neighbor: {selected}")
                    mode = "boundary"
                    boundary_flag = True
                    return selected, mode, boundary_flag

            print("No valid neighbors found after backtracking. Ending.")
            return None, mode, boundary_flag

        # Normalize and sample
        total_eff = sum(eff_weights.values())
        eff_probs = {r: w / total_eff for r, w in eff_weights.items()}

        sorted_neighbors = sorted(
            eff_probs.items(),
            key=lambda x: abs(x[1] - random_value),
        )

        chosen = sorted_neighbors[0][0]
        print(f"Selected neighbor {chosen} with eff_prob ≈ {sorted_neighbors[0][1]:.3f}")

        # Mode transitions based on grain crossing
        chosen_grain = self.grain_of_resid(chosen)

        if target_grain is not None and chosen_grain == target_grain and mode != "target":
            print(f"Crossed into target grain {target_grain} at resid {chosen}")
            mode = "target"
            boundary_flag = False

            # In target grain, drop strict forward thresholds
            self.min_cos_forward_dyn = 0.0
            self.min_forward_step_dyn = 0.0

        return chosen, mode, boundary_flag

    def fallback_select_neighbor(self, fallback_resid, visited_resids, forward_sign, start_grain, target_grain):
        """
        Boundary-like fallback:
          - recompute couplings around fallback_resid (FirstNN)
          - use boundary-style filters (relaxed forward constraints)
          - strong preference for target_grain over start_grain
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

        eff_weights = self._build_eff_weights(
            fallback_resid,
            coupling_probs,
            visited_resids,
            forward_sign,
            cutoff_eff=self.cutoff,
            mode="boundary",
            start_grain=start_grain,
            target_grain=target_grain,
        )

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
