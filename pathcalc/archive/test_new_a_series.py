## fix version with forward direction from grain COM1->COM2 projected in slab plane (z normal)
import math
import random
import os
import shutil
import glob
from typing import Dict, Optional

import numpy as np

from pathcalc import top, gmx, inpman, asann


class PathFinder:
    """
    Path sampling with:
      - NORMAL mode:
          * probability-based neighbor ordering (UNCHANGED)
          * forward constraint now uses a global in-plane direction vector
            from COM(start grain) -> COM(target grain), projected into slab plane
      - RESCUE mode:
          * geometric neighbor search within rescue_radius (no ASANN)
          * uses the same forward constraint (global direction)
          * coupling still computed for rescue candidates and selection is still stochastic (UNCHANGED pattern)
      - FALLBACK mode:
          * backtracking (UNCHANGED structure)
          * uses the same forward constraint (global direction)

    IMPORTANT:
      - Probabilities logic and ordering are NOT changed.
      - Only "moving forward" logic is changed from a single axis to a global direction.
    """

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------
    def __init__(
        self,
        gro_obj,
        all_coms,
        all_resids,
        ham_file,
        topFilePath,
        gmxPath,
        mdpFilePath,
        axis_ranges,
        cutoff,
        bias_axis="y",  # kept for grain assignment only (via axis_ranges)
        rescue_radius=None,
        rescue_top_k=6,
        rescue_forward_eps=0.0,     # forward eps used in RESCUE along path direction
        rescue_cutoff=None,
        rescue_max_neighbors=28,
        slab_normal_axis="z",       # your case: z
        forward_eps=-0.02,          # allow tiny backward wiggle (nm) along path direction
    ):
        self.gro_obj = gro_obj
        self.all_coms = all_coms
        self.all_resids = all_resids
        self.coms = {r: c for r, c in zip(all_resids, all_coms)}

        self.ham_file = ham_file
        self.topFilePath = topFilePath
        self.mdpFilePath = mdpFilePath
        self.gmxPath = gmxPath

        self.axis_ranges = axis_ranges
        self.cutoff = cutoff

        axis = bias_axis.lower()
        self.bias_axis = axis
        self.axis_index = {"x": 0, "y": 1, "z": 2}[axis]

        self.rescue_radius = rescue_radius if rescue_radius is not None else cutoff * 1.5
        self.rescue_cutoff = rescue_cutoff if rescue_cutoff is not None else self.rescue_radius
        self.rescue_top_k = rescue_top_k
        self.rescue_forward_eps = rescue_forward_eps
        self.rescue_max_neighbors = rescue_max_neighbors

        slab_normal_axis = slab_normal_axis.lower()
        if slab_normal_axis not in ("x", "y", "z"):
            raise ValueError("slab_normal_axis must be 'x', 'y', or 'z'")
        self.slab_normal_axis = slab_normal_axis

        # ---------------- NEW: global forward direction (COM grain1 -> COM grain2) ----------------
        self.forward_eps = float(forward_eps)  # NM units (same as COM coordinates)
        self._path_dir = None                  # unit vector (3,), in slab plane
        self._path_initialized = False
        self._start_range_cached = None
        self._target_range_cached = None

        # current path context (so RESCUE/FALLBACK can init direction too)
        self._current_first_source_resid = None

    # ------------------------------------------------------------------
    # BASIC HELPERS
    # ------------------------------------------------------------------
    def get_com(self, resid):
        return self.coms.get(resid)

    def check_axis_range(self, resid):
        com = self.get_com(resid)
        if com is None:
            return None
        coord = com[self.axis_index]
        r1, r2 = self.axis_ranges
        if r1[0] <= coord <= r1[1]:
            return "in_range_1"
        if r2[0] <= coord <= r2[1]:
            return "in_range_2"
        return "out_of_range"

    def _has_crossed(self, start_range, resid):
        curr = self.check_axis_range(resid)
        return curr in ("in_range_1", "in_range_2") and curr != start_range

    def cleanup_specific_files(self, keep_extensions=[".txt"]):
        for filename in os.listdir():
            if os.path.isdir(filename) or any(filename.endswith(ext) for ext in keep_extensions):
                continue
            try:
                os.remove(filename)
            except Exception as e:
                print(f"Could not delete {filename}: {e}")

    # ------------------------------------------------------------------
    # SLAB NORMAL + PROJECTION
    # ------------------------------------------------------------------
    def _slab_normal_vector(self):
        if self.slab_normal_axis == "x":
            return np.array([1.0, 0.0, 0.0], dtype=float)
        if self.slab_normal_axis == "y":
            return np.array([0.0, 1.0, 0.0], dtype=float)
        return np.array([0.0, 0.0, 1.0], dtype=float)

    def _project_into_slab(self, v: np.ndarray) -> np.ndarray:
        """Project vector v onto slab plane (remove component along slab normal)."""
        n = self._slab_normal_vector()
        return v - np.dot(v, n) * n

    def _grain_com(self, which_range):
        pts = [self.get_com(r) for r in self.all_resids if self.check_axis_range(r) == which_range]
        pts = [p for p in pts if p is not None]
        if not pts:
            return None
        return np.mean(np.asarray(pts, dtype=float), axis=0)

    # ------------------------------------------------------------------
    # NEW: GLOBAL PATH DIRECTION (start grain -> target grain), in slab plane
    # ------------------------------------------------------------------
    def _init_path_direction_if_needed(self, first_source_resid: Optional[int] = None):
        """
        Grain-based global direction:
            u_path = proj_slab(COM(other_grain) - COM(start_grain)) / ||...||
    
        Then auto-flip u_path (if needed) so that "forward" matches the actual
        local forward trend from the start grain (prevents accidental reversed direction).
        """
        if self._path_initialized:
            return
    
        if first_source_resid is None:
            first_source_resid = self._current_first_source_resid
    
        if first_source_resid is None:
            self._path_initialized = True
            self._path_dir = None
            return
    
        start_range = self.check_axis_range(first_source_resid)
        if start_range not in ("in_range_1", "in_range_2"):
            self._path_initialized = True
            self._path_dir = None
            print("⚠️ first_source_resid is out_of_range → forward-direction disabled")
            return
    
        target_range = "in_range_2" if start_range == "in_range_1" else "in_range_1"
    
        com_start  = self._grain_com(start_range)
        com_target = self._grain_com(target_range)
    
        if com_start is None or com_target is None:
            self._path_initialized = True
            self._path_dir = None
            self._start_range_cached = start_range
            self._target_range_cached = target_range
            print("⚠️ Could not compute grain COMs reliably → forward-direction disabled")
            return
    
        v = np.asarray(com_target, float) - np.asarray(com_start, float)
        v = self._project_into_slab(v)
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            self._path_initialized = True
            self._path_dir = None
            self._start_range_cached = start_range
            self._target_range_cached = target_range
            print("⚠️ Grain COM separation projects to ~0 in slab plane → disabled")
            return
    
        u = v / nv
    
        # ------------------------------------------------------------------
        # AUTO-FLIP so that "forward" is consistent with local steps
        # (this is the key “always moves forward” guard)
        # ------------------------------------------------------------------
        # sample some start-grain molecules near the start resid (or random from that grain)
        src_com = self.get_com(first_source_resid)
        if src_com is not None:
            src_com = np.asarray(src_com, float)
    
            # pick nearby candidates from same start grain (prefer local neighborhood)
            local = self.neighbors_within_radius(
                first_source_resid, radius_nm=max(self.cutoff, 1.2*self.cutoff), max_neighbors=40
            )
            local = [r for r in local if self.check_axis_range(r) == start_range]
    
            # if local is empty, fall back to random sample from start grain
            if not local:
                grain_res = [r for r in self.all_resids if self.check_axis_range(r) == start_range and r != first_source_resid]
                random.shuffle(grain_res)
                local = grain_res[:40]
    
            # compute progress for candidate steps and flip if most are negative
            progs = []
            for r in local:
                tc = self.get_com(r)
                if tc is None:
                    continue
                tc = np.asarray(tc, float)
                progs.append(float(np.dot(tc - src_com, u)))
    
            if progs:
                frac_neg = sum(p < 0 for p in progs) / len(progs)
                # if majority are "backward", your u is reversed relative to actual local forward direction
                if frac_neg > 0.6:
                    u = -u
                    print("⚠️ Auto-flip: u_path reversed to enforce forward motion from start grain")
    
        self._path_dir = u
        self._path_initialized = True
        self._start_range_cached = start_range
        self._target_range_cached = target_range
    
        print("\n" + "-" * 60)
        print("[GLOBAL PATH DIRECTION INITIALIZATION: GRAIN -> GRAIN]")
        print(f"  slab_normal_axis:  {self.slab_normal_axis}")
        print(f"  start_range:       {start_range}")
        print(f"  target_range:      {target_range}")
        print(f"  COM(start grain):  {np.asarray(com_start)}")
        print(f"  COM(target grain): {np.asarray(com_target)}")
        print(f"  u_path (in-plane): {self._path_dir}")
        print(f"  forward_eps (nm):  {self.forward_eps:.4f}")
        print("-" * 60)
    
    def _path_progress(self, sc: np.ndarray, tc: np.ndarray) -> float:
        """Signed progress along u_path."""
        if self._path_dir is None:
            return 0.0
        return float(np.dot(tc - sc, self._path_dir))

    def _is_forward_along_path(self, sc: np.ndarray, tc: np.ndarray, eps: float) -> bool:
        return self._path_progress(sc, tc) > eps

    # ------------------------------------------------------------------
    # SUMMARY HELPER
    # ------------------------------------------------------------------
    def _make_step_info(self, mode, source_resid, target_resid, first_source_resid):
        sc = np.asarray(self.get_com(source_resid), dtype=float)
        tc = np.asarray(self.get_com(target_resid), dtype=float)

        dist = float(np.linalg.norm(tc - sc))
        src_range = self.check_axis_range(source_resid)
        tgt_range = self.check_axis_range(target_resid)
        start_range = self.check_axis_range(first_source_resid)

        crossed = (
            src_range in ("in_range_1", "in_range_2")
            and tgt_range in ("in_range_1", "in_range_2")
            and src_range != tgt_range
        )

        path_prog = self._path_progress(sc, tc) if self._path_dir is not None else 0.0

        return {
            "mode": mode,
            "source_resid": source_resid,
            "target_resid": target_resid,
            "distance_nm": round(dist, 4),
            "path_progress_nm": round(path_prog, 4),
            "source_grain": "g1" if src_range == "in_range_1" else "g2",
            "target_grain": "g1" if tgt_range == "in_range_1" else "g2",
            "crossed": "YES" if crossed else "NO",
        }

    # ------------------------------------------------------------------
    # RESCUE NEIGHBOR SEARCH (geometric, no ASANN)
    # ------------------------------------------------------------------
    def neighbors_within_radius(self, source_resid, radius_nm, max_neighbors=None):
        src = self.get_com(source_resid)
        if src is None:
            return []

        src = np.asarray(src, dtype=float)
        candidates = []
        for r in self.all_resids:
            if r == source_resid:
                continue
            tgt = self.get_com(r)
            if tgt is None:
                continue
            tgt = np.asarray(tgt, dtype=float)
            d = float(np.linalg.norm(tgt - src))
            if d <= radius_nm:
                candidates.append((r, d))

        candidates.sort(key=lambda x: x[1])
        if max_neighbors is not None and max_neighbors > 0:
            candidates = candidates[:max_neighbors]

        neighbor_resids = [r for r, _ in candidates]
        print(f"  Found {len(neighbor_resids)} neighbors within {radius_nm:.4f} nm")
        if neighbor_resids:
            print(f"  Neighbor resids: {neighbor_resids[:10]}{'...' if len(neighbor_resids) > 10 else ''}")
        return neighbor_resids

    # ------------------------------------------------------------------
    # COUPLING CALCULATION (NORMAL MODE - UNCHANGED LOGIC)
    # ------------------------------------------------------------------
    def avg_cpl(
        self,
        ham_file: str,
        source_resid: int,
        topFilePath: str,
        gmxPath: str,
        mdpFilePath: str,
    ) -> Optional[Dict[int, float]]:

        nn = self.gro_obj.cutAroundRes(
            source_resid, [1.0, 1.0, 1.0], allCOMs=self.all_coms
        )
        if not nn:
            return None

        nn_finder = asann.FirstNN(self.all_coms, self.all_resids, subset_resids=nn)
        neighbors = nn_finder.nearest_neighbors_asann(source_resid)
        if not neighbors:
            return None

        resids_to_write = [source_resid] + neighbors
        self.gro_obj.write_gro("temporary.gro", resids_to_write)

        gmx_runner = gmx.gmx(exe="gmx", gro="temporary.gro")
        gmx_runner.renum()

        renumbered_resids = self.gro_obj.renumbered_resids("temporary.gro")
        resid_mapping = dict(zip(resids_to_write, renumbered_resids))

        new_top_path = "pen-esp.top"
        shutil.copy(topFilePath, new_top_path)
        itp_path = "/data/sgarg/pentacene/sampling_qm_zone_algorithm/gb-pen/input_files"
        itp_files = os.path.join(itp_path, "*.itp")
        for itp_file in glob.glob(itp_files):
            shutil.copy(itp_file, ".")
        pen_top = top(new_top_path)
        pen_top.update_molecule_count("molecule", len(resids_to_write))

        gmx_runner = gmx.gmx(
            exe=gmxPath,
            gro="temporary.gro",
            top="pen-esp.top",
            mdp=mdpFilePath,
            tpr="ham",
        )
        gmx_runner.grompp()

        if not os.path.exists("pen.spec"):
            spec = inpman.inpManager("spec")
            spec.update(natoms=36, nelectrons=102, nallorbitals=102,
                        nfragorbs=1, fragorbs=51)
            spec.save("pen.spec")

        ct = inpman.inpManager("ct")
        ct.update(
            sites=[resid_mapping[r] for r in resids_to_write],
            seed=random.randint(1, 100),
            chargecarrier="hole",
            atomindex=[1, 20, 29],
            typefiles="pen.spec",
            jobtype="NOM",
            internalrelax="onsite",
        )
        ct.save("charge-transfer.dat")

        gmx.gmx(exe=gmxPath, tpr="ham").mdrun(nt=1)

        if not os.path.exists(ham_file):
            return None

        extracted = {}
        with open(ham_file) as f:
            for line in f:
                if line.startswith(("#", "@")):
                    continue
                parts = line.split()
                vals = [abs(float(v)) * 1000 for v in parts[2:2 + len(neighbors)]]
                for r, v in zip(neighbors, vals):
                    extracted[r] = v

        self.cleanup_specific_files()
        return extracted

    # ------------------------------------------------------------------
    # COUPLING CALCULATION FOR EXPLICIT NEIGHBOR LIST (RESCUE MODE)
    # ------------------------------------------------------------------
    def avg_cpl_for_neighbors(self, ham_file, source_resid, neighbors, topFilePath, gmxPath, mdpFilePath):
        neighbors = list(neighbors) if neighbors else []
        if not neighbors:
            return None

        resids_to_write = [source_resid] + neighbors
        self.gro_obj.write_gro("temporary.gro", resids_to_write)

        gmx_runner = gmx.gmx(exe="gmx", gro="temporary.gro")
        gmx_runner.renum()

        renumbered_resids = self.gro_obj.renumbered_resids("temporary.gro")
        resid_mapping = dict(zip(resids_to_write, renumbered_resids))

        new_top_path = "pen-esp.top"
        shutil.copy(topFilePath, new_top_path)
        itp_path = "/data/sgarg/pentacene/sampling_qm_zone_algorithm/gb-pen/input_files"
        for itp_file in glob.glob(os.path.join(itp_path, "*.itp")):
            shutil.copy(itp_file, ".")
        pen_top = top(new_top_path)
        pen_top.update_molecule_count("molecule", len(resids_to_write))

        gmx_runner = gmx.gmx(
            exe=gmxPath,
            gro="temporary.gro",
            top="pen-esp.top",
            mdp=mdpFilePath,
            tpr="ham",
        )
        gmx_runner.grompp()

        if not os.path.exists("pen.spec"):
            spec = inpman.inpManager("spec")
            spec.update(natoms=36, nelectrons=102, nallorbitals=102,
                        nfragorbs=1, fragorbs=51)
            spec.save("pen.spec")

        ct = inpman.inpManager("ct")
        ct.update(
            sites=[resid_mapping[r] for r in resids_to_write],
            seed=random.randint(1, 100),
            chargecarrier="hole",
            atomindex=[1, 20, 29],
            typefiles="pen.spec",
            jobtype="NOM",
            internalrelax="onsite",
        )
        ct.save("charge-transfer.dat")

        gmx.gmx(exe=gmxPath, tpr="ham").mdrun(nt=1)

        if not os.path.exists(ham_file):
            return None

        extracted = {}
        with open(ham_file) as f:
            for line in f:
                if line.startswith(("#", "@")):
                    continue
                parts = line.split()
                vals = [abs(float(v)) * 1000 for v in parts[2:2 + len(neighbors)]]
                for r, v in zip(neighbors, vals):
                    extracted[r] = v

        self.cleanup_specific_files()
        return extracted

    # ------------------------------------------------------------------
    # PROBABILITIES (UNCHANGED)
    # ------------------------------------------------------------------
    def probabilities(self, coupling_data, beta=1 / 25.7):
        if not coupling_data:
            return {}
        maxc = max(coupling_data.values())
        weights = {r: math.exp(beta * (c - maxc)) for r, c in coupling_data.items()}
        Z = sum(weights.values())
        return {r: w / Z for r, w in weights.items()} if Z > 0 else {}

    # ------------------------------------------------------------------
    # MAIN STEP SELECTION
    # ------------------------------------------------------------------
    def select_next_neighbor(
        self,
        probabilities,
        source_resid,
        first_source_resid,
        visited_resids,
        sampled_paths,
    ):
        # store context so rescue/fallback can also init direction if needed
        self._current_first_source_resid = first_source_resid
        if not self._path_initialized:
            self._init_path_direction_if_needed(first_source_resid)

        sc = self.get_com(source_resid)
        sc = np.asarray(sc, dtype=float)

        start_range = self.check_axis_range(first_source_resid)
        crossed = self._has_crossed(start_range, source_resid)

        print(f"\n{'='*60}")
        print(f"SELECTING NEIGHBOR FOR SOURCE RESID: {source_resid}")
        print(f"Start range: {start_range}, Crossed: {crossed}")
        print(f"Slab normal axis: {self.slab_normal_axis}")
        print(f"Global path dir initialized: {self._path_initialized}")
        print(f"Global path dir (u_path): {self._path_dir}")
        print(f"{'='*60}")

        # NORMAL
        if probabilities:
            rnd = random.random()
            print(f"\n[NORMAL MODE] Random number: {rnd:.6f}")
            print(f"\nAll neighbors with probabilities:")
            for tgt, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                print(f"  Resid {tgt}: prob={prob:.6f}")

            # IMPORTANT: This ordering logic is kept UNCHANGED (your method)
            ordered = sorted(probabilities.items(), key=lambda x: abs(x[1] - rnd))
            print(f"\nChecking neighbors (ordered by closest prob to random {rnd:.6f}):")

            for tgt, prob in ordered:
                tc = self.get_com(tgt)
                if tc is None:
                    continue
                tc = np.asarray(tc, dtype=float)

                dist = float(np.linalg.norm(tc - sc))

                # NEW forwardness: dot(step, u_path) > eps
                if self._path_dir is not None:
                    path_prog = self._path_progress(sc, tc)
                    is_forward = self._is_forward_along_path(sc, tc, eps=self.forward_eps)
                else:
                    path_prog = 0.0
                    is_forward = True  # if we can't define direction, don't block movement

                print(f"\n  Resid {tgt}: prob={prob:.6f}, diff={abs(prob - rnd):.6f}")
                print(f"    Distance: {dist:.4f} nm (cutoff: {self.cutoff:.4f})")
                print(f"    Path progress: {path_prog:.4f} nm, Forward: {is_forward} (eps={self.forward_eps:.4f})")

                if tgt in visited_resids:
                    print(f"    ❌ REJECTED: Already visited")
                    continue

                if dist > self.cutoff:
                    print(f"    ❌ REJECTED: Distance > cutoff")
                    continue

                # NEW: always enforce forward along global path if available (before + after crossing)
                if self._path_dir is not None and not is_forward:
                    print(f"    ❌ REJECTED: Not forward along global path direction")
                    continue

                print(f"    ✅ SELECTED")
                return tgt, self._make_step_info("NORMAL", source_resid, tgt, first_source_resid)

            print(f"\n❌ No valid neighbors in NORMAL mode")

        # RESCUE
        print(f"\n[RESCUE MODE]")
        print(f"Searching for neighbors within rescue_radius={self.rescue_radius:.4f} nm (max={self.rescue_max_neighbors})")

        radius_neighbors = self.neighbors_within_radius(
            source_resid,
            radius_nm=self.rescue_radius,
            max_neighbors=self.rescue_max_neighbors
        )

        radius_neighbors = [r for r in radius_neighbors if r not in visited_resids]

        if not radius_neighbors:
            print(f"❌ No unvisited neighbors within rescue radius")
        else:
            print(f"Computing couplings for {len(radius_neighbors)} rescue candidates...")
            cpl = self.avg_cpl_for_neighbors(
                self.ham_file, source_resid, radius_neighbors,
                self.topFilePath, self.gmxPath, self.mdpFilePath
            )

            if cpl:
                print(f"Rescue mode coupling values (meV):")
                for r, c in sorted(cpl.items(), key=lambda x: x[1], reverse=True):
                    print(f"  Resid {r}: {c:.2f} meV")

                rescued = self.rescue_select_neighbor_path_dominant(
                    cpl, source_resid, visited_resids
                )
                if rescued:
                    print(f"✅ RESCUE selected: {rescued}")
                    return rescued, self._make_step_info("RESCUE", source_resid, rescued, first_source_resid)

        print(f"❌ RESCUE failed")

        # FALLBACK
        print(f"\n[FALLBACK MODE - Backtracking]")
        while sampled_paths:
            sampled_paths.pop()
            if not sampled_paths:
                print(f"❌ No more paths to backtrack")
                return None
            fb = sampled_paths[-1]
            print(f"Backtracking to resid {fb}")
            nxt = self.fallback_select_neighbor(fb, visited_resids, first_source_resid)
            if nxt:
                print(f"✅ FALLBACK selected: {nxt}")
                return nxt, self._make_step_info("FALLBACK", fb, nxt, first_source_resid)
            print(f"❌ No valid neighbor from {fb}, continuing backtrack")

        print(f"\n❌ ALL MODES FAILED")
        return None

    # ------------------------------------------------------------------
    # RESCUE (PATH-DOMINANT WITH STOCHASTIC SELECTION)  [structure preserved]
    # ------------------------------------------------------------------
    def rescue_select_neighbor_path_dominant(self, coupling_data, source_resid, visited_resids):
        if not coupling_data:
            return None

        # ensure direction exists if possible
        if not self._path_initialized:
            self._init_path_direction_if_needed(self._current_first_source_resid)

        sc = self.get_com(source_resid)
        sc = np.asarray(sc, dtype=float)

        print(f"  Filtering candidates (cutoff={self.rescue_cutoff:.4f}, eps={self.rescue_forward_eps:.4f}):")
        candidates = []

        for r, cpl in coupling_data.items():
            tc = self.get_com(r)
            if tc is None:
                continue
            tc = np.asarray(tc, dtype=float)
            dist = float(np.linalg.norm(tc - sc))

            if self._path_dir is not None:
                prog = self._path_progress(sc, tc)
                is_forward = self._is_forward_along_path(sc, tc, eps=self.rescue_forward_eps)
            else:
                prog = 0.0
                is_forward = True

            print(f"    Resid {r}: cpl={cpl:.2f}, dist={dist:.4f}, path_prog={prog:.4f}")

            if r in visited_resids:
                print(f"      ❌ Visited")
                continue

            if dist > self.rescue_cutoff:
                print(f"      ❌ Too far")
                continue

            if self._path_dir is not None and not is_forward:
                print(f"      ❌ Not forward (global path)")
                continue

            print(f"      ✓ Added to candidates")
            candidates.append((r, cpl, prog))

        if not candidates:
            return None

        # Sort by coupling (descending) and take top-k (UNCHANGED)
        candidates.sort(key=lambda x: x[1], reverse=True)
        topk = candidates[: self.rescue_top_k]

        print(f"\n  Top-{self.rescue_top_k} candidates by coupling:")
        for r, cpl, prog in topk:
            print(f"    Resid {r}: cpl={cpl:.2f}, prog={prog:.4f}")

        # Create probabilities based on progress (UNCHANGED pattern)
        progress_values = {c[0]: c[2] for c in topk}
        total_progress = sum(progress_values.values())

        if total_progress <= 0:
            probabilities = {r: 1.0 / len(topk) for r in progress_values.keys()}
        else:
            probabilities = {r: prog / total_progress for r, prog in progress_values.items()}

        rnd = random.random()
        print(f"\n  Rescue random number: {rnd:.6f}")
        print(f"  Progress-based probabilities:")
        for r, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            cpl_val = next(c[1] for c in topk if c[0] == r)
            prog_val = progress_values[r]
            print(f"    Resid {r}: prob={prob:.6f}, cpl={cpl_val:.2f}, prog={prog_val:.4f}")

        ordered = sorted(probabilities.items(), key=lambda x: abs(x[1] - rnd))
        selected_resid, selected_prob = ordered[0]
        selected_data = next(c for c in topk if c[0] == selected_resid)

        print(f"\n  Stochastically selected: {selected_resid}")
        print(f"    (cpl={selected_data[1]:.2f}, prog={selected_data[2]:.4f}, prob={selected_prob:.6f}, diff={abs(selected_prob - rnd):.6f})")

        return selected_resid

    # ------------------------------------------------------------------
    # FALLBACK (UNCHANGED STRUCTURE + NEW forward check)
    # ------------------------------------------------------------------
    def fallback_select_neighbor(self, source_resid, visited_resids, first_source_resid):
        # ensure direction exists if possible
        self._current_first_source_resid = first_source_resid
        if not self._path_initialized:
            self._init_path_direction_if_needed(first_source_resid)

        print(f"  Computing couplings for fallback source {source_resid}")
        cpl = self.avg_cpl(
            self.ham_file, source_resid,
            self.topFilePath, self.gmxPath, self.mdpFilePath
        )
        probs = self.probabilities(cpl)
        if not probs:
            return None

        sc = self.get_com(source_resid)
        sc = np.asarray(sc, dtype=float)

        rnd = random.random()
        print(f"  Fallback random: {rnd:.6f}")

        # IMPORTANT: ordering logic unchanged
        ordered = sorted(probs.items(), key=lambda x: abs(x[1] - rnd))

        for tgt, prob in ordered:
            tc = self.get_com(tgt)
            if tc is None:
                continue
            tc = np.asarray(tc, dtype=float)

            dist = float(np.linalg.norm(tc - sc))

            if self._path_dir is not None:
                prog = self._path_progress(sc, tc)
                is_forward = self._is_forward_along_path(sc, tc, eps=self.forward_eps)
            else:
                prog = 0.0
                is_forward = True

            print(f"    Resid {tgt}: prob={prob:.6f}, dist={dist:.4f}, path_prog={prog:.4f}")

            if tgt in visited_resids:
                print(f"      ❌ Visited")
                continue

            if dist > self.cutoff:
                print(f"      ❌ Too far")
                continue

            if self._path_dir is not None and not is_forward:
                print(f"      ❌ Not forward (global path)")
                continue

            print(f"      ✅ Selected")
            return tgt

        return None