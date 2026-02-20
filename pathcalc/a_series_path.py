## fix version with grain2 sampling only after crossing
import math
import random
import os
import shutil
import time
import glob
from typing import Dict, Optional

import numpy as np

from pathcalc import top, gmx, inpman, asann


class PathFinder:
    """
    Path sampling with:
      - NORMAL mode:
          * before crossing: axis-forward bias + coupling bias
          * after crossing: coupling-only   (OLD) -> now: coupling + post-cross dir bias
      - RESCUE mode:
          * geometric neighbor search within rescue_radius (no ASANN)
          * strongly axis-forward (dominant), coupling only as tie-breaker (OLD)
            -> now: after crossing, use post-cross dir forwardness instead of bias-axis forwardness
      - FALLBACK mode:
          * backtracking using same NORMAL logic (OLD)
            -> now: after crossing, also applies post-cross dir forwardness

    Logic preserved exactly from your earlier working versions,
    with only the requested post-cross direction bias added.
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
        bias_axis="y",
        rescue_radius=None,
        rescue_top_k=6,
        rescue_forward_eps=0.0,
        rescue_cutoff=None,
        rescue_max_neighbors=28,
        # ---------------- NEW ----------------
        slab_normal_axis="z",     # assume slab normal is z (common for your setups)
        post_cross_eps=-0.05,     # allow small wiggle; reject strong backward
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

        # ---------------- NEW: post-cross direction cache ----------------
        slab_normal_axis = slab_normal_axis.lower()
        if slab_normal_axis not in ("x", "y", "z"):
            raise ValueError("slab_normal_axis must be 'x', 'y', or 'z'")
        self.slab_normal_axis = slab_normal_axis
        self.post_cross_eps = post_cross_eps

        self._post_cross_dir = None               # unit vector
        self._post_cross_initialized = False      # computed once per path run
        self._post_cross_target_range = None      # which grain we entered

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

    def _axis_progress(self, start_range, src, tgt):
        if start_range == "in_range_1":
            return tgt - src
        if start_range == "in_range_2":
            return src - tgt
        return 0.0

    def _is_axis_forward(self, start_range, src, tgt, eps=0.0):
        return self._axis_progress(start_range, src, tgt) > eps

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
    # NEW: ATOM COORDS ACCESS (for long-axis PCA per residue)
    # ------------------------------------------------------------------
    def _get_residue_coords(self, resid):
        """
        Return (N,3) numpy array of atomic coords for a residue from gro_obj.

        We try a few common method names to avoid breaking your existing gro_obj.
        If none match, we raise a clear error telling what to implement.
        """
        # Most likely (based on your style)
        if hasattr(self.gro_obj, "get_residue_coords"):
            coords = self.gro_obj.get_residue_coords(resid)
            return np.asarray(coords, dtype=float)

        # Alternatives
        if hasattr(self.gro_obj, "get_res_coords"):
            coords = self.gro_obj.get_res_coords(resid)
            return np.asarray(coords, dtype=float)

        if hasattr(self.gro_obj, "residue_coords") and callable(getattr(self.gro_obj, "residue_coords")):
            coords = self.gro_obj.residue_coords(resid)
            return np.asarray(coords, dtype=float)

        if hasattr(self.gro_obj, "resid_coords"):
            coords = self.gro_obj.resid_coords.get(resid)
            if coords is not None:
                return np.asarray(coords, dtype=float)

        raise AttributeError(
            "gro_obj does not provide atomic coordinates per residue. "
            "Please add one of: get_residue_coords(resid), get_res_coords(resid), "
            "residue_coords(resid), or a dict attribute resid_coords[resid] -> (N,3) coords."
        )

    def _slab_normal_vector(self):
        if self.slab_normal_axis == "x":
            return np.array([1.0, 0.0, 0.0])
        if self.slab_normal_axis == "y":
            return np.array([0.0, 1.0, 0.0])
        return np.array([0.0, 0.0, 1.0])

    def _grain_com(self, which_range):
        pts = [self.get_com(r) for r in self.all_resids if self.check_axis_range(r) == which_range]
        pts = [p for p in pts if p is not None]
        if not pts:
            return None
        return np.mean(np.asarray(pts, dtype=float), axis=0)

    def _compute_avg_long_axis_for_grain(self, grain_range):
        """
        Compute averaged long-axis unit vector for all residues in the given grain.
        Long axis per residue is PCA principal axis of its atomic coordinates.
        Uses sign-consistent averaging to avoid cancellation.
        """
        resids = [r for r in self.all_resids if self.check_axis_range(r) == grain_range]
        if not resids:
            return None

        axes = []
        for r in resids:
            coords = self._get_residue_coords(r)
            if coords is None or len(coords) < 3:
                continue
            X = coords - coords.mean(axis=0)
            C = np.cov(X.T)
            evals, evecs = np.linalg.eigh(C)           # ascending
            v = evecs[:, np.argmax(evals)]            # principal axis
            n = np.linalg.norm(v)
            if n < 1e-12:
                continue
            v = v / n
            axes.append(v)

        if not axes:
            return None

        axes = np.asarray(axes, dtype=float)

        # sign consistency
        ref = axes[0]
        for i in range(len(axes)):
            if np.dot(axes[i], ref) < 0:
                axes[i] *= -1

        u_long = axes.mean(axis=0)
        n = np.linalg.norm(u_long)
        if n < 1e-12:
            return None
        return u_long / n

    def _init_post_cross_direction_if_needed(self, start_range, current_resid):
        """
        Initialize post-cross direction once when we first detect we are in the other grain.
        Direction = (avg long-axis of entered grain) x (slab normal), oriented away from GB
        (i.e., from start grain COM -> entered grain COM).
        """
        if self._post_cross_initialized:
            return

        target_range = self.check_axis_range(current_resid)
        if target_range not in ("in_range_1", "in_range_2") or target_range == start_range:
            return

        print("\n" + "-" * 60)
        print("[POST-CROSS DIRECTION INITIALIZATION]")
        print(f"  Start grain:   {start_range}")
        print(f"  Entered grain: {target_range}")
        print(f"  Slab normal axis: {self.slab_normal_axis}")
        print("-" * 60)

        u_long = self._compute_avg_long_axis_for_grain(target_range)
        if u_long is None:
            print("  ❌ Could not compute averaged long axis for entered grain. Post-cross bias disabled.")
            self._post_cross_initialized = True
            self._post_cross_dir = None
            self._post_cross_target_range = target_range
            return

        n_slab = self._slab_normal_vector()
        v_perp = np.cross(u_long, n_slab)
        nv = np.linalg.norm(v_perp)
        if nv < 1e-12:
            print("  ❌ u_long parallel to slab normal -> cross product ~ 0. Post-cross bias disabled.")
            self._post_cross_initialized = True
            self._post_cross_dir = None
            self._post_cross_target_range = target_range
            return
        v_perp = v_perp / nv

        # orient away from GB: from start grain COM -> entered grain COM
        com_start = self._grain_com(start_range)
        com_target = self._grain_com(target_range)

        if com_start is None or com_target is None:
            print("  ⚠️ Could not compute grain COMs reliably. Keeping v_perp sign as-is.")
        else:
            sep = com_target - com_start
            ns = np.linalg.norm(sep)
            if ns > 1e-12:
                sep = sep / ns
                if np.dot(v_perp, sep) < 0:
                    v_perp *= -1

        print(f"  Averaged long-axis u_long: {u_long}")
        print(f"  Perp direction v_perp:     {v_perp}")
        print(f"  post_cross_eps (wiggle):   {self.post_cross_eps:.4f}")
        print("-" * 60)

        self._post_cross_dir = v_perp
        self._post_cross_initialized = True
        self._post_cross_target_range = target_range

    def _post_cross_progress(self, sc, tc):
        """
        Progress along post-cross direction (entered grain transport direction).
        Positive -> moving "forward" into the grain.
        """
        if self._post_cross_dir is None:
            return 0.0
        step = tc - sc
        return float(np.dot(step, self._post_cross_dir))

    # ------------------------------------------------------------------
    # SUMMARY HELPER
    # ------------------------------------------------------------------
    def _make_step_info(self, mode, source_resid, target_resid, first_source_resid):
        sc = self.get_com(source_resid)
        tc = self.get_com(target_resid)

        dist = math.dist(sc, tc)
        src_range = self.check_axis_range(source_resid)
        tgt_range = self.check_axis_range(target_resid)
        start_range = self.check_axis_range(first_source_resid)

        axis_prog = self._axis_progress(
            start_range,
            sc[self.axis_index],
            tc[self.axis_index],
        )

        crossed = (
            src_range in ("in_range_1", "in_range_2")
            and tgt_range in ("in_range_1", "in_range_2")
            and src_range != tgt_range
        )

        return {
            "mode": mode,
            "source_resid": source_resid,
            "target_resid": target_resid,
            "distance_nm": round(dist, 4),
            "axis_progress_nm": round(axis_prog, 4),
            "source_grain": "g1" if src_range == "in_range_1" else "g2",
            "target_grain": "g1" if tgt_range == "in_range_1" else "g2",
            "crossed": "YES" if crossed else "NO",
        }

    # ------------------------------------------------------------------
    # RESCUE NEIGHBOR SEARCH (geometric, no ASANN)
    # ------------------------------------------------------------------
    def neighbors_within_radius(self, source_resid, radius_nm, max_neighbors=None):
        """
        Find neighbors within radius_nm using COM distance (geometric search).
        No ASANN algorithm used - pure distance-based filtering.

        Returns list of residue IDs sorted by distance, optionally truncated to max_neighbors.
        """
        src = self.get_com(source_resid)
        if src is None:
            return []

        candidates = []
        for r in self.all_resids:
            if r == source_resid:
                continue
            tgt = self.get_com(r)
            if tgt is None:
                continue
            d = math.dist(src, tgt)
            if d <= radius_nm:
                candidates.append((r, d))

        # Sort by distance
        candidates.sort(key=lambda x: x[1])

        # Optionally limit to max_neighbors
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
        """
        Compute couplings for an explicit list of neighbors (used in RESCUE mode).

        This bypasses ASANN and uses the provided neighbor list directly.
        """
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
    # PROBABILITIES
    # ------------------------------------------------------------------
    def probabilities(self, coupling_data, beta=1 / 25.7):
        if not coupling_data:
            return {}
        maxc = max(coupling_data.values())
        weights = {
            r: math.exp(beta * (c - maxc)) for r, c in coupling_data.items()
        }
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
        sc = self.get_com(source_resid)
        start_range = self.check_axis_range(first_source_resid)
        crossed = self._has_crossed(start_range, source_resid)
        use_axis = not crossed  # OLD meaning: only before crossing

        # NEW: initialize post-cross direction once (first time we are in the other grain)
        if crossed and not self._post_cross_initialized:
            self._init_post_cross_direction_if_needed(start_range, source_resid)

        print(f"\n{'='*60}")
        print(f"SELECTING NEIGHBOR FOR SOURCE RESID: {source_resid}")
        print(f"Start range: {start_range}, Crossed: {crossed}, Use axis: {use_axis}")
        if crossed:
            print(f"Post-cross dir initialized: {self._post_cross_initialized}")
            print(f"Post-cross dir: {self._post_cross_dir}")
        print(f"{'='*60}")

        # NORMAL
        if probabilities:
            rnd = random.random()
            print(f"\n[NORMAL MODE] Random number: {rnd:.6f}")
            print(f"\nAll neighbors with probabilities:")
            for tgt, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                print(f"  Resid {tgt}: prob={prob:.6f}")

            ordered = sorted(probabilities.items(), key=lambda x: abs(x[1] - rnd))
            print(f"\nChecking neighbors (ordered by closest prob to random {rnd:.6f}):")

            for tgt, prob in ordered:
                tc = self.get_com(tgt)
                dist = math.dist(sc, tc)
                axis_prog = self._axis_progress(start_range, sc[self.axis_index], tc[self.axis_index])
                is_forward = self._is_axis_forward(start_range, sc[self.axis_index], tc[self.axis_index])

                post_prog = None
                post_forward = None
                if crossed:
                    post_prog = self._post_cross_progress(np.asarray(sc), np.asarray(tc))
                    post_forward = (post_prog > self.post_cross_eps)

                print(f"\n  Resid {tgt}: prob={prob:.6f}, diff={abs(prob - rnd):.6f}")
                print(f"    Distance: {dist:.4f} nm (cutoff: {self.cutoff:.4f})")
                print(f"    Axis progress: {axis_prog:.4f} nm, Forward: {is_forward}")
                if crossed:
                    print(f"    Post-cross progress: {post_prog:.4f} nm, Forward: {post_forward} (eps={self.post_cross_eps:.4f})")

                if tgt in visited_resids:
                    print(f"    ❌ REJECTED: Already visited")
                    continue

                if dist > self.cutoff:
                    print(f"    ❌ REJECTED: Distance > cutoff")
                    continue

                # BEFORE crossing: same old axis-forward condition
                if use_axis:
                    if not is_forward:
                        print(f"    ❌ REJECTED: Not forward along axis")
                        continue

                # AFTER crossing: NEW directional constraint (soft forward)
                if crossed and self._post_cross_dir is not None:
                    if not post_forward:
                        print(f"    ❌ REJECTED: Not forward along post-cross direction")
                        continue

                print(f"    ✅ SELECTED")
                return tgt, self._make_step_info("NORMAL", source_resid, tgt, first_source_resid)

            print(f"\n❌ No valid neighbors in NORMAL mode")

        # RESCUE
        print(f"\n[RESCUE MODE]")
        print(f"Searching for neighbors within rescue_radius={self.rescue_radius:.4f} nm (max={self.rescue_max_neighbors})")

        # Find neighbors geometrically within rescue_radius
        radius_neighbors = self.neighbors_within_radius(
            source_resid,
            radius_nm=self.rescue_radius,
            max_neighbors=self.rescue_max_neighbors
        )

        # Filter out visited
        radius_neighbors = [r for r in radius_neighbors if r not in visited_resids]

        if not radius_neighbors:
            print(f"❌ No unvisited neighbors within rescue radius")
        else:
            print(f"Computing couplings for {len(radius_neighbors)} rescue candidates...")
            # Compute couplings for the radius-based neighbor list
            cpl = self.avg_cpl_for_neighbors(
                self.ham_file, source_resid, radius_neighbors,
                self.topFilePath, self.gmxPath, self.mdpFilePath
            )

            if cpl:
                print(f"Rescue mode coupling values (meV):")
                for r, c in sorted(cpl.items(), key=lambda x: x[1], reverse=True):
                    print(f"  Resid {r}: {c:.2f} meV")

                rescued = self.rescue_select_neighbor_axis_dominant(
                    cpl, source_resid, visited_resids, start_range, crossed=crossed
                )
                if rescued:
                    print(f"✅ RESCUE selected: {rescued}")
                    return rescued, self._make_step_info(
                        "RESCUE", source_resid, rescued, first_source_resid
                    )

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
            nxt = self.fallback_select_neighbor(
                fb, visited_resids, first_source_resid
            )
            if nxt:
                print(f"✅ FALLBACK selected: {nxt}")
                return nxt, self._make_step_info(
                    "FALLBACK", fb, nxt, first_source_resid
                )
            print(f"❌ No valid neighbor from {fb}, continuing backtrack")

        print(f"\n❌ ALL MODES FAILED")
        return None

    # ------------------------------------------------------------------
    # RESCUE (AXIS-DOMINANT WITH STOCHASTIC SELECTION)
    # ------------------------------------------------------------------
    def rescue_select_neighbor_axis_dominant(
        self, coupling_data, source_resid, visited_resids, start_range, crossed=False
    ):
        if not coupling_data:
            return None

        sc = self.get_com(source_resid)
        src_coord = sc[self.axis_index]

        if crossed and not self._post_cross_initialized:
            # ensure direction exists even if rescue called immediately after crossing
            self._init_post_cross_direction_if_needed(start_range, source_resid)

        print(f"  Filtering candidates (cutoff={self.rescue_cutoff:.4f}, eps={self.rescue_forward_eps:.4f}):")
        candidates = []
        for r, cpl in coupling_data.items():
            tc = self.get_com(r)
            dist = math.dist(sc, tc)

            axis_prog = self._axis_progress(start_range, src_coord, tc[self.axis_index])
            is_forward_axis = self._is_axis_forward(start_range, src_coord, tc[self.axis_index], self.rescue_forward_eps)

            post_prog = None
            is_forward_post = None
            if crossed and self._post_cross_dir is not None:
                post_prog = self._post_cross_progress(np.asarray(sc), np.asarray(tc))
                is_forward_post = (post_prog > self.post_cross_eps)

            if crossed and self._post_cross_dir is not None:
                print(f"    Resid {r}: cpl={cpl:.2f}, dist={dist:.4f}, post_prog={post_prog:.4f}")
            else:
                print(f"    Resid {r}: cpl={cpl:.2f}, dist={dist:.4f}, prog={axis_prog:.4f}")

            if r in visited_resids:
                print(f"      ❌ Visited")
                continue

            if dist > self.rescue_cutoff:
                print(f"      ❌ Too far")
                continue

            # BEFORE crossing: original forward check on bias axis
            if not crossed:
                if not is_forward_axis:
                    print(f"      ❌ Not forward")
                    continue

            # AFTER crossing: forward check on post-cross direction
            if crossed and self._post_cross_dir is not None:
                if not is_forward_post:
                    print(f"      ❌ Not forward (post-cross)")
                    continue

            print(f"      ✓ Added to candidates")

            # keep "axis_prog" slot as selection metric for rescue stochasticity:
            # - before crossing: axis_prog (as before)
            # - after crossing: post_prog (so rescue is direction-dominant in entered grain too)
            prog_for_sampling = post_prog if (crossed and self._post_cross_dir is not None) else axis_prog
            candidates.append((r, cpl, prog_for_sampling))

        if not candidates:
            return None

        # Sort by coupling (descending) and take top-k (UNCHANGED)
        candidates.sort(key=lambda x: x[1], reverse=True)
        topk = candidates[: self.rescue_top_k]

        print(f"\n  Top-{self.rescue_top_k} candidates by coupling:")
        for r, cpl, prog in topk:
            print(f"    Resid {r}: cpl={cpl:.2f}, prog={prog:.4f}")

        # Create probabilities based on progress (UNCHANGED pattern)
        progress_values = {c[0]: c[2] for c in topk}  # {resid: progress}

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
        print(f"    (cpl={selected_data[1]:.2f}, prog={selected_data[2]:.4f}, prob={selected_prob:.6f}, diff from random={abs(selected_prob - rnd):.6f})")

        return selected_resid

    # ------------------------------------------------------------------
    # FALLBACK (UNCHANGED LOGIC + NEW post-cross forward check)
    # ------------------------------------------------------------------
    def fallback_select_neighbor(self, source_resid, visited_resids, first_source_resid):
        print(f"  Computing couplings for fallback source {source_resid}")
        cpl = self.avg_cpl(
            self.ham_file, source_resid,
            self.topFilePath, self.gmxPath, self.mdpFilePath
        )
        probs = self.probabilities(cpl)
        if not probs:
            return None

        sc = self.get_com(source_resid)
        start_range = self.check_axis_range(first_source_resid)
        crossed = self._has_crossed(start_range, source_resid)
        use_axis = not crossed

        # NEW: initialize post-cross direction if needed
        if crossed and not self._post_cross_initialized:
            self._init_post_cross_direction_if_needed(start_range, source_resid)

        rnd = random.random()
        print(f"  Fallback random: {rnd:.6f}, use_axis: {use_axis}")

        ordered = sorted(probs.items(), key=lambda x: abs(x[1] - rnd))

        for tgt, prob in ordered:
            tc = self.get_com(tgt)
            dist = math.dist(sc, tc)

            print(f"    Resid {tgt}: prob={prob:.6f}, dist={dist:.4f}")

            if tgt in visited_resids:
                print(f"      ❌ Visited")
                continue

            if dist > self.cutoff:
                print(f"      ❌ Too far")
                continue

            if use_axis:
                if not self._is_axis_forward(
                    start_range,
                    sc[self.axis_index],
                    tc[self.axis_index],
                ):
                    print(f"      ❌ Not forward")
                    continue

            # NEW: after crossing, also reject strong backward along post-cross direction
            if crossed and self._post_cross_dir is not None:
                post_prog = self._post_cross_progress(np.asarray(sc), np.asarray(tc))
                if post_prog <= self.post_cross_eps:
                    print(f"      ❌ Not forward (post-cross), post_prog={post_prog:.4f}")
                    continue

            print(f"      ✅ Selected")
            return tgt

        return None






## version with bias axis --working correctly but moves back and forth in g2
# import math
# import random
# import os
# import shutil
# import time
# import glob
# from typing import Dict, Optional

# from pathcalc import top, gmx, inpman, asann


# class PathFinder:
#     """
#     Path sampling with:
#       - NORMAL mode:
#           * before crossing: axis-forward bias + coupling bias
#           * after crossing: coupling-only
#       - RESCUE mode:
#           * geometric neighbor search within rescue_radius (no ASANN)
#           * strongly axis-forward (dominant), coupling only as tie-breaker
#       - FALLBACK mode:
#           * backtracking using same NORMAL logic

#     Logic preserved exactly from your earlier working versions.
#     """

#     # ------------------------------------------------------------------
#     # INIT
#     # ------------------------------------------------------------------
#     def __init__(
#         self,
#         gro_obj,
#         all_coms,
#         all_resids,
#         ham_file,
#         topFilePath,
#         gmxPath,
#         mdpFilePath,
#         axis_ranges,
#         cutoff,
#         bias_axis="y",
#         rescue_radius=None,
#         rescue_top_k=6,
#         rescue_forward_eps=0.0,
#         rescue_cutoff=None,
#         rescue_max_neighbors=28,
#     ):
#         self.gro_obj = gro_obj
#         self.all_coms = all_coms
#         self.all_resids = all_resids
#         self.coms = {r: c for r, c in zip(all_resids, all_coms)}

#         self.ham_file = ham_file
#         self.topFilePath = topFilePath
#         self.mdpFilePath = mdpFilePath
#         self.gmxPath = gmxPath

#         self.axis_ranges = axis_ranges
#         self.cutoff = cutoff

#         axis = bias_axis.lower()
#         self.bias_axis = axis
#         self.axis_index = {"x": 0, "y": 1, "z": 2}[axis]

#         self.rescue_radius = rescue_radius if rescue_radius is not None else cutoff * 1.5
#         self.rescue_cutoff = rescue_cutoff if rescue_cutoff is not None else self.rescue_radius
#         self.rescue_top_k = rescue_top_k
#         self.rescue_forward_eps = rescue_forward_eps
#         self.rescue_max_neighbors = rescue_max_neighbors

#     # ------------------------------------------------------------------
#     # BASIC HELPERS
#     # ------------------------------------------------------------------
#     def get_com(self, resid):
#         return self.coms.get(resid)

#     def check_axis_range(self, resid):
#         com = self.get_com(resid)
#         if com is None:
#             return None
#         coord = com[self.axis_index]
#         r1, r2 = self.axis_ranges
#         if r1[0] <= coord <= r1[1]:
#             return "in_range_1"
#         if r2[0] <= coord <= r2[1]:
#             return "in_range_2"
#         return "out_of_range"

#     def _axis_progress(self, start_range, src, tgt):
#         if start_range == "in_range_1":
#             return tgt - src
#         if start_range == "in_range_2":
#             return src - tgt
#         return 0.0

#     def _is_axis_forward(self, start_range, src, tgt, eps=0.0):
#         return self._axis_progress(start_range, src, tgt) > eps

#     def _has_crossed(self, start_range, resid):
#         curr = self.check_axis_range(resid)
#         return curr in ("in_range_1", "in_range_2") and curr != start_range
    
#     def cleanup_specific_files(self, keep_extensions=[".txt"]):
#         for filename in os.listdir():
#             if os.path.isdir(filename) or any(filename.endswith(ext) for ext in keep_extensions):
#                 continue
#             try:
#                 os.remove(filename)
#             except Exception as e:
#                 print(f"Could not delete {filename}: {e}")

#     # ------------------------------------------------------------------
#     # SUMMARY HELPER
#     # ------------------------------------------------------------------
#     def _make_step_info(self, mode, source_resid, target_resid, first_source_resid):
#         sc = self.get_com(source_resid)
#         tc = self.get_com(target_resid)

#         dist = math.dist(sc, tc)
#         src_range = self.check_axis_range(source_resid)
#         tgt_range = self.check_axis_range(target_resid)
#         start_range = self.check_axis_range(first_source_resid)

#         axis_prog = self._axis_progress(
#             start_range,
#             sc[self.axis_index],
#             tc[self.axis_index],
#         )

#         crossed = (
#             src_range in ("in_range_1", "in_range_2")
#             and tgt_range in ("in_range_1", "in_range_2")
#             and src_range != tgt_range
#         )

#         return {
#             "mode": mode,
#             "source_resid": source_resid,
#             "target_resid": target_resid,
#             "distance_nm": round(dist, 4),
#             "axis_progress_nm": round(axis_prog, 4),
#             "source_grain": "g1" if src_range == "in_range_1" else "g2",
#             "target_grain": "g1" if tgt_range == "in_range_1" else "g2",
#             "crossed": "YES" if crossed else "NO",
#         }

#     # ------------------------------------------------------------------
#     # RESCUE NEIGHBOR SEARCH (geometric, no ASANN)
#     # ------------------------------------------------------------------
#     def neighbors_within_radius(self, source_resid, radius_nm, max_neighbors=None):
#         """
#         Find neighbors within radius_nm using COM distance (geometric search).
#         No ASANN algorithm used - pure distance-based filtering.
        
#         Returns list of residue IDs sorted by distance, optionally truncated to max_neighbors.
#         """
#         src = self.get_com(source_resid)
#         if src is None:
#             return []

#         candidates = []
#         for r in self.all_resids:
#             if r == source_resid:
#                 continue
#             tgt = self.get_com(r)
#             if tgt is None:
#                 continue
#             d = math.dist(src, tgt)
#             if d <= radius_nm:
#                 candidates.append((r, d))

#         # Sort by distance
#         candidates.sort(key=lambda x: x[1])
        
#         # Optionally limit to max_neighbors
#         if max_neighbors is not None and max_neighbors > 0:
#             candidates = candidates[:max_neighbors]

#         neighbor_resids = [r for r, _ in candidates]
#         print(f"  Found {len(neighbor_resids)} neighbors within {radius_nm:.4f} nm")
#         if neighbor_resids:
#             print(f"  Neighbor resids: {neighbor_resids[:10]}{'...' if len(neighbor_resids) > 10 else ''}")
        
#         return neighbor_resids

#     # ------------------------------------------------------------------
#     # COUPLING CALCULATION (NORMAL MODE - UNCHANGED LOGIC)
#     # ------------------------------------------------------------------
#     def avg_cpl(
#         self,
#         ham_file: str,
#         source_resid: int,
#         topFilePath: str,
#         gmxPath: str,
#         mdpFilePath: str,
#     ) -> Optional[Dict[int, float]]:

#         nn = self.gro_obj.cutAroundRes(
#             source_resid, [1.0, 1.0, 1.0], allCOMs=self.all_coms
#         )
#         if not nn:
#             return None

#         nn_finder = asann.FirstNN(self.all_coms, self.all_resids, subset_resids=nn)
#         neighbors = nn_finder.nearest_neighbors_asann(source_resid)
#         if not neighbors:
#             return None

#         resids_to_write = [source_resid] + neighbors
#         self.gro_obj.write_gro("temporary.gro", resids_to_write)

#         gmx_runner = gmx.gmx(exe="gmx", gro="temporary.gro")
#         gmx_runner.renum()

#         renumbered_resids = self.gro_obj.renumbered_resids("temporary.gro")
#         resid_mapping = dict(zip(resids_to_write, renumbered_resids))

#         new_top_path = "pen-esp.top"
#         shutil.copy(topFilePath, new_top_path)
#         itp_path = "/data/sgarg/pentacene/sampling_qm_zone_algorithm/gb-pen/input_files"
#         itp_files = os.path.join(itp_path, "*.itp")
#         for itp_file in glob.glob(itp_files):
#             shutil.copy(itp_file, ".")
#         pen_top = top(new_top_path)
#         pen_top.update_molecule_count("molecule", len(resids_to_write))

#         gmx_runner = gmx.gmx(
#             exe=gmxPath,
#             gro="temporary.gro",
#             top="pen-esp.top",
#             mdp=mdpFilePath,
#             tpr="ham",
#         )
#         gmx_runner.grompp()

#         if not os.path.exists("pen.spec"):
#             spec = inpman.inpManager("spec")
#             spec.update(natoms=36, nelectrons=102, nallorbitals=102,
#                         nfragorbs=1, fragorbs=51)
#             spec.save("pen.spec")

#         ct = inpman.inpManager("ct")
#         ct.update(
#             sites=[resid_mapping[r] for r in resids_to_write],
#             seed=random.randint(1, 100),
#             chargecarrier="hole",
#             atomindex=[1, 20, 29],
#             typefiles="pen.spec",
#             jobtype="NOM",
#             internalrelax="onsite",
#         )
#         ct.save("charge-transfer.dat")

#         gmx.gmx(exe=gmxPath, tpr="ham").mdrun(nt=1)

#         if not os.path.exists(ham_file):
#             return None

#         extracted = {}
#         with open(ham_file) as f:
#             for line in f:
#                 if line.startswith(("#", "@")):
#                     continue
#                 parts = line.split()
#                 vals = [abs(float(v)) * 1000 for v in parts[2:2 + len(neighbors)]]
#                 for r, v in zip(neighbors, vals):
#                     extracted[r] = v
                    
#         self.cleanup_specific_files()
        
#         return extracted

#     # ------------------------------------------------------------------
#     # COUPLING CALCULATION FOR EXPLICIT NEIGHBOR LIST (RESCUE MODE)
#     # ------------------------------------------------------------------
#     def avg_cpl_for_neighbors(self, ham_file, source_resid, neighbors, topFilePath, gmxPath, mdpFilePath):
#         """
#         Compute couplings for an explicit list of neighbors (used in RESCUE mode).
        
#         This bypasses ASANN and uses the provided neighbor list directly.
#         """
#         neighbors = list(neighbors) if neighbors else []
#         if not neighbors:
#             return None

#         resids_to_write = [source_resid] + neighbors
#         self.gro_obj.write_gro("temporary.gro", resids_to_write)

#         gmx_runner = gmx.gmx(exe="gmx", gro="temporary.gro")
#         gmx_runner.renum()

#         renumbered_resids = self.gro_obj.renumbered_resids("temporary.gro")
#         resid_mapping = dict(zip(resids_to_write, renumbered_resids))

#         new_top_path = "pen-esp.top"
#         shutil.copy(topFilePath, new_top_path)
#         itp_path = "/data/sgarg/pentacene/sampling_qm_zone_algorithm/gb-pen/input_files"
#         for itp_file in glob.glob(os.path.join(itp_path, "*.itp")):
#             shutil.copy(itp_file, ".")
#         pen_top = top(new_top_path)
#         pen_top.update_molecule_count("molecule", len(resids_to_write))

#         gmx_runner = gmx.gmx(
#             exe=gmxPath,
#             gro="temporary.gro",
#             top="pen-esp.top",
#             mdp=mdpFilePath,
#             tpr="ham",
#         )
#         gmx_runner.grompp()

#         if not os.path.exists("pen.spec"):
#             spec = inpman.inpManager("spec")
#             spec.update(natoms=36, nelectrons=102, nallorbitals=102,
#                         nfragorbs=1, fragorbs=51)
#             spec.save("pen.spec")

#         ct = inpman.inpManager("ct")
#         ct.update(
#             sites=[resid_mapping[r] for r in resids_to_write],
#             seed=random.randint(1, 100),
#             chargecarrier="hole",
#             atomindex=[1, 20, 29],
#             typefiles="pen.spec",
#             jobtype="NOM",
#             internalrelax="onsite",
#         )
#         ct.save("charge-transfer.dat")

#         gmx.gmx(exe=gmxPath, tpr="ham").mdrun(nt=1)

#         if not os.path.exists(ham_file):
#             return None

#         extracted = {}
#         with open(ham_file) as f:
#             for line in f:
#                 if line.startswith(("#", "@")):
#                     continue
#                 parts = line.split()
#                 vals = [abs(float(v)) * 1000 for v in parts[2:2 + len(neighbors)]]
#                 for r, v in zip(neighbors, vals):
#                     extracted[r] = v
                    
#         self.cleanup_specific_files()
        
#         return extracted

#     # ------------------------------------------------------------------
#     # PROBABILITIES
#     # ------------------------------------------------------------------
#     def probabilities(self, coupling_data, beta=1 / 25.7):
#         if not coupling_data:
#             return {}
#         maxc = max(coupling_data.values())
#         weights = {
#             r: math.exp(beta * (c - maxc)) for r, c in coupling_data.items()
#         }
#         Z = sum(weights.values())
#         return {r: w / Z for r, w in weights.items()} if Z > 0 else {}

#     # ------------------------------------------------------------------
#     # MAIN STEP SELECTION
#     # ------------------------------------------------------------------
#     def select_next_neighbor(
#         self,
#         probabilities,
#         source_resid,
#         first_source_resid,
#         visited_resids,
#         sampled_paths,
#     ):
#         sc = self.get_com(source_resid)
#         start_range = self.check_axis_range(first_source_resid)
#         crossed = self._has_crossed(start_range, source_resid)
#         use_axis = not crossed

#         print(f"\n{'='*60}")
#         print(f"SELECTING NEIGHBOR FOR SOURCE RESID: {source_resid}")
#         print(f"Start range: {start_range}, Crossed: {crossed}, Use axis: {use_axis}")
#         print(f"{'='*60}")

#         # NORMAL
#         if probabilities:
#             rnd = random.random()
#             print(f"\n[NORMAL MODE] Random number: {rnd:.6f}")
#             print(f"\nAll neighbors with probabilities:")
#             for tgt, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
#                 print(f"  Resid {tgt}: prob={prob:.6f}")
            
#             ordered = sorted(probabilities.items(), key=lambda x: abs(x[1] - rnd))
#             print(f"\nChecking neighbors (ordered by closest prob to random {rnd:.6f}):")
            
#             for tgt, prob in ordered:
#                 tc = self.get_com(tgt)
#                 dist = math.dist(sc, tc)
#                 axis_prog = self._axis_progress(start_range, sc[self.axis_index], tc[self.axis_index])
#                 is_forward = self._is_axis_forward(start_range, sc[self.axis_index], tc[self.axis_index])
                
#                 print(f"\n  Resid {tgt}: prob={prob:.6f}, diff={abs(prob - rnd):.6f}")
#                 print(f"    Distance: {dist:.4f} nm (cutoff: {self.cutoff:.4f})")
#                 print(f"    Axis progress: {axis_prog:.4f} nm, Forward: {is_forward}")
                
#                 if tgt in visited_resids:
#                     print(f"    ❌ REJECTED: Already visited")
#                     continue
                    
#                 if dist > self.cutoff:
#                     print(f"    ❌ REJECTED: Distance > cutoff")
#                     continue
                    
#                 if use_axis:
#                     if not is_forward:
#                         print(f"    ❌ REJECTED: Not forward along axis")
#                         continue
                
#                 print(f"    ✅ SELECTED")
#                 return tgt, self._make_step_info("NORMAL", source_resid, tgt, first_source_resid)
            
#             print(f"\n❌ No valid neighbors in NORMAL mode")

#         # RESCUE
#         print(f"\n[RESCUE MODE]")
#         print(f"Searching for neighbors within rescue_radius={self.rescue_radius:.4f} nm (max={self.rescue_max_neighbors})")
        
#         # Find neighbors geometrically within rescue_radius
#         radius_neighbors = self.neighbors_within_radius(
#             source_resid,
#             radius_nm=self.rescue_radius,
#             max_neighbors=self.rescue_max_neighbors
#         )
        
#         # Filter out visited
#         radius_neighbors = [r for r in radius_neighbors if r not in visited_resids]
        
#         if not radius_neighbors:
#             print(f"❌ No unvisited neighbors within rescue radius")
#         else:
#             print(f"Computing couplings for {len(radius_neighbors)} rescue candidates...")
#             # Compute couplings for the radius-based neighbor list
#             cpl = self.avg_cpl_for_neighbors(
#                 self.ham_file, source_resid, radius_neighbors,
#                 self.topFilePath, self.gmxPath, self.mdpFilePath
#             )
            
#             if cpl:
#                 print(f"Rescue mode coupling values (meV):")
#                 for r, c in sorted(cpl.items(), key=lambda x: x[1], reverse=True):
#                     print(f"  Resid {r}: {c:.2f} meV")
                
#                 rescued = self.rescue_select_neighbor_axis_dominant(
#                     cpl, source_resid, visited_resids, start_range
#                 )
#                 if rescued:
#                     print(f"✅ RESCUE selected: {rescued}")
#                     return rescued, self._make_step_info(
#                         "RESCUE", source_resid, rescued, first_source_resid
#                     )
        
#         print(f"❌ RESCUE failed")

#         # FALLBACK
#         print(f"\n[FALLBACK MODE - Backtracking]")
#         while sampled_paths:
#             sampled_paths.pop()
#             if not sampled_paths:
#                 print(f"❌ No more paths to backtrack")
#                 return None
#             fb = sampled_paths[-1]
#             print(f"Backtracking to resid {fb}")
#             nxt = self.fallback_select_neighbor(
#                 fb, visited_resids, first_source_resid
#             )
#             if nxt:
#                 print(f"✅ FALLBACK selected: {nxt}")
#                 return nxt, self._make_step_info(
#                     "FALLBACK", fb, nxt, first_source_resid
#                 )
#             print(f"❌ No valid neighbor from {fb}, continuing backtrack")

#         print(f"\n❌ ALL MODES FAILED")
#         return None

#     # ------------------------------------------------------------------
#     # RESCUE (AXIS-DOMINANT WITH STOCHASTIC SELECTION)
#     # ------------------------------------------------------------------
#     def rescue_select_neighbor_axis_dominant(
#         self, coupling_data, source_resid, visited_resids, start_range
#     ):
#         if not coupling_data:
#             return None

#         sc = self.get_com(source_resid)
#         src_coord = sc[self.axis_index]

#         print(f"  Filtering candidates (cutoff={self.rescue_cutoff:.4f}, eps={self.rescue_forward_eps:.4f}):")
#         candidates = []
#         for r, cpl in coupling_data.items():
#             tc = self.get_com(r)
#             dist = math.dist(sc, tc)
#             axis_prog = self._axis_progress(start_range, src_coord, tc[self.axis_index])
#             is_forward = self._is_axis_forward(start_range, src_coord, tc[self.axis_index], self.rescue_forward_eps)
            
#             print(f"    Resid {r}: cpl={cpl:.2f}, dist={dist:.4f}, prog={axis_prog:.4f}")
            
#             if r in visited_resids:
#                 print(f"      ❌ Visited")
#                 continue
            
#             if dist > self.rescue_cutoff:
#                 print(f"      ❌ Too far")
#                 continue
            
#             if not is_forward:
#                 print(f"      ❌ Not forward")
#                 continue
            
#             print(f"      ✓ Added to candidates")
#             candidates.append((r, cpl, axis_prog))

#         if not candidates:
#             return None

#         # Sort by coupling (descending) and take top-k
#         candidates.sort(key=lambda x: x[1], reverse=True)
#         topk = candidates[: self.rescue_top_k]
        
#         print(f"\n  Top-{self.rescue_top_k} candidates by coupling:")
#         for r, cpl, prog in topk:
#             print(f"    Resid {r}: cpl={cpl:.2f}, prog={prog:.4f}")
        
#         # Create probabilities based on axis progress (like NORMAL mode uses coupling probabilities)
#         # Higher progress = higher probability
#         progress_values = {c[0]: c[2] for c in topk}  # {resid: progress}
        
#         # Normalize to probabilities
#         total_progress = sum(progress_values.values())
#         if total_progress <= 0:
#             # Fallback: equal probability if all progress is 0 or negative
#             probabilities = {r: 1.0 / len(topk) for r in progress_values.keys()}
#         else:
#             probabilities = {r: prog / total_progress for r, prog in progress_values.items()}
        
#         # Stochastic selection (same logic as NORMAL mode)
#         rnd = random.random()
#         print(f"\n  Rescue random number: {rnd:.6f}")
#         print(f"  Progress-based probabilities:")
#         for r, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
#             cpl_val = next(c[1] for c in topk if c[0] == r)
#             prog_val = progress_values[r]
#             print(f"    Resid {r}: prob={prob:.6f}, cpl={cpl_val:.2f}, prog={prog_val:.4f}")
        
#         # Find closest probability to random number
#         ordered = sorted(probabilities.items(), key=lambda x: abs(x[1] - rnd))
#         selected_resid, selected_prob = ordered[0]
        
#         # Get details for selected
#         selected_data = next(c for c in topk if c[0] == selected_resid)
        
#         print(f"\n  Stochastically selected: {selected_resid}")
#         print(f"    (cpl={selected_data[1]:.2f}, prog={selected_data[2]:.4f}, prob={selected_prob:.6f}, diff from random={abs(selected_prob - rnd):.6f})")
        
#         return selected_resid

#     # ------------------------------------------------------------------
#     # FALLBACK (UNCHANGED LOGIC)
#     # ------------------------------------------------------------------
#     def fallback_select_neighbor(self, source_resid, visited_resids, first_source_resid):
#         print(f"  Computing couplings for fallback source {source_resid}")
#         cpl = self.avg_cpl(
#             self.ham_file, source_resid,
#             self.topFilePath, self.gmxPath, self.mdpFilePath
#         )
#         probs = self.probabilities(cpl)
#         if not probs:
#             return None

#         sc = self.get_com(source_resid)
#         start_range = self.check_axis_range(first_source_resid)
#         crossed = self._has_crossed(start_range, source_resid)
#         use_axis = not crossed

#         rnd = random.random()
#         print(f"  Fallback random: {rnd:.6f}, use_axis: {use_axis}")
        
#         ordered = sorted(probs.items(), key=lambda x: abs(x[1] - rnd))

#         for tgt, prob in ordered:
#             tc = self.get_com(tgt)
#             dist = math.dist(sc, tc)
            
#             print(f"    Resid {tgt}: prob={prob:.6f}, dist={dist:.4f}")
            
#             if tgt in visited_resids:
#                 print(f"      ❌ Visited")
#                 continue
            
#             if dist > self.cutoff:
#                 print(f"      ❌ Too far")
#                 continue
            
#             if use_axis:
#                 if not self._is_axis_forward(
#                     start_range,
#                     sc[self.axis_index],
#                     tc[self.axis_index],
#                 ):
#                     print(f"      ❌ Not forward")
#                     continue
            
#             print(f"      ✅ Selected")
#             return tgt

#         return None





#---------------------------------------------#
##version using COM vector
# import math
# import random
# import os
# import shutil
# import time
# import glob
# from pathcalc import top, gmx, inpman, asann


# class DirectionalPathFinder:
#     """
#     Biased random-walk path sampling across a grain boundary using a 2-stage decision rule.

#     Stage A (NORMAL):
#         - Compute couplings for ASANN first-nearest neighbors
#         - Convert couplings -> probabilities
#         - Apply directional penalty: p_eff = p_cpl * cos(theta)^direction_alpha
#         - Enforce:
#             * forward constraint: cos(theta) >= min_cos_forward
#             * distance constraint: step distance <= cutoff

#     Stage B (RESCUE) (only if Stage A has no valid candidates):
#         - Expand candidate neighbors using COM-distance within rescue_radius_nm
#           (no ASANN used here)
#         - Compute couplings for this explicit neighbor list
#         - Convert couplings -> probabilities
#         - Direction dominates:
#             p_eff = (p_cpl ^ rescue_cpl_lambda) * max(cos(theta), 0) ^ rescue_dir_power
#         - Enforce:
#             * forward constraint: cos(theta) >= rescue_min_cos_forward
#             * distance constraint: step distance <= rescue_radius_nm

#     If both fail, the selection function backtracks using your existing logic.

#     Summary bookkeeping:
#         - self.last_decision is set on every decision/stop
#         - self.decision_history appends a record each time (optional use by caller)
#     """

#     def __init__(
#         self,
#         gro_obj,
#         all_coms: list,
#         all_resids,
#         ham_file,
#         topFilePath,
#         gmxPath,
#         mdpFilePath,
#         gb_vector,
#         g1_resids,
#         g2_resids,
#         cutoff: float,
#         min_cos_forward: float = 0.0,
#         direction_alpha: float = 4.0,
#         # Rescue-mode controls
#         rescue_radius_nm: float = 2.0,
#         rescue_max_neighbors: int = 28,
#         rescue_dir_power: float = 8.0,
#         rescue_cpl_lambda: float = 0.2,
#         rescue_min_cos_forward: float | None = None,
#     ):
#         # --- Core data ---
#         self.gro_obj = gro_obj
#         self.all_coms = all_coms
#         self.all_resids = all_resids
#         self.coms = {resid: com for resid, com in zip(all_resids, all_coms)}

#         # --- I/O paths for coupling evaluation ---
#         self.ham_file = ham_file
#         self.topFilePath = topFilePath
#         self.mdpFilePath = mdpFilePath
#         self.gmxPath = gmxPath

#         # --- Step constraints ---
#         self.cutoff = float(cutoff)

#         # --- Grain membership ---
#         self.g1 = set(int(r) for r in g1_resids)
#         self.g2 = set(int(r) for r in g2_resids)

#         # --- GB direction (unit vector) ---
#         if gb_vector is None:
#             raise ValueError("gb_vector must be provided.")
#         gx, gy, gz = gb_vector
#         norm = math.sqrt(gx * gx + gy * gy + gz * gz)
#         if norm == 0:
#             raise ValueError("gb_vector has zero length.")
#         self.gb_vector = (gx / norm, gy / norm, gz / norm)

#         # --- NORMAL mode directional parameters ---
#         self.min_cos_forward = float(min_cos_forward)
#         self.direction_alpha = float(direction_alpha)

#         # --- RESCUE mode parameters ---
#         self.rescue_radius_nm = float(rescue_radius_nm)
#         self.rescue_max_neighbors = int(rescue_max_neighbors)
#         self.rescue_dir_power = float(rescue_dir_power)
#         self.rescue_cpl_lambda = float(rescue_cpl_lambda)
#         self.rescue_min_cos_forward = (
#             float(rescue_min_cos_forward)
#             if rescue_min_cos_forward is not None
#             else float(min_cos_forward)
#         )

#         # --- Summary bookkeeping (used by main script for per-step summary file) ---
#         self.last_decision = None         # dict set on every decision/stop
#         self.decision_history = []        # optional history

#     # ---------------------------------------------------------------------
#     # Basic helpers
#     # ---------------------------------------------------------------------

#     def _set_decision(self, payload: dict) -> None:
#         """Store the last decision and append to the in-memory history."""
#         self.last_decision = payload
#         self.decision_history.append(payload)

#     def get_com(self, resid):
#         """Return COM for resid, with a debug print if missing."""
#         com = self.coms.get(resid)
#         if com is None:
#             print(f"Resid {resid} not found in all_coms.")
#         return com

#     def grain_of_resid(self, resid):
#         """Return 'g1', 'g2', or 'none' for a resid."""
#         if resid in self.g1:
#             return "g1"
#         if resid in self.g2:
#             return "g2"
#         return "none"

#     def cleanup_specific_files(self, keep_extensions=(".txt",)):
#         """
#         Remove all files in current working directory except:
#           - directories
#           - files with extensions in keep_extensions
#         """
#         for filename in os.listdir():
#             if os.path.isdir(filename):
#                 continue
#             if any(filename.endswith(ext) for ext in keep_extensions):
#                 continue
#             try:
#                 os.remove(filename)
#             except Exception as e:
#                 print(f"Could not delete {filename}: {e}")

#     # ---------------------------------------------------------------------
#     # Neighborhood generation (RESCUE uses this; NORMAL uses ASANN)
#     # ---------------------------------------------------------------------

#     def neighbors_within_radius(self, source_resid: int, radius_nm: float, max_neighbors: int | None = None):
#         """
#         Collect residues within radius_nm (COM-distance) of source_resid.

#         - Pure geometric filtering (no ASANN)
#         - Sorted by distance
#         - Optionally truncated to max_neighbors
#         """
#         src = self.get_com(source_resid)
#         if src is None:
#             return []

#         candidates = []
#         for r in self.all_resids:
#             if r == source_resid:
#                 continue
#             tgt = self.get_com(r)
#             if tgt is None:
#                 continue
#             d = math.dist(src, tgt)
#             if d <= radius_nm:
#                 candidates.append((r, d))

#         candidates.sort(key=lambda x: x[1])
#         if max_neighbors is not None and max_neighbors > 0:
#             candidates = candidates[:max_neighbors]

#         return [r for r, _ in candidates]

#     # ---------------------------------------------------------------------
#     # Coupling calculation
#     # ---------------------------------------------------------------------

#     def avg_cpl(self, ham_file, source_resid, topFilePath, gmxPath, mdpFilePath):
#         """
#         NORMAL-mode coupling evaluation:
#           1) cutAroundRes(...) to build a local subset
#           2) ASANN FirstNN to choose first-nearest neighbors
#           3) compute couplings for [source + neighbors]
#         """
#         start = time.time()
#         nn_subset = self.gro_obj.cutAroundRes(source_resid, [1.0, 1.0, 1.0], allCOMs=self.all_coms)
#         print(f"Time taken to cut around residue {source_resid}: {time.time() - start:.2f} s")

#         if not nn_subset:
#             print(f"No neighbors found for residue {source_resid}. Ending walk.")
#             return None

#         nn_finder = asann.FirstNN(self.all_coms, self.all_resids, subset_resids=nn_subset)
#         neighbors = nn_finder.nearest_neighbors_asann(source_resid)

#         if not neighbors:
#             print(f"No neighbors found for residue {source_resid}. Ending walk.")
#             return None

#         return self.avg_cpl_for_neighbors(ham_file, source_resid, neighbors, topFilePath, gmxPath, mdpFilePath)

#     def avg_cpl_for_neighbors(self, ham_file, source_resid, neighbors, topFilePath, gmxPath, mdpFilePath):
#         """
#         Compute couplings for an explicit neighbor list (used in RESCUE mode too).

#         This function:
#           - writes a temporary gro with [source + neighbors]
#           - renumbers it
#           - builds topology and inputs
#           - runs grompp + mdrun
#           - reads TB_HAMILTONIAN.xvg and returns coupling per neighbor (meV, abs)
#         """
#         neighbors = list(neighbors) if neighbors else []
#         if not neighbors:
#             return None

#         resids_to_write = [source_resid] + neighbors
#         self.gro_obj.write_gro("temporary.gro", resids_to_write)

#         # Renumber gro and map original resids -> new resids
#         gmx_runner = gmx.gmx(exe="gmx", gro="temporary.gro")
#         gmx_runner.renum()
#         renumbered_resids = self.gro_obj.renumbered_resids("temporary.gro")
#         resid_mapping = dict(zip(resids_to_write, renumbered_resids))

#         # Topology setup
#         new_top_path = "pen-esp.top"
#         shutil.copy(topFilePath, new_top_path)

#         itp_path = "/data/sgarg/pentacene/sampling_qm_zone_algorithm/gb-pen/input_files"
#         for itp_file in glob.glob(os.path.join(itp_path, "*.itp")):
#             shutil.copy(itp_file, ".")

#         pen_top = top(new_top_path)
#         pen_top.update_molecule_count("molecule", len(resids_to_write))

#         # Build tpr
#         gmx_runner = gmx.gmx(
#             exe=gmxPath,
#             gro="temporary.gro",
#             top=new_top_path,
#             mdp=mdpFilePath,
#             tpr="ham",
#         )
#         try:
#             gmx_runner.grompp()
#         except Exception as e:
#             print(f"Error running grompp: {e}")

#         # spec file (only if missing)
#         if not os.path.exists("pen.spec"):
#             spec_manager = inpman.inpManager("spec")
#             spec_manager.update(
#                 natoms=36,
#                 nelectrons=102,
#                 nallorbitals=102,
#                 nfragorbs=1,
#                 fragorbs=51,
#             )
#             spec_manager.save("pen.spec")
#         else:
#             print("spec file already exists.")

#         # charge-transfer.dat (sites are renumbered)
#         renum_sites = [resid_mapping[r] for r in resids_to_write]
#         ct_manager = inpman.inpManager("ct")
#         ct_manager.update(
#             sites=renum_sites,
#             seed=random.randint(1, 100),
#             chargecarrier="hole",
#             atomindex=[1, 20, 29],
#             typefiles="pen.spec",
#             jobtype="NOM",
#             internalrelax="onsite",
#         )
#         ct_manager.save("charge-transfer.dat")

#         # Run
#         gmx_runner = gmx.gmx(exe=gmxPath, tpr="ham")
#         gmx_runner.mdrun(nt=1)

#         if not os.path.exists(ham_file):
#             print(f"File {ham_file} not found.")
#             return None

#         # Parse TB_HAMILTONIAN.xvg
#         extracted = {}
#         with open(ham_file, "r") as file:
#             for line in file:
#                 if not line.strip() or line.startswith(("#", "@")):
#                     continue
#                 parts = line.split()
#                 try:
#                     values = [float(parts[i]) for i in range(2, 2 + len(neighbors))]
#                 except (IndexError, ValueError):
#                     print(f"Error processing line: {line.strip()}")
#                     continue

#                 for neighbor_resid, cpl_value in zip(neighbors, values):
#                     extracted[neighbor_resid] = abs(cpl_value) * 1000.0  # meV

#         for neigh, val in extracted.items():
#             print(f"  Neighbor {neigh}: {val:.1f} meV")

#         self.cleanup_specific_files()
#         return extracted

#     def probabilities(self, coupling_data, beta=(1 / 25.7)):
#         """
#         Convert coupling magnitudes -> normalized probabilities using a Boltzmann-like form.
#         p_i ∝ exp(beta * (J_i - J_max))
#         """
#         if not coupling_data:
#             return {}

#         max_cpl = max(coupling_data.values())
#         exp_weights = {rid: math.exp(beta * (cpl - max_cpl)) for rid, cpl in coupling_data.items()}
#         total = sum(exp_weights.values())

#         if total <= 0:
#             return {r: 0.0 for r in coupling_data}
#         return {r: w / total for r, w in exp_weights.items()}

#     # ---------------------------------------------------------------------
#     # Geometry: distance + direction (cosine)
#     # ---------------------------------------------------------------------

#     def check_too_far(self, source_com, target_com):
#         """NORMAL distance constraint: <= self.cutoff."""
#         d = math.dist(source_com, target_com)
#         print(f"Distance SR–NR: {d:.2f} nm")
#         return d > self.cutoff

#     def check_too_far_rescue(self, source_com, target_com):
#         """RESCUE distance constraint: <= self.rescue_radius_nm."""
#         d = math.dist(source_com, target_com)
#         print(f"[RESCUE] Distance SR–NR: {d:.2f} nm (limit={self.rescue_radius_nm:.2f} nm)")
#         return d > self.rescue_radius_nm

#     def _step_cos(self, source_resid, target_resid, forward_sign=1.0):
#         """
#         Signed cosine between (source->target) step vector and the GB direction.
#         forward_sign selects +gb_vector (g1->g2) or -gb_vector (g2->g1).
#         """
#         src = self.get_com(source_resid)
#         tgt = self.get_com(target_resid)
#         if src is None or tgt is None:
#             return None

#         step = (tgt[0] - src[0], tgt[1] - src[1], tgt[2] - src[2])
#         step_norm = math.sqrt(step[0] ** 2 + step[1] ** 2 + step[2] ** 2)
#         if step_norm == 0:
#             return None

#         dot = (
#             step[0] * self.gb_vector[0] +
#             step[1] * self.gb_vector[1] +
#             step[2] * self.gb_vector[2]
#         )
#         cos_raw = dot / step_norm
#         cos_theta = cos_raw * forward_sign

#         # Debug
#         print(f"    Step vector: ({step[0]:.3f}, {step[1]:.3f}, {step[2]:.3f})")
#         print(f"    GB vector: ({self.gb_vector[0]:.3f}, {self.gb_vector[1]:.3f}, {self.gb_vector[2]:.3f})")
#         print(f"    Raw cos(theta) = {cos_raw:.3f}, forward_sign = {forward_sign:.1f}, final cos(theta) = {cos_theta:.3f}")

#         return cos_theta

#     # ---------------------------------------------------------------------
#     # Selection helpers
#     # ---------------------------------------------------------------------

#     def _forward_sign_for_path(self, first_source_resid):
#         """
#         Defines what 'forward' means:
#           - if start in g1 -> forward is +gb_vector
#           - if start in g2 -> forward is -gb_vector
#         """
#         g = self.grain_of_resid(first_source_resid)
#         if g == "g1":
#             print(f"First source {first_source_resid} in g1: forward = +gb_vector")
#             return 1.0
#         if g == "g2":
#             print(f"First source {first_source_resid} in g2: forward = -gb_vector")
#             return -1.0
#         print(f"First source {first_source_resid} in neither grain: forward = +gb_vector (default)")
#         return 1.0

#     def _sample_by_closest_prob(self, probs_dict, random_value):
#         """
#         Your existing sampling rule:
#         choose the residue whose probability is closest to random_value.
#         (Not inverse-CDF sampling; kept unchanged.)
#         """
#         sorted_neighbors = sorted(probs_dict.items(), key=lambda x: abs(x[1] - random_value))
#         chosen_resid, chosen_prob = sorted_neighbors[0]
#         return chosen_resid, chosen_prob

#     def _build_eff_weights_normal(self, coupling_probs, source_resid, visited_resids, forward_sign):
#         """NORMAL: eff = p_cpl * cos(theta)^direction_alpha with forward + distance filters."""
#         src_com = self.get_com(source_resid)
#         if src_com is None:
#             return {}

#         eff = {}
#         for target_resid, p_cpl in coupling_probs.items():
#             if target_resid in visited_resids:
#                 continue

#             cos_theta = self._step_cos(source_resid, target_resid, forward_sign)
#             if cos_theta is None:
#                 continue

#             print(f"Resid {target_resid}: cos(theta) = {cos_theta:.3f}")

#             if cos_theta < self.min_cos_forward:
#                 print(f"  -> below min_cos_forward={self.min_cos_forward}, reject.")
#                 continue

#             tgt_com = self.get_com(target_resid)
#             if tgt_com is None:
#                 continue

#             if self.check_too_far(src_com, tgt_com):
#                 print("  -> too far, reject.")
#                 continue

#             dir_factor = cos_theta ** self.direction_alpha
#             w = p_cpl * dir_factor
#             print(f"  [NORMAL] p_cpl={p_cpl:.3f}, dir_factor={dir_factor:.3f}, p_eff={w:.3e}")

#             if w > 0.0:
#                 eff[target_resid] = w

#         return eff

#     def _build_eff_weights_rescue(self, rescue_probs, source_resid, visited_resids, forward_sign):
#         """RESCUE: eff = (p_cpl^λ) * max(cos,0)^power with forward + rescue-distance filters."""
#         src_com = self.get_com(source_resid)
#         if src_com is None:
#             return {}

#         eff = {}
#         for target_resid, p_cpl in rescue_probs.items():
#             if target_resid in visited_resids:
#                 continue

#             cos_theta = self._step_cos(source_resid, target_resid, forward_sign)
#             if cos_theta is None:
#                 continue

#             print(f"Resid {target_resid}: cos(theta) = {cos_theta:.3f}")

#             if cos_theta < self.rescue_min_cos_forward:
#                 print(f"  -> below rescue_min_cos_forward={self.rescue_min_cos_forward}, reject.")
#                 continue

#             tgt_com = self.get_com(target_resid)
#             if tgt_com is None:
#                 continue

#             if self.check_too_far_rescue(src_com, tgt_com):
#                 print("  -> too far for rescue-radius, reject.")
#                 continue

#             w_dir = max(cos_theta, 0.0) ** self.rescue_dir_power
#             w_cpl = max(p_cpl, 1e-15) ** self.rescue_cpl_lambda
#             w = w_cpl * w_dir

#             print(
#                 f"  [RESCUE] p_cpl={p_cpl:.3f}, (p_cpl^λ)={w_cpl:.3e}, "
#                 f"(cos^p)={w_dir:.3e}, p_eff={w:.3e}"
#             )

#             if w > 0.0:
#                 eff[target_resid] = w

#         return eff

#     # ---------------------------------------------------------------------
#     # Public API: choose next neighbor (with backtracking)
#     # ---------------------------------------------------------------------

#     def select_next_neighbor(self, coupling_probs, source_resid, first_source_resid, visited_resids, sampled_paths):
#         """
#         Select next residue using:
#           1) NORMAL attempt
#           2) RESCUE attempt (radius neighbors)
#           3) Backtracking via fallback_select_neighbor()

#         Always sets self.last_decision before returning.
#         """
#         forward_sign = self._forward_sign_for_path(first_source_resid)

#         while True:
#             rv = random.uniform(0, 1)
#             print(f"Random number for source_resid {source_resid}: {rv}")

#             # ------------------ Stage A: NORMAL ------------------
#             eff_w = self._build_eff_weights_normal(
#                 coupling_probs=coupling_probs,
#                 source_resid=source_resid,
#                 visited_resids=visited_resids,
#                 forward_sign=forward_sign,
#             )

#             if eff_w:
#                 total = sum(eff_w.values())
#                 eff_probs = {r: w / total for r, w in eff_w.items()}
#                 chosen, chosen_p = self._sample_by_closest_prob(eff_probs, rv)
#                 print(f"[NORMAL] Selected neighbor {chosen} with eff_prob ≈ {chosen_p:.3f}")
#                 self._set_decision({"mode": "NORMAL", "source": source_resid, "chosen": chosen})
#                 return chosen

#             # ------------------ Stage B: RESCUE ------------------
#             print(
#                 f"[RESCUE] No valid NORMAL neighbors at {source_resid}. "
#                 f"Expanding within radius {self.rescue_radius_nm} nm (keep {self.rescue_max_neighbors})..."
#             )

#             radius_neighbors = self.neighbors_within_radius(
#                 source_resid,
#                 radius_nm=self.rescue_radius_nm,
#                 max_neighbors=self.rescue_max_neighbors,
#             )
#             radius_neighbors = [r for r in radius_neighbors if r not in visited_resids]

#             if radius_neighbors:
#                 print(f"[RESCUE] Candidate neighbors (within radius): {radius_neighbors}")

#                 avg_cpl_values = self.avg_cpl_for_neighbors(
#                     self.ham_file,
#                     source_resid,
#                     radius_neighbors,
#                     self.topFilePath,
#                     self.gmxPath,
#                     self.mdpFilePath,
#                 )

#                 if avg_cpl_values:
#                     rescue_probs = self.probabilities(avg_cpl_values)
#                     rescue_w = self._build_eff_weights_rescue(
#                         rescue_probs=rescue_probs,
#                         source_resid=source_resid,
#                         visited_resids=visited_resids,
#                         forward_sign=forward_sign,
#                     )

#                     if rescue_w:
#                         total = sum(rescue_w.values())
#                         rescue_eff_probs = {r: w / total for r, w in rescue_w.items()}
#                         chosen, chosen_p = self._sample_by_closest_prob(rescue_eff_probs, rv)
#                         print(f"[RESCUE] Selected neighbor {chosen} with eff_prob ≈ {chosen_p:.3f}")
#                         self._set_decision({"mode": "RESCUE", "source": source_resid, "chosen": chosen})
#                         return chosen

#                     print("[RESCUE] Still no valid forward candidates after rescue scoring.")
#                 else:
#                     print("[RESCUE] Could not compute couplings for rescue neighbor list.")
#             else:
#                 print("[RESCUE] No radius neighbors available (excluding visited).")

#             # ------------------ Backtracking ------------------
#             print("No valid neighbors (after NORMAL + RESCUE); backtracking...")

#             while sampled_paths:
#                 sampled_paths.pop()

#                 if not sampled_paths:
#                     print("No previous residues to backtrack to. Stopping.")
#                     self._set_decision({
#                         "mode": "STOP",
#                         "source": source_resid,
#                         "chosen": None,
#                         "reason": "Backtracking exhausted: no previous residues"
#                     })
#                     return None

#                 fallback_resid = sampled_paths[-1]
#                 selected = self.fallback_select_neighbor(fallback_resid, visited_resids, forward_sign)
#                 print(f"Selected neighbor for fallback resid {fallback_resid}: {selected}")

#                 if selected is not None:
#                     print(f"Backtracking successful. New selected neighbor: {selected}")
#                     # Note: fallback_select_neighbor sets last_decision itself
#                     return selected

#             print("No valid neighbors found after backtracking. Ending.")
#             self._set_decision({
#                 "mode": "STOP",
#                 "source": source_resid,
#                 "chosen": None,
#                 "reason": "No valid neighbors after NORMAL+RESCUE+backtracking"
#             })
#             return None

#     def fallback_select_neighbor(self, fallback_resid, visited_resids, forward_sign):
#         """
#         Backtracking helper:
#           - recompute couplings around fallback_resid (NORMAL neighborhood)
#           - try NORMAL selection
#           - if NORMAL fails, try RESCUE expansion from fallback_resid

#         Always sets self.last_decision before returning.
#         """
#         avg_cpl_values = self.avg_cpl(
#             self.ham_file,
#             fallback_resid,
#             self.topFilePath,
#             self.gmxPath,
#             self.mdpFilePath,
#         )
#         if not avg_cpl_values:
#             print(f"No coupling values for fallback_resid {fallback_resid}.")
#             self._set_decision({
#                 "mode": "FALLBACK-FAIL",
#                 "source": fallback_resid,
#                 "chosen": None,
#                 "reason": "No coupling values in fallback (avg_cpl returned None)"
#             })
#             return None

#         coupling_probs = self.probabilities(avg_cpl_values)
#         rv = random.uniform(0, 1)
#         print(f"Random number for fallback_resid {fallback_resid}: {rv}")

#         # -------- Try NORMAL from fallback --------
#         eff_w = self._build_eff_weights_normal(
#             coupling_probs=coupling_probs,
#             source_resid=fallback_resid,
#             visited_resids=visited_resids,
#             forward_sign=forward_sign,
#         )
#         if eff_w:
#             total = sum(eff_w.values())
#             eff_probs = {r: w / total for r, w in eff_w.items()}
#             chosen, chosen_p = self._sample_by_closest_prob(eff_probs, rv)
#             print(f"[FALLBACK-NORMAL] Selected {chosen} with eff_prob ≈ {chosen_p:.3f}")
#             self._set_decision({"mode": "FALLBACK-NORMAL", "source": fallback_resid, "chosen": chosen})
#             return chosen

#         # -------- Try RESCUE from fallback --------
#         print(
#             f"[FALLBACK-RESCUE] No valid NORMAL neighbors at {fallback_resid}. "
#             f"Expanding within radius {self.rescue_radius_nm} nm..."
#         )

#         radius_neighbors = self.neighbors_within_radius(
#             fallback_resid,
#             radius_nm=self.rescue_radius_nm,
#             max_neighbors=self.rescue_max_neighbors,
#         )
#         radius_neighbors = [r for r in radius_neighbors if r not in visited_resids]

#         if not radius_neighbors:
#             print("[FALLBACK-RESCUE] No radius neighbors available.")
#             self._set_decision({
#                 "mode": "FALLBACK-FAIL",
#                 "source": fallback_resid,
#                 "chosen": None,
#                 "reason": "No rescue radius neighbors available in fallback"
#             })
#             return None

#         avg_cpl_rescue = self.avg_cpl_for_neighbors(
#             self.ham_file,
#             fallback_resid,
#             radius_neighbors,
#             self.topFilePath,
#             self.gmxPath,
#             self.mdpFilePath,
#         )
#         if not avg_cpl_rescue:
#             print("[FALLBACK-RESCUE] Could not compute couplings for rescue list.")
#             self._set_decision({
#                 "mode": "FALLBACK-FAIL",
#                 "source": fallback_resid,
#                 "chosen": None,
#                 "reason": "avg_cpl_for_neighbors returned None in fallback rescue"
#             })
#             return None

#         rescue_probs = self.probabilities(avg_cpl_rescue)
#         rescue_w = self._build_eff_weights_rescue(
#             rescue_probs=rescue_probs,
#             source_resid=fallback_resid,
#             visited_resids=visited_resids,
#             forward_sign=forward_sign,
#         )
#         if not rescue_w:
#             print("[FALLBACK-RESCUE] Still no valid candidates after rescue scoring.")
#             self._set_decision({
#                 "mode": "FALLBACK-FAIL",
#                 "source": fallback_resid,
#                 "chosen": None,
#                 "reason": "No valid candidates after fallback rescue scoring"
#             })
#             return None

#         total = sum(rescue_w.values())
#         rescue_eff_probs = {r: w / total for r, w in rescue_w.items()}
#         chosen, chosen_p = self._sample_by_closest_prob(rescue_eff_probs, rv)
#         print(f"[FALLBACK-RESCUE] Selected {chosen} with eff_prob ≈ {chosen_p:.3f}")
#         self._set_decision({"mode": "FALLBACK-RESCUE", "source": fallback_resid, "chosen": chosen})
#         return chosen
