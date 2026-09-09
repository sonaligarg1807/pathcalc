"""Axis-biased PathFinder: normal/rescue/fallback neighbor selection."""
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
