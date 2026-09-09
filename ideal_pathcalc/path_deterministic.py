import math
import os
import shutil
import glob
from pathcalc import top, gmx, inpman, asann


class PathFinder:
    """
    Greedy, deterministic next-site selector with following conditions:
      - Exclude visited residues (no revisits).
      - Consider coupling levels in strictly descending order.
      - If prev_resid is None (first step): pick the first in the top coupling level.
      - Else: among the current coupling level, keep ONLY candidates that are:
              (a) forward relative to (curr - prev), AND
              (b) within an angle threshold w.r.t. the forward direction.
            If none at that coupling level, drop to the next lower coupling level.
      - If no level yields a candidate, raise RuntimeError.

    Margin safety is checked by the caller (main script).
    """

    def __init__(self, gro_obj, all_coms, all_resids, ham_file,
                 topFilePath, gmxPath, mdpFilePath):
        self.gro_obj = gro_obj
        self.all_coms = all_coms
        self.all_resids = list(all_resids)
        self.coms = {resid: com for resid, com in zip(self.all_resids, self.all_coms)}
        self.ham_file = ham_file
        self.topFilePath = topFilePath
        self.mdpFilePath = mdpFilePath
        self.gmxPath = gmxPath

    # ------------------------- Utilities -------------------------

    def get_com(self, resid):
        """Return COM (x,y,z) in nm for a residue, or None."""
        return self.coms.get(resid)

    def get_box_nm(self):
        """
        Return (Lx, Ly, Lz) in nm.
        Works with gro.box_nm, gro.boxSize, or gro.box.
        If triclinic (9 values), use the first three cell vector lengths (vxx, vyy, vzz).
        """
        if hasattr(self.gro_obj, "box_nm"):
            box = list(self.gro_obj.box_nm)
        elif hasattr(self.gro_obj, "boxSize"):
            box = list(self.gro_obj.boxSize)
        elif hasattr(self.gro_obj, "box"):
            box = list(self.gro_obj.box)
        else:
            raise AttributeError("gro_obj has no 'box_nm', 'boxSize', or 'box' attribute (nm expected).")

        if len(box) < 3:
            raise ValueError(f"Box line has fewer than 3 values: {box}")

        return float(box[0]), float(box[1]), float(box[2])

    def resid_has_L_margin(self, resid, Lx_margin, Ly_margin, Lz_margin):
        """
        True if residue COM is at least the given margin away from each face.
        Margins can be different in x, y, z.
        """
        com = self.get_com(resid)
        if com is None:
            return False
        Lx, Ly, Lz = self.get_box_nm()
        x, y, z = com
        return (Lx_margin <= x <= Lx - Lx_margin) and \
               (Ly_margin <= y <= Ly - Ly_margin) and \
               (Lz_margin <= z <= Lz - Lz_margin)


    def _cleanup_specific_files(self, keep_extensions=(".txt",)):
        for filename in os.listdir():
            if os.path.isdir(filename) or filename.endswith(keep_extensions):
                continue
            try:
                os.remove(filename)
            except Exception:
                pass

    # ------------------------- Direction / Angle helpers -------------------------

    def _vec(self, a, b):
        """Return vector a->b given two 3D COMs."""
        return (b[0] - a[0], b[1] - a[1], b[2] - a[2])

    def _norm(self, v):
        return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    def _angle_deg(self, prev_resid, curr_resid, cand_resid):
        """
        Angle (degrees) between forward direction (curr - prev) and
        candidate displacement (cand - curr). Returns None if undefined.
        """
        com_prev = self.get_com(prev_resid)
        com_curr = self.get_com(curr_resid)
        com_cand = self.get_com(cand_resid)
        if com_prev is None or com_curr is None or com_cand is None:
            return None
        fwd = self._vec(com_prev, com_curr)
        disp = self._vec(com_curr, com_cand)
        nf = self._norm(fwd)
        nd = self._norm(disp)
        if nf == 0.0 or nd == 0.0:
            return None
        dot = fwd[0] * disp[0] + fwd[1] * disp[1] + fwd[2] * disp[2]
        cosang = max(-1.0, min(1.0, dot / (nf * nd)))
        return math.degrees(math.acos(cosang))

    def _is_forward(self, prev_resid, curr_resid, cand_resid):
        """
        Forward if (cand - curr) · (curr - prev) > 0. If prev_resid is None, returns None.
        """
        if prev_resid is None:
            return None
        com_prev = self.get_com(prev_resid)
        com_curr = self.get_com(curr_resid)
        com_cand = self.get_com(cand_resid)
        if com_prev is None or com_curr is None or com_cand is None:
            return None
        fwd = (com_curr[0]-com_prev[0], com_curr[1]-com_prev[1], com_curr[2]-com_prev[2])
        disp = (com_cand[0]-com_curr[0], com_cand[1]-com_curr[1], com_cand[2]-com_curr[2])
        proj = fwd[0]*disp[0] + fwd[1]*disp[1] + fwd[2]*disp[2]
        return proj > 0.0

    # ------------------------- Coupling evaluation -------------------------

    def avg_cpl(self, ham_file, source_resid, topFilePath, gmxPath, mdpFilePath):
        """
        Compute |coupling| (meV) from source_resid to its ASANN nearest neighbors.
        Returns: dict {neighbor_resid: coupling_meV}
        """
        # Neighborhood pre-cut (±1 nm) → ASANN selects true first NNs.
        nn = self.gro_obj.cutAroundRes(source_resid, [1.0, 1.0, 1.0], allCOMs=self.all_coms)
        if not nn:
            return None

        nn_finder = asann.FirstNN(self.all_coms, self.all_resids, subset_resids=nn)
        neighbors = nn_finder.nearest_neighbors_asann(source_resid)
        if not neighbors:
            return None

        # Write subset GRO and renumber
        resids_to_write = [source_resid] + neighbors
        self.gro_obj.write_gro("temporary.gro", resids_to_write)

        gmx_runner = gmx.gmx(exe="gmx", gro="temporary.gro")
        gmx_runner.renum()
        renumbered_resids = self.gro_obj.renumbered_resids("temporary.gro")
        resid_mapping = dict(zip(resids_to_write, renumbered_resids))

        # Copy/adjust topology
        new_top_path = "pen-esp.top"
        shutil.copy(topFilePath, new_top_path)
        itp_path = "/home/sgarg/pathcalc/inps"  # adjust if needed
        for itp_file in glob.glob(os.path.join(itp_path, "*.itp")):
            shutil.copy(itp_file, ".")
        pen_top = top(new_top_path)
        pen_top.update_molecule_count("molecule", len(resids_to_write))

        # Prepare/run short job to produce TB_HAMILTONIAN.xvg
        gmx_runner = gmx.gmx(exe=gmxPath, gro="temporary.gro", top=new_top_path, mdp=mdpFilePath, tpr="ham")
        try:
            gmx_runner.grompp()
        except Exception:
            pass

        if not os.path.exists("pen.spec"):
            spec_manager = inpman.inpManager("spec")
            spec_manager.update(natoms=36, nelectrons=102, nallorbitals=102, nfragorbs=1, fragorbs=51)
            spec_manager.save("pen.spec")

        renum_sites = [resid_mapping[r] for r in resids_to_write]
        ct_manager = inpman.inpManager("ct")
        ct_manager.update(
            sites=renum_sites,
            seed=1,  # deterministic
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
            return None

        # Parse couplings (abs, meV)
        extracted = {}
        with open(ham_file, "r") as fh:
            for line in fh:
                if line.strip() and not line.startswith(("#", "@")):
                    parts = line.split()
                    try:
                        values = [float(parts[i]) for i in range(2, 2 + len(neighbors))]
                        for nbr, cpl in zip(neighbors, values):
                            extracted[nbr] = abs(cpl) * 1000.0
                    except (IndexError, ValueError):
                        continue

        self._cleanup_specific_files()
        return extracted

    # ------------------------- Modified selector with angle constraint -------------------------

    def select_next_greedy(self, current_resid, prev_resid=None, visited=None, angle_threshold_deg=0.0):
        """
        Selection with descending coupling levels.
        If prev_resid is not None, only accept candidates that are:
            - forward (proj > 0), AND
            - angle(prev->curr , curr->cand) <= angle_threshold_deg
        If none at a coupling level, drop to the next lower level.
        If none at any level, raise RuntimeError that includes the smallest angle seen.

        First step (prev_resid is None): take the first at the top coupling level.
        """
        if visited is None:
            visited = set()

        cpls = self.avg_cpl(self.ham_file, current_resid, self.topFilePath, self.gmxPath, self.mdpFilePath)
        if not cpls:
            raise RuntimeError("No next residue chosen: missing coupling data.")

        # Exclude visited
        candidates = [(nbr, val) for nbr, val in cpls.items() if nbr not in visited]
        if not candidates:
            raise RuntimeError("No next residue chosen: all neighbors already visited.")

        # Sort by coupling desc ONLY (stable within ties)
        candidates.sort(key=lambda x: -x[1])

        # Group by equal coupling (within eps), highest down
        eps = 1e-12
        levels = []
        for nbr, val in candidates:
            if not levels:
                levels.append([(nbr, val)])
            else:
                last_val = levels[-1][0][1]
                if abs(val - last_val) < eps:
                    levels[-1].append((nbr, val))
                else:
                    levels.append([(nbr, val)])

        # Track the best (smallest) angle found among forward candidates at any level
        # to emit a helpful message if we fail the angle threshold everywhere.
        best_angle_info = None  # (angle_deg, resid, coupling_meV)

        for lvl in levels:
            if prev_resid is None:
                # No direction defined yet
                return lvl[0][0]

            # Build forward candidates for this level and measure angles
            forward_with_angles = []
            for nbr, val in lvl:
                if not self._is_forward(prev_resid, current_resid, nbr):
                    continue
                ang = self._angle_deg(prev_resid, current_resid, nbr)
                if ang is None:
                    continue
                forward_with_angles.append((nbr, val, ang))
                if (best_angle_info is None) or (ang < best_angle_info[0]):
                    best_angle_info = (ang, nbr, val)

            # Filter by angle threshold within this coupling level
            qualified = [(nbr, val) for (nbr, val, ang) in forward_with_angles if ang <= angle_threshold_deg]

            if qualified:
                # Take the first within this coupling level that meets angle constraint
                return qualified[0][0]

            # Else: drop to next (lower) coupling level

        # Exhausted all levels → include smallest angle in error for user clarity
        if best_angle_info is not None:
            ang, rid, cpl = best_angle_info
            raise RuntimeError(
                f"No next residue chosen: no forward residue within {angle_threshold_deg}° at any coupling level. "
                f"Smallest forward angle encountered was {ang:.2f}° at resid {rid} (|cpl|={cpl:.3f} meV)."
            )
        else:
            raise RuntimeError(
                "No next residue chosen: no forward residue found relative to the previous step."
            )

