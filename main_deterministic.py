#!/usr/bin/env python3
import os
import sys
from ideal_pathcalc import gro, path_deterministic  # PathFinder is in this module

# ------------------- USER CONFIG (adjust paths as needed) -------------------
topFilePath = "/home/sgarg/pathcalc/inps/pen-esp.top"
groFilePath = "/home/sgarg/pathcalc/inps/ham.gro"
mdpFilePath = "/home/sgarg/pathcalc/inps/namd-qmmm.mdp"
gmxPath     = "/home/fghalami/GMX/Gromacs-SH/build_test/src/kernel"
ham_file    = "TB_HAMILTONIAN.xvg"
# ---------------------------------------------------------------------------

def prompt_int(msg):
    while True:
        raw = input(msg).strip()
        try:
            return int(raw)
        except ValueError:
            print("Please enter an integer.")

def prompt_float(msg):
    while True:
        raw = input(msg).strip()
        try:
            return float(raw)
        except ValueError:
            print("Please enter a number (float).")

def main():
    # Inputs
    start_resid      = prompt_int("Enter starting resid (int): ")
    Lx_margin        = prompt_float("Enter Lx margin (nm): ")
    Ly_margin        = prompt_float("Enter Ly margin (nm): ")
    Lz_margin        = prompt_float("Enter Lz margin (nm): ")
    path_length_N    = prompt_int("Enter total number of QM sites in path (>=1): ")
    angle_threshold  = prompt_float("Enter max forward angle threshold (degrees): ")
    if path_length_N < 1:
        raise RuntimeError("Path length must be at least 1.")

    # Load structure + COMs
    pen_gro = gro(groFilePath)
    all_coms   = pen_gro.MP_resCOMs
    all_resids = list(pen_gro.allRes)
    if start_resid not in all_resids:
        raise RuntimeError("No next residue chosen: start resid not found.")

    # Init PathFinder
    pf = path_deterministic.PathFinder(
        gro_obj=pen_gro,
        all_coms=all_coms,
        all_resids=all_resids,
        ham_file=ham_file,
        topFilePath=topFilePath,
        gmxPath=gmxPath,
        mdpFilePath=mdpFilePath,
    )

    # -------- Start-site boundary check (print COM + ranges + violations) --------
    if not pf.resid_has_L_margin(start_resid, Lx_margin, Ly_margin, Lz_margin):
        com = pf.get_com(start_resid)
        if com is None:
            x, y, z = (float('nan'), float('nan'), float('nan'))
        else:
            x, y, z = map(float, com)
        Lx, Ly, Lz = pf.get_box_nm()
        lo_x, hi_x = Lx_margin, Lx - Lx_margin
        lo_y, hi_y = Ly_margin, Ly - Ly_margin
        lo_z, hi_z = Lz_margin, Lz - Lz_margin

        violations = []
        if not (lo_x <= x <= hi_x): violations.append(f"x∉[{lo_x:.4f},{hi_x:.4f}]")
        if not (lo_y <= y <= hi_y): violations.append(f"y∉[{lo_y:.4f},{hi_y:.4f}]")
        if not (lo_z <= z <= hi_z): violations.append(f"z∉[{lo_z:.4f},{hi_z:.4f}]")
        viol_str = "; ".join(violations) if violations else "unknown (NaN COM?)"

        print(f"[boundary] Start resid {start_resid} COM (nm): x={x:.4f}, y={y:.4f}, z={z:.4f}", flush=True)
        print(f"[boundary] Box (nm): Lx={Lx:.4f}, Ly={Ly:.4f}, Lz={Lz:.4f} ; "
            f"margins: ({Lx_margin:.4f}, {Ly_margin:.4f}, {Lz_margin:.4f})", flush=True)
        print(f"[boundary] Allowed ranges: x∈[{lo_x:.4f},{hi_x:.4f}], "
            f"y∈[{lo_y:.4f},{hi_y:.4f}], z∈[{lo_z:.4f},{hi_z:.4f}]", flush=True)
        print(f"[boundary] Violations: {viol_str}", flush=True)
        raise RuntimeError("Site close to boundary, chose another site")

    # -------- Working directory --------
    root_dir = os.getcwd()
    workdir = f"SR_{start_resid}"
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)

    # -------- Greedy path build --------
    path_seq = [start_resid]      # accepted path so far
    visited = {start_resid}
    prev_resid = None
    curr_resid = start_resid

    while len(path_seq) < path_length_N:
        try:
            nxt = pf.select_next_greedy(
                current_resid=curr_resid,
                prev_resid=prev_resid,
                visited=visited,
                angle_threshold_deg=angle_threshold
            )
        except RuntimeError as e:
            # Save partial path, then propagate error with user-friendly context
            with open(f"QM_path_{start_resid}.txt", "w") as f:
                for r in path_seq:
                    f.write(f"{r}\n")
            os.chdir(root_dir)
            print(f"[selector] {str(e)}", flush=True)
            raise

        # -------- Margin-check BEFORE accepting/appending --------
        if not pf.resid_has_L_margin(nxt, Lx_margin, Ly_margin, Lz_margin):
            # COM + box diagnostics for the failing candidate
            com = pf.get_com(nxt)
            if com is None:
                x, y, z = (float('nan'), float('nan'), float('nan'))
            else:
                x, y, z = map(float, com)
            Lx, Ly, Lz = pf.get_box_nm()
            lo_x, hi_x = Lx_margin, Lx - Lx_margin
            lo_y, hi_y = Ly_margin, Ly - Ly_margin
            lo_z, hi_z = Lz_margin, Lz - Lz_margin

            violations = []
            if not (lo_x <= x <= hi_x): violations.append(f"x∉[{lo_x:.4f},{hi_x:.4f}]")
            if not (lo_y <= y <= hi_y): violations.append(f"y∉[{lo_y:.4f},{hi_y:.4f}]")
            if not (lo_z <= z <= hi_z): violations.append(f"z∉[{lo_z:.4f},{hi_z:.4f}]")
            viol_str = "; ".join(violations) if violations else "unknown (NaN COM?)"

            print(f"[boundary] Candidate resid {nxt} COM (nm): x={x:.4f}, y={y:.4f}, z={z:.4f}", flush=True)
            print(f"[boundary] Box (nm): Lx={Lx:.4f}, Ly={Ly:.4f}, Lz={Lz:.4f} ; "
                f"margins: ({Lx_margin:.4f}, {Ly_margin:.4f}, {Lz_margin:.4f})", flush=True)
            print(f"[boundary] Allowed ranges: x∈[{lo_x:.4f},{hi_x:.4f}], "
                f"y∈[{lo_y:.4f},{hi_y:.4f}], z∈[{lo_z:.4f},{hi_z:.4f}]", flush=True)
            print(f"[boundary] Violations: {viol_str}", flush=True)

            # Save partial path (without failing candidate)
            xcount = len(path_seq)
            n = path_length_N
            with open(f"QM_path_{start_resid}.txt", "w") as f:
                for r in path_seq:
                    f.write(f"{r}\n")
            os.chdir(root_dir)

            # Include COM + violations in the exception so it shows in stderr
            raise RuntimeError(
                f"site close to boundary, increase size of box. {xcount}/{n} sites are chosen "
                f"(resid {nxt} COM=({x:.4f},{y:.4f},{z:.4f}) nm; violations: {viol_str})"
            )

        # Accept and advance
        path_seq.append(nxt)
        visited.add(nxt)
        prev_resid, curr_resid = curr_resid, nxt

    # -------- Save final path --------
    outname = f"QM_path_{start_resid}.txt"
    with open(outname, "w") as f:
        for r in path_seq:
            f.write(f"{r}\n")

    print(f"Saved: {os.path.abspath(outname)}")
    os.chdir(root_dir)

if __name__ == "__main__":
    main()
