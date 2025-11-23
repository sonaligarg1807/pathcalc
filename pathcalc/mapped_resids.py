# mapped_resids.py
"""
Utilities to map grain-residue IDs from an old slab .gro to a new
reference .gro (e.g. NVT-equilibrated), using COM-based nearest-neighbour
matching.

Also writes out:
  - g1_resids_nvt.txt / g2_resids_nvt.txt (mapped residue lists)
  - g1_nvt.gro / g2_nvt.gro (subset structures from the reference .gro)
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple

import numpy as np

from pathcalc import gro  # adjust if your gro class is imported differently


# ----------------------------------------------------------------------
# Low-level helpers
# ----------------------------------------------------------------------

def _compute_resids_and_coms(gro_obj) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (resids, COMs) from a gro object.

    resids: (N,) int array
    COMs  : (N, 3) float array
    """
    # assuming gro_obj.allRes and gro_obj.MP_resCOMs exist
    all_resids = [int(r) for r in gro_obj.allRes]
    coms = np.asarray(gro_obj.MP_resCOMs, dtype=float)

    if coms.shape[0] != len(all_resids):
        raise ValueError(
            f"[mapped_resids] MP_resCOMs length {coms.shape[0]} "
            f"!= number of residues {len(all_resids)}"
        )

    return np.asarray(all_resids, dtype=int), coms


def map_resids_by_com(
    src_gro_path: str,
    ref_gro_path: str,
    cutoff: float = 0.3
) -> Dict[int, int]:
    """
    Map residue IDs from src_gro_path (old numbering) to residue IDs in
    ref_gro_path (e.g. NVT) using COM-based nearest-neighbour mapping.

    Returns:
        mapping: dict {old_resid -> new_resid}
    """
    src = gro(src_gro_path)
    ref = gro(ref_gro_path)

    src_resids, src_coms = _compute_resids_and_coms(src)
    ref_resids, ref_coms = _compute_resids_and_coms(ref)

    mapping: Dict[int, int] = {}

    for i, r in enumerate(src_resids):
        dists = np.linalg.norm(ref_coms - src_coms[i], axis=1)
        j = int(np.argmin(dists))
        dmin = float(dists[j])

        if dmin > cutoff:
            print(
                f"[mapped_resids] WARN: src resid {r} → ref resid {ref_resids[j]} "
                f"at distance {dmin:.3f} nm (> {cutoff} nm)"
            )

        mapping[int(r)] = int(ref_resids[j])

    return mapping


def write_subset_gro(
    ref_gro_path: str,
    selected_resids: Iterable[int],
    out_gro_path: str,
) -> None:
    """
    Write a subset .gro file from ref_gro_path containing only atoms whose
    residue ID (first 5 columns) is in selected_resids.

    Keeps the original residue numbers from the reference .gro.
    """
    selected = set(int(r) for r in selected_resids)

    with open(ref_gro_path, "r") as f:
        lines = f.readlines()

    if len(lines) < 3:
        raise ValueError(f"[mapped_resids] {ref_gro_path} does not look like a valid .gro file")

    title = lines[0].rstrip("\n")
    natoms_total = int(lines[1].strip())

    atom_lines = lines[2:2 + natoms_total]
    box_line = lines[2 + natoms_total]

    kept: List[str] = []
    for line in atom_lines:
        if not line.strip():
            continue
        # standard .gro: resid in columns 1–5
        resid_str = line[0:5]
        try:
            resid = int(resid_str)
        except ValueError:
            # if something odd, just skip
            continue
        if resid in selected:
            kept.append(line)

    os.makedirs(os.path.dirname(out_gro_path) or ".", exist_ok=True)
    with open(out_gro_path, "w") as out:
        out.write(f"{title} (subset)\n")
        out.write(f"{len(kept):5d}\n")
        for l in kept:
            out.write(l)
        out.write(box_line)


def y_range_from_resids(
    resids: Iterable[int],
    coms_dict: Dict[int, np.ndarray],
) -> Tuple[float, float]:
    """
    Compute (min_y, max_y) from a dict resid -> COM.
    """
    ys = [coms_dict[int(r)][1] for r in resids]
    return float(min(ys)), float(max(ys))


# ----------------------------------------------------------------------
# High-level convenience function
# ----------------------------------------------------------------------

def map_and_write_grains_to_nvt(
    g1_gro_path: str,
    g2_gro_path: str,
    ref_gro_path: str,
    out_dir: str,
    cutoff: float = 0.3,
) -> Tuple[List[int], List[int]]:
    """
    High-level function:

      1. Map residues from g1_gro_path and g2_gro_path (old slab numbering)
         onto ref_gro_path (e.g. NVT-equilibrated .gro) using COMs.
      2. Write:
           - g1_resids_nvt.txt, g2_resids_nvt.txt
           - g1_nvt.gro, g2_nvt.gro
         into out_dir.
      3. Return:
           (g1_resids_nvt, g2_resids_nvt) lists.
    """
    os.makedirs(out_dir, exist_ok=True)

    # old -> new mappings
    map_g1 = map_resids_by_com(g1_gro_path, ref_gro_path, cutoff=cutoff)
    map_g2 = map_resids_by_com(g2_gro_path, ref_gro_path, cutoff=cutoff)

    g1_resids_nvt = sorted(set(map_g1.values()))
    g2_resids_nvt = sorted(set(map_g2.values()))

    # write resid lists
    g1_txt = os.path.join(out_dir, "g1_resids_nvt.txt")
    g2_txt = os.path.join(out_dir, "g2_resids_nvt.txt")

    with open(g1_txt, "w") as f:
        for r in g1_resids_nvt:
            f.write(f"{r}\n")

    with open(g2_txt, "w") as f:
        for r in g2_resids_nvt:
            f.write(f"{r}\n")

    # write subset .gro files from reference (NVT) structure
    g1_gro_nvt = os.path.join(out_dir, "g1_nvt.gro")
    g2_gro_nvt = os.path.join(out_dir, "g2_nvt.gro")

    write_subset_gro(ref_gro_path, g1_resids_nvt, g1_gro_nvt)
    write_subset_gro(ref_gro_path, g2_resids_nvt, g2_gro_nvt)

    print(f"[mapped_resids] Wrote mapped residue lists to:\n  {g1_txt}\n  {g2_txt}")
    print(f"[mapped_resids] Wrote mapped .gro files to:\n  {g1_gro_nvt}\n  {g2_gro_nvt}")

    return g1_resids_nvt, g2_resids_nvt
