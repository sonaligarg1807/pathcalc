#!/usr/bin/env python3
"""
ResidExtractor that selects residues by COM ranges and optionally writes:
 - a text file with selected resid + COM
 - a full-atom .gro using the user's gro object via gro_obj.write_gro(output_path, resids_to_write)
"""

import numpy as np
import random
from typing import List, Tuple, Union

class ResidExtractor:
    def __init__(self, gro_obj, natoms: int, resids: List[int], coms: List[Tuple[float,float,float]]):
        """
        gro_obj: instance of your gro class (must implement write_gro(output_path, resids_to_write))
        natoms: atoms per residue (kept for compatibility; not required)
        resids: list-like of residue ids (ints)
        coms: list-like of (x,y,z) floats corresponding to resids (same order)
        """
        self.gro_obj = gro_obj
        # try to record original gro filename if available
        self.gro_file = getattr(gro_obj, "path", None)
        self.natoms = natoms
        self.resids = list(resids)
        self.coms = list(coms)
        if len(self.resids) != len(self.coms):
            raise ValueError("resids and coms must have the same length")

    def extract_source_resids(self,
                              axes_count: int,
                              axes: str,
                              selection_choice: str,
                              y_range_choice: int,
                              x_limits: Union[Tuple[float,float], None],
                              y_limits_range_1: Union[Tuple[float,float], List[Tuple[float,float]]],
                              y_limits_range_2: Union[Tuple[float,float], List[Tuple[float,float]]],
                              z_limits: Union[Tuple[float,float], None],
                              num_to_select: int = None,
                              output_file: str = "random_resids.txt",
                              gro_output_file: str = None) -> List[Tuple[int,float,float,float]]:
        """
        Select residues based on axis limits and either return all matches (selection_choice='yes')
        or randomly sample num_to_select residues from the matches (selection_choice='no').

        Writes a text file with header "# resid   x_com   y_com   z_com" to output_file.

        If gro_output_file is provided, calls self.gro_obj.write_gro(gro_output_file, resids_list)
        where resids_list is a list of residue ids (ints) to write (preserves your gro-class formatting).
        """

        # normalize axes input
        axes_list = [a.strip().lower() for a in axes.split(",") if a.strip()]
        if len(axes_list) != axes_count:
            raise ValueError(f"axes_count ({axes_count}) != number of axes provided ({len(axes_list)})")

        # default axis limits
        axis_limits = {
            'x': x_limits if x_limits else (-np.inf, np.inf),
            'y': None,
            'z': z_limits if z_limits else (-np.inf, np.inf)
        }

        # choose y limits based on y_range_choice
        if 'y' in axes_list:
            if y_range_choice == 1:
                axis_limits['y'] = y_limits_range_1
            elif y_range_choice == 2:
                axis_limits['y'] = y_limits_range_2
            else:
                raise ValueError("y_range_choice must be 1 or 2")

        if selection_choice not in ("yes", "no"):
            raise ValueError("selection_choice must be 'yes' or 'no'")

        # select by COM
        matched_resids = []
        for resid, com in zip(self.resids, self.coms):
            x_com, y_com, z_com = com

            if 'x' in axes_list:
                x_min, x_max = axis_limits['x']
                if not (x_min <= x_com <= x_max):
                    continue

            if 'z' in axes_list:
                z_min, z_max = axis_limits['z']
                if not (z_min <= z_com <= z_max):
                    continue

            if 'y' in axes_list:
                y_limits = axis_limits['y']
                # support tuple single range or list of ranges
                if isinstance(y_limits, tuple):
                    if not (y_limits[0] <= y_com <= y_limits[1]):
                        continue
                elif isinstance(y_limits, list):
                    if not any((start <= y_com <= end) for (start, end) in y_limits):
                        continue
                else:
                    # no y limits provided (shouldn't happen because of earlier checks)
                    pass

            matched_resids.append(resid)

        # final selection
        if selection_choice == "yes":
            chosen = matched_resids
        else:  # "no" => random sampling
            if num_to_select is None:
                raise ValueError("num_to_select must be provided when selection_choice='no'")
            k = min(num_to_select, len(matched_resids))
            chosen = random.sample(matched_resids, k)

        # build final_resids with COMs
        final_resids = []
        for resid in chosen:
            # find index in self.resids to obtain COM
            try:
                idx = self.resids.index(resid)
            except ValueError:
                # skip if resid not found (shouldn't happen)
                continue
            x_com, y_com, z_com = self.coms[idx]
            final_resids.append((resid, float(x_com), float(y_com), float(z_com)))

        # write text output
        with open(output_file, "w") as fh:
            fh.write("# resid   x_com   y_com   z_com\n")
            for resid, x_com, y_com, z_com in final_resids:
                fh.write(f"{resid:<6d} {x_com:>10.5f} {y_com:>10.5f} {z_com:>10.5f}\n")

        print(f"Extracted {len(final_resids)} residues with COMs and saved to '{output_file}'")

        # optionally write full-atom GRO using the user's gro class writer
        if gro_output_file:
            # the gro writer expects a list of resids; preserve integer resid list order
            resids_to_write = [int(r[0]) for r in final_resids]
            # call user's writer
            try:
                # your write_gro signature: write_gro(self, output_path: str, resids_to_write: list)
                self.gro_obj.write_gro(gro_output_file, resids_to_write)
                print(f"GRO file written with {len(resids_to_write)} residues -> '{gro_output_file}'")
            except Exception as e:
                print(f"Error while calling gro_obj.write_gro(): {e}")

        return final_resids
