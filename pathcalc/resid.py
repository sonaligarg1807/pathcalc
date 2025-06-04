import numpy as np
import random
import sys

class ResidExtractor:
    def __init__(self, gro_file, natoms, resids, coms):
        self.gro_file = gro_file
        self.natoms = natoms
        self.resids = resids
        self.coms = coms

    def extract_source_resids(self, axes_count, axes, selection_choice, y_range_choice, x_limits, y_limits_range_1, y_limits_range_2, z_limits, num_to_select=None, output_file="random_resids.txt"):
        
        # Pre-check axes
        axes = [axis.strip().lower() for axis in axes.split(",")]
        if len(axes) != axes_count:
            print(f"Error: You specified {axes_count} axes, but provided {len(axes)} axes.")
            sys.exit(1)
        
        # Default axis limits
        axis_limits = {
            'x': x_limits if x_limits else (-np.inf, np.inf),
            'y': None,  # y is handled below based on y_range_choice
            'z': z_limits if z_limits else (-np.inf, np.inf)
        }

        # Set y limits
        if 'y' in axes:
            if y_range_choice == 1:
                axis_limits['y'] = y_limits_range_1
            elif y_range_choice == 2:
                axis_limits['y'] = y_limits_range_2
            else:
                print("Invalid choice for y_range_choice. Please enter 1 or 2.")
                sys.exit(1)

        if selection_choice not in ["yes", "no"]:
            print("Invalid choice for selection. Please enter 'yes' or 'no'.")
            sys.exit(1)

        selected_resids = []

        for resid, com in zip(self.resids, self.coms):
            x_com, y_com, z_com = com

            if 'x' in axes:
                x_min, x_max = axis_limits['x']
                if not (x_min <= x_com <= x_max):
                    continue

            if 'z' in axes:
                z_min, z_max = axis_limits['z']
                if not (z_min <= z_com <= z_max):
                    continue

            if 'y' in axes:
                y_limits = axis_limits['y']
                if isinstance(y_limits, tuple):
                    if not (y_limits[0] <= y_com <= y_limits[1]):
                        continue
                elif isinstance(y_limits, list):
                    if not any(start <= y_com <= end for start, end in y_limits):
                        continue

            selected_resids.append(resid)

        # Final selection
        if selection_choice == "yes":
            # final_resids = selected_resids
            final_resids = [(resid, *self.coms[self.resids.index(resid)]) for resid in selected_resids]
        else:
            if num_to_select is None:
                raise ValueError("num_to_select must be specified when selection_choice is 'no'")
            sampled_resids = random.sample(selected_resids, min(num_to_select, len(selected_resids)))
            final_resids = [(resid, *self.coms[self.resids.index(resid)]) for resid in sampled_resids]

        # Write to file with COM coordinates
        with open(output_file, "w") as f:
            f.write("# resid   x_com   y_com   z_com\n")
            for resid, x_com, y_com, z_com in final_resids:
                f.write(f"{resid:<6} {x_com:>10.5f} {y_com:>10.5f} {z_com:>10.5f}\n")

        print(f"Extracted {len(final_resids)} residues with COMs and saved to {output_file}")
        return final_resids
            