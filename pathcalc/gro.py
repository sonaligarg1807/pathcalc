import multiprocessing
import numpy as np

from scipy.spatial import cKDTree
from collections import OrderedDict

class gro:
    def __init__(self, groFilePath: str):
        self.groFilePath = groFilePath

    @property  
    def loadFile(self):
        with open(self.groFilePath, 'r') as f:
            grof = [line.strip() for line in f if line.strip()]
        return grof

    @property
    def natoms(self) -> int:
        return int(self.loadFile[1].split()[0])

    @property
    def allRes(self) -> set:
        allres = []
        for l in self.loadFile[2:-1]:
            allres.append(int(l.split()[0][:-3]))
        return set(allres)

    @property
    def nRes(self) -> int:
        return len(self.allRes)

    @property
    def natomsPerMol(self) -> int:
        return int(self.natoms/self.nRes)

    @property
    def resCOMs(self) -> list:
        allCOM = []
        for res in self.allRes:
            allCOM.append(self.getCOM(res))
        return allCOM

    @property
    def MP_resCOMs(self) -> list:
        with multiprocessing.Pool() as pool:
            allCOM = pool.map(self.getCOM, self.allRes)
        return allCOM

    @property
    def boxSize(self) -> list:
        box = []
        for i in self.loadFile[-1].split():
            box.append(float(i))
        return box

    def getRes(self, resid: int):
        if resid not in self.allRes:
            raise noSuchResidError(f'There is no {resid}')
        data = []
        for line in self.loadFile[2:-1]:  
            line_resid = int(line.split()[0][:-3].strip()) 
            if line_resid == resid:
                data.append(line)
        return data

    def getResCrd(self, resid: int) -> np.ndarray:
        data = self.getRes(resid)
        crd = []
        for line in data:
            l = line.split()
            if len(l) <= 6:
                crd.append([i for i in l[-3:]])
            else:
                crd.append([i for i in l[-6:-3]])
        return np.asarray(crd, dtype=float)

    def getResAtomLbls(self, resid: int) -> list:
        data = self.getRes(resid)
        atomLbls = []
        for line in data:
            l = line.split()
            atomLbls.append(l[1][0])
        return atomLbls

    def getCOM(self, resid: int) -> np.ndarray:
        atomicMass = {'H': 1.008, 'C': 12.011}
        tmass = 0 
        tweightedCrd = 0 
        for c, l in zip (self.getResCrd(resid), self.getResAtomLbls(resid)):
            tmass += atomicMass.get(l)
            tweightedCrd += c * atomicMass.get(l)
        return tweightedCrd/tmass
    
    def cutAroundRes(self, resid: int, radii: list[float], allCOMs = None) -> list:
        if allCOMs is None:
            allCOMs = self.MP_resCOMs
        
        source_index = list(self.allRes).index(resid)
        source_com = np.array(allCOMs[source_index])
        
        #build KDTree once
        com_array = np.array(allCOMs)
        tree = cKDTree(com_array)
        
        #using isotropic radius taking average if its a list
        radius = np.mean(radii)
        
        #query all indices within radius
        neighbor_indices = tree.query_ball_point(source_com, r=radius)
        
        #map back to residue IDs
        cut = [list(self.allRes)[i] for i in neighbor_indices]
        return cut

    def showIt(self, resids: list) -> None:
        print(f'{len(resids)*self.natomsPerMol}\n')
        for r in resids:
            for c, l in zip (self.getResCrd(r), self.getResAtomLbls(r)):
                print(f'{l}\t {c[0]*10:6.4f}\t {c[1]*10:6.4f}\t {c[2]*10:6.4f}')
    
    # def write_resids_as_gro(self, resids: list[int], output_name: str = "selected.gro"):
    #     """
    #     Save a selection of residues to a new GRO file via XYZ → PDB → GRO conversion,
    #     then restore original box dimensions.
    
    #     Parameters:
    #         resids (list): List of residue IDs to extract and write.
    #         output_name (str): Final GRO filename to save.
    #     """
    #     # Step 1: Create temporary XYZ file
    #     xyz_path = tempfile.NamedTemporaryFile(delete=False, suffix=".xyz").name
    #     with open(xyz_path, "w") as f:
    #         f.write(f"{len(resids) * self.natomsPerMol}\n\n")
    #         for r in resids:
    #             for c, l in zip(self.getResCrd(r), self.getResAtomLbls(r)):
    #                 f.write(f'{l}\t{c[0]*10:.6f}\t{c[1]*10:.6f}\t{c[2]*10:.6f}\n')
    
    #     # Step 2: Convert XYZ → PDB using obabel
    #     pdb_path = xyz_path.replace(".xyz", ".pdb")
    #     subprocess.run(["obabel", "-ixyz", xyz_path, "-opdb", "-O", pdb_path], check=True)
    
    #     # Step 3: Convert PDB → GRO using GROMACS
    #     temp_gro = xyz_path.replace(".xyz", "_converted.gro")
    #     subprocess.run(["gmx", "editconf", "-f", pdb_path, "-resnr", "1", "-o", temp_gro], check=True)
    
    #     # Step 4: Replace last line with original box dimensions
    #     with open(temp_gro, "r") as f:
    #         lines = f.readlines()
    #     lines[-1] = "  {:10.5f}{:10.5f}{:10.5f}\n".format(*self.boxSize)
    #     with open(output_name, "w") as f:
    #         f.writelines(lines)
    
    #     # Clean up temporary files
    #     os.remove(xyz_path)
    #     os.remove(pdb_path)
    #     os.remove(temp_gro)
    #     print(f"GRO file written: {output_name}")   
        
        # def write_gro(self, output_gro: str, resids: list):
        #     """
        #     Extracts given residues from the gro file, writes to a new gro file,
        #     and optionally renumbers using GROMACS editconf.

    #     Parameters:
    #     - resids: list of integers (residue IDs to extract)
    #     - output_gro: filename to write the extracted GRO file
    #     - renumbered_output: optional filename for renumbered output
    #     """
    #     title = "Extracted residues"
    #     box = self.boxSize
    #     extracted_lines = []

    #     atom_index = 1
    #     for resid in resids:
    #         if resid not in self.allRes:
    #             print(f"Warning: Residue {resid} not found.")
    #             continue
    
    #         for line in self.getRes(resid):
    #             parts = line.split()
    #             if len(parts) < 6:
    #                 raise ValueError(f"Line too short to parse: {line}")
    
    #             # Handle merged residue number and name, e.g., "2en-"
    #             merged_res = parts[0]
    #             res_num = ''.join(filter(str.isdigit, merged_res))
    #             res_name = ''.join(filter(str.isalpha, merged_res))
    
    #             try:
    #                 res_num = int(res_num)
    #             except ValueError:
    #                 raise ValueError(f"Cannot extract residue number from '{merged_res}'")
    
    #             atom_name = parts[1]
    #             x, y, z = map(float, parts[3:6])
    
    #             # Optional velocities
    #             vx = vy = vz = None
    #             if len(parts) >= 9:
    #                 vx, vy, vz = map(float, parts[6:9])
    
    #             # Construct formatted GRO line
    #             line_fmt = f"{res_num:5d}{res_name:<5}{atom_name:>5}{atom_index:5d}{x:8.3f}{y:8.3f}{z:8.3f}"
    #             if vx is not None:
    #                 line_fmt += f"{vx:8.4f}{vy:8.4f}{vz:8.4f}"
    
    #             extracted_lines.append(line_fmt)
    #             atom_index += 1
    
    #     if not extracted_lines:
    #         raise ValueError("No residues extracted. Check your input residue IDs.")
    
    #     with open(output_gro, 'w') as f:
    #         f.write(f"{title}\n")
    #         f.write(f"{len(extracted_lines)}\n")
    #         for line in extracted_lines:
    #             f.write(f"{line}\n")
    #         f.write("   {:0.5f}   {:0.5f}   {:0.5f}\n".format(*box))
    
    #     print(f"Extracted GRO written to: {output_gro}")
                
    def write_gro(self, output_path: str, resids_to_write: list, title: str) -> None:
        """
        Write a .gro file with only the specified resids, preserving original formatting.
        """
        lines = []
        atom_index = 1
    
        for resid in resids_to_write:
            res_lines = self.getRes(resid)
            if not res_lines:
                print(f"[WARNING] No lines found for resid {resid}. Skipping.")
                continue
    
            print(f"[INFO] Writing residue {resid} with {len(res_lines)} atoms")
            
            for line in res_lines:
                new_line = line
                lines.append(new_line)
                atom_index += 1
    
        with open(output_path, 'w') as f:
            f.write(f"{title}\n")
            f.write(f"{atom_index - 1}\n")
            for line in lines:
                f.write(f"{line}\n")
            f.write(f"  {self.boxSize[0]:.5f}  {self.boxSize[1]:.5f}  {self.boxSize[2]:.5f}\n")
            
    def renumbered_resids(self, gro_file: str) -> OrderedDict:
        """
        Renumber residues in the GRO file and return an OrderedDict mapping old to new resid numbers.
        """
        renumbered_resids = []
        with open(gro_file, 'r') as f:
            lines = f.readlines()[2:-1]
            for line in lines:
                resid = int(line[:5])
                if len(renumbered_resids) == 0 or resid != renumbered_resids[-1]:
                    renumbered_resids.append(resid)
        return renumbered_resids
                