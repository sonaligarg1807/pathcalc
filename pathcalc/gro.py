import multiprocessing
import numpy as np
import itertools
from itertools import count
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
        
        com_array = np.array(allCOMs)
        tree = cKDTree(com_array)
        
        radius = np.mean(radii)
        
        neighbor_indices = tree.query_ball_point(source_com, r=radius)
        
        cut = [list(self.allRes)[i] for i in neighbor_indices]
        return cut

    def showIt(self, resids: list) -> None:
        print(f'{len(resids)*self.natomsPerMol}\n')
        for r in resids:
            for c, l in zip (self.getResCrd(r), self.getResAtomLbls(r)):
                print(f'{l}\t {c[0]*10:6.4f}\t {c[1]*10:6.4f}\t {c[2]*10:6.4f}')

    #generalised...            
    def write_gro(self, output_path: str, resids_to_write: list):
    
        box = self.boxSize
        lines = []
        atom_index_counter = count(1)
    
        for resid in resids_to_write:
            try:
                atom_lines = self.getRes(resid)
                crds = self.getResCrd(resid)
                atom_lbls = self.getResAtomLbls(resid)
    
                for i, (line, coord, label) in enumerate(zip(atom_lines, crds, atom_lbls)):
                    atom_index = next(atom_index_counter)
                    resid_str = f"{resid:5d}"
                    rename = line.split()[0][-3:]
                    atom_name = f"{label.upper()}QM"
                    x, y, z = coord.astype(float)
    
                    lines.append(f"{resid_str}{rename:<5}{atom_name:>5}{atom_index:5d}{x:8.3f}{y:8.3f}{z:8.3f}")
            except Exception as e:
                print(f"Resid {resid} skipped: {e}")
    
        # Write the new GRO file
        with open(output_path, "w") as f:
            f.write("Extracted GRO file\n")
            f.write(f"{len(lines)}\n")
            f.writelines([line + "\n" for line in lines])
            f.write("{:10.5f}{:10.5f}{:10.5f}\n".format(*box))
            
    def renumbered_resids(self, gro_file: str) -> OrderedDict:
        renumbered_resids = []
        with open(gro_file, 'r') as f:
            lines = f.readlines()[2:-1]
            for line in lines:
                resid = int(line[:5])
                if len(renumbered_resids) == 0 or resid != renumbered_resids[-1]:
                    renumbered_resids.append(resid)
        return renumbered_resids
                