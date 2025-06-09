import math
import random
import os
import shutil
import time
import glob
from pathcalc import top, gmx, inpman, asann


class PathFinder:
    def __init__(self, gro_obj, all_coms: list, all_resids, ham_file, topFilePath, gmxPath, mdpFilePath, y_ranges, cutoff):
        self.gro_obj = gro_obj
        self.all_coms = all_coms
        self.all_resids = all_resids
        self.coms = {resid: com for resid, com in zip(all_resids, all_coms)}
        self.ham_file = ham_file
        self.topFilePath = topFilePath
        self.mdpFilePath = mdpFilePath
        self.gmxPath = gmxPath
        self.y_ranges = y_ranges
        self.cutoff = cutoff
        
        
        
    def get_com(self, resid):
        if resid in self.coms:
            return self.coms[resid]
        else:
            print(f"Resid {resid} not found in all_coms.")
            return None
        
    def cleanup_specific_files(self, keep_extensions=[".txt"]):
        for filename in os.listdir():
            if os.path.isdir(filename) or any(filename.endswith(ext) for ext in keep_extensions):
                continue
            try:
                os.remove(filename)
            except Exception as e:
                print(f"Could not delete {filename}: {e}")
        
    def avg_cpl(self, ham_file, source_resid, topFilePath, gmxPath, mdpFilePath):
        start = time.time()
        nn = self.gro_obj.cutAroundRes(source_resid, [1., 1., 1.], allCOMs=self.all_coms)
        print(f"Time taken to cut around residue {source_resid}: {time.time() - start} seconds")
        if not nn:
            print(f"No neighbors found for residue {source_resid}. Ending walk.")
            return None
        
        nn_finder = asann.FirstNN(self.all_coms, self.all_resids, subset_resids=nn)
        neighbors = nn_finder.nearest_neighbors_asann(source_resid)
        if not neighbors:
            print(f"No neighbors found for residue {source_resid}. Ending walk.")
            return None
        
        resids_to_write = [source_resid] + neighbors
        self.gro_obj.write_gro("temporary.gro", resids_to_write)
        
        gmx_runner = gmx.gmx(exe="gmx", gro="temporary.gro")
        gmx_runner.renum()
        
        #getting renumbered resids
        renumbered_resids = self.gro_obj.renumbered_resids("temporary.gro")
        resid_mapping = dict(zip(resids_to_write, renumbered_resids))
    
        # modifying topology according to selected residues
        new_top_path = "pen-esp.top"
        shutil.copy(topFilePath, new_top_path)
        itp_path = "/data/sgarg/pentacene/sampling_qm_zone_algorithm/gb-pen/input_files" #update the path
        itp_files = os.path.join(itp_path, "*.itp")
        for itp_file in glob.glob(itp_files):
            shutil.copy(itp_file, ".")
        pen_top = top(new_top_path)
        pen_top.update_molecule_count("molecule", len(resids_to_write))
    
        # start = time.time()
        gmx_runner = gmx.gmx(exe=gmxPath, gro="temporary.gro", top=new_top_path, mdp=mdpFilePath, tpr="ham")
        try:
            gmx_runner.grompp()
        except Exception as e:
            print(f"Error running grompp: {e}")
        # print(f"Time taken: {time.time() - start} seconds")

        if not os.path.exists("pen.spec"):
            spec_manager = inpman.inpManager("spec")
            spec_manager.update(natoms=36, nelectrons=102, nallorbitals=102, nfragorbs=1, fragorbs=51)
            spec_manager.save('pen.spec')
        else:
            print("spec file already exists.")
        
        start = time.time()
        renum_sites = [resid_mapping[r] for r in resids_to_write]
        
        ct_manager = inpman.inpManager("ct")
        ct_manager.update(sites=renum_sites, 
                        seed=random.randint(1, 100), chargecarrier="hole", 
                        atomindex=[1, 20, 29], typefiles="pen.spec", 
                        jobtype="NOM", internalrelax="onsite")
        ct_manager.save('charge-transfer.dat')
        
        gmx_runner = gmx.gmx(exe=gmxPath, tpr="ham")
        gmx_runner.mdrun(nt=1)
        if not os.path.exists(ham_file):
            print(f"File {ham_file} not found.")
            return None
        
        extracted = {}
        with open(ham_file, 'r') as file:
            for line in file:
                if line.strip() and not line.startswith(("#", "@")):
                    parts = line.split()
                    try:
                        values = [float(parts[i]) for i in range(2, 2 + len(neighbors))]
                        for neighbor_resid, cpl_value in zip(neighbors, values):
                            extracted[neighbor_resid] = abs(cpl_value) * 1000  # Convert to meV
                    except (IndexError, ValueError):
                        print(f"Error processing line: {line.strip()}")
                        continue
                    
        for neigh, val in extracted.items():
            print(f"  Neighbor {neigh}: {val} meV")
        
        self.cleanup_specific_files()
        
        return extracted
    
    def probabilities(self, coupling_data, beta=(1 / 25.7)):
        if not coupling_data:
            return {}
        max_coupling = max(coupling_data.values())
        exp_weights = {resid: math.exp(beta * (cpl - max_coupling)) for resid, cpl in coupling_data.items()}
        total_weight = sum(exp_weights.values())
        probabilities = {resid: (weight / total_weight if total_weight > 0 else 0.0) for resid, weight in exp_weights.items()}
        return probabilities
    
    def check_too_far(self, source_com, selected_neighbor_com):
        distance = math.dist(source_com, selected_neighbor_com)
        print(f"Distance between source_resid and selected neighbor_resid: {distance:.2f} nm")
        return distance > self.cutoff  # for y45 gb system it was 0.70nm and for single crystal and y30 it is 0.65nm; b30:0.60nm; b60: 0.85nm; b45: 0.70nm
    
    def check_y_range(self, resid):
        com = self.get_com(resid)
        if com is None:
            return None
        y_COM = com[1]
        if self.y_ranges[0][0] <= y_COM <= self.y_ranges[0][1]:
            print(f"Resid {resid} satisfies y_COM {y_COM:.2f} in range {self.y_ranges[0]}")
            return "in_range_1"
        elif self.y_ranges[1][0] <= y_COM <= self.y_ranges[1][1]:
            print(f"Resid {resid} satisfies y_COM {y_COM:.2f} in range {self.y_ranges[1]}")
            return "in_range_2"
        else:
            print(f"Resid {resid} does NOT satisfy y_COM {y_COM:.2f} in any range")
            return "out_of_range"
        
    def check_visited(self, selected_neighbor, visited_resids):
        return selected_neighbor in visited_resids
    
    def select_next_neighbor(self, probabilities, source_resid, first_source_resid, visited_resids, sampled_paths):
        """
        Selects a valid neighbor for the given source_resid based on probabilities.
        If no valid neighbor is found, backtracks and calls fallback_select_neighbor.
        """
        while True:
            random_value = random.uniform(0, 1)
            print(f"Random number for source_resid {source_resid}: {random_value}")
            
            source_com = self.get_com(source_resid)
            source_range = self.check_y_range(first_source_resid)
    
            sorted_neighbors = sorted(probabilities.items(), key=lambda x: abs(x[1] - random_value))
            for target_resid, _ in sorted_neighbors:
                if target_resid in visited_resids:
                    print(f"Neighbor {target_resid} has already been visited. Skipping...")
                    continue
    
                target_com = self.get_com(target_resid)
                
                #check range
                if source_range == "in_range_1" and target_com[1] <= source_com[1]:
                    print(f"Neighbor {target_resid} does not move forward in range 1. Skipping...")
                    continue
                elif source_range == "in_range_2" and target_com[1] >= source_com[1]:
                    print(f"Neighbor {target_resid} does not move forward in range 2. Skipping...")
                    continue
    
                #check too far
                if self.check_too_far(source_com, target_com):
                    print(f"Neighbor {target_resid} is too far. Skipping...")
                    continue
    
                selected_neighbor = target_resid
                return selected_neighbor
    
            # Backtracking
            while sampled_paths:
                del sampled_paths[-1]  # Remove last
                if not sampled_paths:
                    print("No previous residues to backtrack to. Stopping.")
                    return None
    
                fallback_resid = sampled_paths[-1]
                selected_neighbor = self.fallback_select_neighbor(fallback_resid, visited_resids)
                print(f"Selected neighbor for fallback resid {fallback_resid}: {selected_neighbor}")
                if selected_neighbor:
                    print(f"Backtracking successful. New selected neighbor: {selected_neighbor}")
                    return selected_neighbor
    
            print("No valid neighbors found. Ending.")
            return None


    def fallback_select_neighbor(self, fallback_resid, visited_resids):
        avg_cpl_values = self.avg_cpl(self.ham_file, fallback_resid,
                                    self.topFilePath, self.gmxPath, self.mdpFilePath)
        
        probabilities = self.probabilities(avg_cpl_values)
        
        random_value = random.uniform(0, 1)
        
        source_com = self.get_com(fallback_resid)
        source_range = self.check_y_range(fallback_resid)
    
        sorted_neighbors = sorted(probabilities.items(), key=lambda x: abs(x[1] - random_value))
        for target_resid, _ in sorted_neighbors:
            if target_resid in visited_resids:
                print(f"Neighbor {target_resid} has already been visited. Skipping...")
                continue
    
            target_com = self.get_com(target_resid)
    
            #check range
            if source_range == "in_range_1" and target_com[1] <= source_com[1]:
                print(f"Neighbor {target_resid} does not move forward in range 1. Skipping...")
                continue
            elif source_range == "in_range_2" and target_com[1] >= source_com[1]:
                print(f"Neighbor {target_resid} does not move forward in range 2. Skipping...")
                continue
    
            if self.check_too_far(source_com, target_com):
                print(f"Neighbor {target_resid} is too far. Skipping...")
                continue
    
            selected_neighbor = target_resid
            print(f"Selected neighbor for fallback: {selected_neighbor} with probability {probabilities[target_resid]}")
            return selected_neighbor
    
        print(f"No valid neighbors found for fallback_resid {fallback_resid}. Backtracking further...")
        return None