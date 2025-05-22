#fast and original logic
import numpy as np
import math

class FirstNN:
    def __init__(self, all_coms, all_resids, subset_resids):
        self.all_resids= all_resids
        self.subset_resids = subset_resids
        self.resid_to_full_index = {resid: idx for idx, resid in enumerate(all_resids)}
        self.coms = np.array([all_coms[self.resid_to_full_index[resid]] for resid in subset_resids])
        self.resid_to_subset_index = {resid: idx for idx, resid in enumerate(subset_resids)}
        self.nres = len(subset_resids)
        
    def get_pbc_vectors(self, coords, nb_atoms=None):
        if nb_atoms is None:
            nb_atoms = coords.shape[0]
        vectors = coords[np.newaxis, :, :] - coords[:nb_atoms, np.newaxis, :]
        return vectors
    
    def get_sorted_distances(self, coords, nb_atoms=None):
        if nb_atoms is None:
            nb_atoms = coords.shape[0]
        vectors = self.get_pbc_vectors(coords, nb_atoms=nb_atoms)
        distances = np.sqrt(np.sum(vectors**2, axis=-1))
        sorted_index_axis1 = np.argsort(distances, axis=-1)
        sorted_index_axis0 = np.arange(nb_atoms)[:, None]
        distances = distances[sorted_index_axis0, sorted_index_axis1]
        vectors = vectors[sorted_index_axis0, sorted_index_axis1]
        return distances, vectors, sorted_index_axis1
    
    def get_SANN(self, all_distances):
        nb_coords = all_distances.shape[1]
        list_sann_CN = []
        list_sann_radius = []
        list_dist_sum = all_distances[:, 1:4].sum(axis=1)
        for (dist_sum, atom_distances) in zip(list_dist_sum, all_distances):
            sann_CN = 3
            while (sann_CN + 1 < nb_coords) and (dist_sum / (sann_CN - 2) >= atom_distances[sann_CN + 1]):
                dist_sum += atom_distances[sann_CN + 1]
                sann_CN += 1
            list_sann_CN.append(sann_CN)
            list_sann_radius.append(dist_sum / (sann_CN - 2))
        return np.array(list_sann_CN), np.array(list_sann_radius)
    
    def dist_to_barycenter(self, nearest_neighbors, nearest_distances, radius):
        list_SA = 1 - nearest_distances / radius
        bary_vector = np.sum(nearest_neighbors * list_SA[:, np.newaxis], axis=0) / np.sum(list_SA)
        return math.sqrt(np.sum(bary_vector**2))
    
    def angular_correction(self, nearest_neighbors, nearest_distances, radius):
        alpha = self.dist_to_barycenter(nearest_neighbors, nearest_distances, radius) / radius
        return (alpha + math.sqrt(alpha**2 + 3 * alpha)) / 3
    
    def get_ASANN(self, sorted_distances, sorted_vectors, sann_CNs, sann_radii):
        nb_coords = sorted_distances.shape[1]
        list_asann_CN = []
        list_asann_radius = []
        for (atom_distances, atom_neighbours, sann_CN, sann_radius) in zip(sorted_distances, sorted_vectors, sann_CNs, sann_radii):
            nearest_distances = atom_distances[1:sann_CN+1]
            nearest_neighbours = atom_neighbours[1:sann_CN+1]
            ang_corr = self.angular_correction(nearest_neighbours, nearest_distances, sann_radius)
            beta = 2*(1 - ang_corr)
            asann_CN = int(beta) + 1
            dist_sum = atom_distances[1:asann_CN+1].sum()
            while (asann_CN + 1 < nb_coords) and (dist_sum / (asann_CN - beta) >= atom_distances[asann_CN + 1]):
                dist_sum += atom_distances[asann_CN + 1]
                asann_CN += 1
            list_asann_CN.append(asann_CN)
            list_asann_radius.append(dist_sum / (asann_CN - beta))
        return np.array(list_asann_CN), np.array(list_asann_radius)
    
    def nearest_neighbors_asann(self, source_resid):
        if source_resid not in self.resid_to_subset_index:
            print(f"Residue ID {source_resid} not found in the file.")
            return []
        
        source_idx = self.resid_to_subset_index[source_resid]
        source_coord = self.coms[source_idx]
        
        other_indices = [i for i in range(len(self.coms)) if i != source_idx]
        other_coords = [self.coms[i] for i in other_indices]
        other_vectors = np.array(other_coords) - source_coord
        other_distances = np.linalg.norm(other_vectors, axis=1)
        
        all_distances = np.insert(other_distances, 0, 0.0)
        all_vectors = np.insert(other_vectors, 0, np.zeros(3), axis=0)
        
        sorted_indices = np.argsort(all_distances)
        sorted_distances = all_distances[sorted_indices].reshape(1, -1)
        sorted_vectors = all_vectors[sorted_indices].reshape(1, -1, 3)
        
        sann_CNs, sann_radii = self.get_SANN(sorted_distances)
        asann_CNs, asann_radii = self.get_ASANN(sorted_distances, sorted_vectors, sann_CNs, sann_radii)
        
        neighbor_sorted_indices = sorted_indices[1:asann_CNs[0] + 1]
        neighbor_indices = [other_indices[i - 1] for i in neighbor_sorted_indices if i > 0]
        neighbors_resids = [self.subset_resids[i] for i in neighbor_indices]
        
        return neighbors_resids
