# PathCalc

This package, called **`pathcalc`**, provides path-finding algorithms for sampling quantum mechanical (QM) paths in molecular systems.  
It supports **two cases**:

- **`pathcalc`** → Grain boundary systems  
  Uses a **biased random walk** algorithm for sampling QM paths in systems with grain boundaries.  

- **`ideal_pathcalc`** → Ideal molecular crystals  
  Uses a **greedy deterministic walk** algorithm for sampling QM paths in ideal (defect-free) molecular systems.  

---

## 📂 Focus: `ideal_pathcalc`

This README explains usage for the **`ideal_pathcalc`** workflow.  
Here, the goal is to **deterministically trace a path of molecules** in an ideal molecular crystal based on **electronic coupling, forward direction, and box margin constraints**.

---

## 📂 Package Structure (Relevant to `ideal_pathcalc`)

```
ideal_pathcalc/
│
├── gro.py              # Parser and utilities for .gro files
├── gmx.py              # Wrapper around GROMACS commands
├── inpman.py           # Input manager for .dat/.spec files
├── top.py              # Topology file handler
├── asann.py            # Neighbor finding (ASANN-based)
├── path_deterministic.py # PathFinder class (core greedy algorithm)
│
└── main_deterministic.py # CLI entrypoint script
```

- Each Python file inside `ideal_pathcalc/` defines a class or helper for a specific part of the workflow.  
- The **core logic** is in `path_deterministic.py`, which defines the `PathFinder` class.  
- The `PathFinder` class also contains the input parameters for `charge_transfer.dat` and `.spec` file. In case of any modifications, change that accordingly.  
- All paths to the input files like **starting structure** file, **topology file**, **Gromacs-SH**, has to be added in `main_deterministic.py` to generate paths.

---

## ⚙️ How It Works (`ideal_pathcalc`)

1. The **user specifies** upon running `main_deterministic.py`:
   - A **starting residue ID**
   - Margins (`Lx`, `Ly`, `Lz`) in nm: residues must be at least this far from the box boundaries
   - Desired **path length** (number of QM sites in the path)
   - An **angle threshold** (degrees): ensures the next step is roughly straight.

2. The program then:
   - Loads the `.gro` structure and computes centers of mass (COMs) of all residues
   - Starts at the chosen residue
   - Iteratively selects the **next residue** based on:
     - **Highest coupling** (from TB Hamiltonian calculation)
     - **Forward direction** relative to the previous step
     - **Angle ≤ threshold**
   - Enforces **boundary checks** so no residue is within the forbidden margin of the box faces
   - Stops early with a **clear error message** if no valid next residue is found or a candidate is too close to the boundary

3. The output is a text file with the selected sequence of residues in the QM path.

---

## ▶️ Usage

1. Clone the repository or copy the package into your working directory.

2. Ensure you have the prerequisites:
   - Python 3.8+  
   - GROMACS (with the SH patch if needed) in your `PATH`  
   - Required Python dependencies (e.g., `numpy`)

3. Run the entrypoint script:

   ```
   python main_deterministic.py
   ```

4. Enter the prompted values (example for a typical run):
   - Enter starting resid (int): `126`  
   - Enter Lx margin (nm): `1.0`  
   - Enter Ly margin (nm): `1.0`  
   - Enter Lz margin (nm): `1.0`  
   - Enter total number of QM sites in path (>=1): `2`  
   - Enter max forward angle threshold (degrees): `5.0`

5. The script will:
   - Create a subfolder `SR_<start_resid>`
   - Run coupling calculations and greedy path selection
   - Save the chosen sequence in `QM_path_<start_resid>.txt` inside the subfolder

### 📄 Output

- **`QM_path_<start_resid>.txt`** — Plain text file, one residue ID per line, showing the selected path (inside `SR_<start_resid>` subfolder).  
- If the path fails (boundary/angle issues), the script prints detailed diagnostics and saves any partial path to the same filename.

Diagnostics include:
- COM coordinates of the failing site
- Box size and allowed ranges
- Which axis violated the margin
- The smallest angle found when no site satisfies the angle threshold

### 🔍 Example Diagnostic Messages

**Boundary violation:**
```
[boundary] Candidate resid 126 COM (nm): x=2.3874, y=1.4212, z=2.8062
[boundary] Box (nm): Lx=3.5000, Ly=2.8000, Lz=3.7000 ; margins: (1.0000,1.0000,1.0000)
[boundary] Allowed ranges: x∈[1.0000,2.5000], y∈[1.0000,1.8000], z∈[1.0000,2.7000]
[boundary] Violations: z∉[1.0000,2.7000]
RuntimeError: site close to boundary, increase size of box. {xcount}/{n} sites are chosen
```

**Angle violation:**
```
RuntimeError: No next residue chosen: no forward candidate within 5° at any coupling level. 
Smallest forward angle encountered was 12.35° at resid 210 (|cpl|=43.211 meV).
```

