# Quantum Mechanical Path Sampler

This package, called **`pathcalc`**, provides path-finding algorithms for sampling quantum mechanical (QM) paths in molecular systems.
It supports **two cases**:

- **`pathcalc`** → Grain boundary systems
  Uses **biased random walk** algorithms for sampling QM paths in systems with grain boundaries.

- **`ideal_pathcalc`** → Ideal molecular crystals
  Uses a **greedy deterministic walk** algorithm for sampling QM paths in ideal (defect-free) molecular systems.

`pathcalc` and `ideal_pathcalc` share their low-level helpers (`gro.py`, `top.py`, `gmx.py`,
`inpman.py`, `asann.py`) — the canonical copies live in `pathcalc/`, and `ideal_pathcalc`
imports them from there instead of keeping duplicates.

---

## Entry-point scripts

All runnable scripts live at the repo root and end in `_main.py`. They fall into two groups:

### Grain-boundary biased walk (`pathcalc.path`)

Same core algorithm, in increasing order of capability — pick the simplest one that
covers what you need, or read `parallel_walk_autograin_main.py` for the most complete example:

| Script | Description |
|---|---|
| `serial_walk_main.py` | Single-threaded reference implementation. Walks one source residue at a time. |
| `parallel_walk_main.py` | Same algorithm, one worker process per source residue. **This is what `submit_main.sh` runs.** |
| `parallel_walk_autograin_main.py` | Parallel, plus automatically derives the two grains' y-ranges from grain `.gro` files (via `pathcalc/mapped_resids.py`) instead of hardcoding them, and enforces a max path length. |

### Other algorithms

| Script | Algorithm module | Description |
|---|---|---|
| `axis_biased_walk_main.py` | `pathcalc/a_series_path.py` | Axis-biased walk with a geometric "rescue" search and backtracking fallback when the normal ASANN-neighbor bias fails. |
| `directional_walk_main.py` | `pathcalc/directional_bias_path.py` | Biases steps along the actual grain1→grain2 connecting vector (not a fixed axis). Takes CLI flags (`--max-path-len`, `--cutoff`, etc. — run with `--help`). |
| `deterministic_walk_main.py` | `ideal_pathcalc/path_deterministic.py` | Greedy, deterministic walk for ideal (defect-free) crystals — no randomness, highest-coupling forward step each time. |

`pathcalc/archive/` holds earlier, unused draft versions of `directional_bias_path.py` and
`a_series_path.py`, kept for reference only (see `pathcalc/archive/README.md`).

---

## 📂 Focus: `pathcalc` (grain-boundary systems)

Applicable to systems with a **single grain boundary separating two grains**. The goal
is to sample a QM path that starts in one grain and biases each step toward, and
eventually across, the boundary into the other grain.

### Package structure

```
pathcalc/
│
├── path.py                    # PathFinder class — biased walk, fixed y-range grains
├── a_series_path.py           # PathFinder class — axis-biased walk + rescue/fallback
├── directional_bias_path.py   # DirectionalPathFinder — biased along the grain1→grain2 vector
├── resid.py                   # ResidExtractor: pick source residues by COM range
├── mapped_resids.py           # maps grain residue IDs between two .gro files
│
├── gro.py / gmx.py / inpman.py / top.py / asann.py   # shared low-level helpers
└── archive/                    # superseded draft algorithm versions
```

- Three interchangeable biased-walk algorithms live here: `path.py` (fixed y-range grains),
  `a_series_path.py` (axis bias with rescue/fallback), and `directional_bias_path.py`
  (bias along the actual grain1→grain2 vector, not a fixed axis). Each has a matching
  `..._main.py` entrypoint — see the table above.
- `resid.py` and `mapped_resids.py` are shared setup utilities for picking source
  residues and locating the two grains, used regardless of which algorithm runs.

### ⚙️ How it works (`pathcalc`)

1. The **user configures**, at the top of the chosen `..._main.py` script:
   - Input file paths (`.top`, `.gro`, `.mdp`, GROMACS binary path)
   - The two grains — either as fixed y-ranges/axis-ranges, or as separate `.gro`
     files whose ranges are detected automatically
   - Source residues to start walks from — random samples within a COM range, or
     every residue belonging to the two grains
2. For each source residue, the program:
   - Loads the `.gro` structure and computes centers of mass (COMs) of all residues
   - Computes coupling to the ASANN nearest neighbors at the current residue
   - Picks the next residue **probabilistically**, weighted by coupling, subject to a
     **directional/axis bias** toward the other grain
   - Stops the walk once it **crosses into the other grain**, reaches a **max path
     length**, or has **no valid next residue**
3. Each source residue's walk runs in its own `SR_<source_resid>` subfolder, and
   source residues can be walked in parallel (`parallel_walk_main.py`,
   `parallel_walk_autograin_main.py`, `axis_biased_walk_main.py`, `directional_walk_main.py`).

---

## 📂 Focus: `ideal_pathcalc` / `deterministic_walk_main.py`

The goal here is to **deterministically trace a path of molecules** in an ideal molecular
crystal based on **electronic coupling, forward direction, and box margin constraints**.

### Package structure

```
ideal_pathcalc/
│
├── path_deterministic.py   # PathFinder class (core greedy algorithm)
└── __init__.py             # re-exports gro/top from pathcalc/

pathcalc/
├── gro.py      # parser and utilities for .gro files (shared)
├── gmx.py      # wrapper around GROMACS commands (shared)
├── inpman.py   # input manager for .dat/.spec files (shared)
├── top.py      # topology file handler (shared)
└── asann.py    # neighbor finding (ASANN-based) (shared)
```

- The **core logic** is in `path_deterministic.py`, which defines the `PathFinder` class.
- The `PathFinder` class also contains the input parameters for `charge_transfer.dat` and `.spec` file. In case of any modifications, change that accordingly.
- All paths to the input files like **starting structure** file, **topology file**, **Gromacs-SH**, have to be added in `deterministic_walk_main.py` to generate paths.

---

## ⚙️ How it works (`deterministic_walk_main.py`)

1. The **user specifies** upon running `deterministic_walk_main.py`:
   - A **starting residue ID**
   - Margins (`Lx`, `Ly`, `Lz`) in nm: residues must be at least this far from the box boundaries
   - Desired **path length** (number of QM sites in the path)
   - An **angle threshold** (degrees): ensures the next step is roughly straight.

2. The program then:
   - Loads the `.gro` structure and computes centers of mass (COMs) of all residues
   - Starts at the chosen residue
   - Iteratively selects the **next residue** based on:
     - **Highest coupling** from **1 step NOM onsite** calculations
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
   - Required Python dependencies (e.g., `numpy`, `scipy`)

3. Run the entrypoint script:

   ```
   python deterministic_walk_main.py
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

For the grain-boundary scripts (`serial_walk_main.py`, `parallel_walk_main.py`,
`parallel_walk_autograin_main.py`, `axis_biased_walk_main.py`, `directional_walk_main.py`),
edit the file paths at the top of the script (or pass CLI flags, for `directional_walk_main.py`)
and run directly, e.g. `python parallel_walk_main.py`, or submit via `submit_main.sh` on a
grid-engine cluster.

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
