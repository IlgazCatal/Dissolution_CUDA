# main.py  —  dissolution_cuda with solvent/solute library and all fixes
import numpy as np
import math
import time
import os
import sys
import importlib
import matplotlib


def _pick_backend():
    """Choose a matplotlib backend that will actually import on this machine.

    matplotlib.use() is lazy: it records the choice but does not import the
    backend until the first figure is created. So a try/except wrapped around
    use() never fires here -- the ImportError surfaces much later, at the
    unguarded plt.figure() in main_driver, after the whole simulation has
    already run. Probe the backend module directly instead, and defer to an
    explicit MPLBACKEND when the user has set one.
    """
    if os.environ.get("MPLBACKEND"):
        return None
    candidates = ["TkAgg", "Agg"]
    # On Linux, importing the Tk backend succeeds even with no display, so the
    # import probe alone would pick a backend that then dies with TclError.
    if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        candidates.remove("TkAgg")
    for backend in candidates:
        try:
            importlib.import_module("matplotlib.backends.backend_" + backend.lower())
            return backend
        except Exception:
            continue
    return None


_backend = _pick_backend()
if _backend:
    matplotlib.use(_backend)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from numba import cuda
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_normal_float32,
)

# ============================================================
#  SOLVENT LIBRARY
# ============================================================
# PROVENANCE -- solvent-dependent solubility ("Cs_ratio" below)
#
# Cs_ratio is the ratio of molar solubility in that solvent to solubility in
# water at 298.15 K. Sources, roughly in decreasing confidence:
#   - NaCl in DMSO 0.0775 M: Unni et al., "Solubilities of sodium chloride in
#     binary mixtures of water and diphenyl sulfoxide / DMSO", 1979.
#   - Ibuprofen: Jouyban et al., JCED 55 (2010) 5252, ~1.2055 M in ethanol at
#     298.15 K; other groups report up to ~4.3 M, so spread is large.
#   - Benzoic acid: markedly MORE soluble in alcohols/acetone than in water,
#     and its solubility peaks in the more polar protic solvents.
#   - Sucrose: ~200 g/100 mL water at 25 C => 5.87 M
#     (200 g / 342.2965 g/mol / 0.1 L). NOTE: some secondary sources quote
#     ~89 g/100 mL (~2.6 M); the 2.6 M figure is NOT what 200 g/100 mL gives.
#   - Aspirin: NTP 50782 lists it as soluble in acetone/ethanol and
#     essentially insoluble in petroleum ether, matching the hexane ratio.
#   - Caffeine in DMSO ~0.025 M; caffeine in hydrocarbons is very low.
# Values marked in-line as "essentially insoluble" are order-of-magnitude
# placeholders (~1e-3 ratio), NOT measurements -- they only need to keep Cs far
# below the single-particle concentration step so the solvent reads as
# non-solubilising. They are deliberately coarse, and were not invented to
# produce a particular dissolved fraction.
#
# Aqueous Cs_sim values are the same data expressed directly, and match the
# ratios within rounding.
SOLVENTS = {
    "water": {
        "name": "Water (H2O)",
        "viscosity_Pa_s": 1.0e-3, "density_kg_m3": 997.0,
        "Tm_K": 273.15, "Tb_K": 373.15, "D_scale": 1.0,
        # Solubility relative to water (ratio of molar solubility in this
        # solvent to that in water, at 298.15 K). Water is the reference, so
        # every ratio is 1.0 by construction; the solvent is omitted here.
    },
    "ethanol": {
        "name": "Ethanol (C2H5OH)",
        "viscosity_Pa_s": 1.1e-3, "density_kg_m3": 789.0,
        "Tm_K": 159.0, "Tb_K": 351.5, "D_scale": 0.9,
        "Cs_ratio": {
            "nacl": 0.0016,       # 0.0088 M vs 5.44 M in water
            "kcl": 0.00096,       # 0.0039 M vs 4.07 M
            "sucrose": 0.0058,    # 0.034 M, sparingly soluble
            "caffeine": 0.26,     # 0.029 M vs 0.112 M
            "benzoic_acid": 129.0,  # 3.6 M vs 0.028 M -- far MORE soluble
            "ibuprofen": 12000.0,   # 1.2 M vs 1e-4 M
            "aspirin": 60.0,        # 1.11 M vs 0.0185 M
            "caco3": 0.02,
        },
    },
    "methanol": {
        "name": "Methanol (CH3OH)",
        "viscosity_Pa_s": 0.55e-3, "density_kg_m3": 792.0,
        "Tm_K": 175.5, "Tb_K": 337.8, "D_scale": 1.8,
        "Cs_ratio": {
            "nacl": 0.04,        # ~0.19-0.24 M vs 5.44 M
            "kcl": 0.014,        # ~0.056 M vs 4.07 M
            "sucrose": 0.010,    # ~0.06 M
            "caffeine": 0.45,    # ~0.050 M vs 0.112 M
            "benzoic_acid": 164.0,  # 4.6 M vs 0.028 M
            "ibuprofen": 56000.0,   # ~5.6 M vs 1e-4 M
            "aspirin": 30.0,
            "caco3": 0.01,
        },
    },
    "acetone": {
        "name": "Acetone ((CH3)2CO)",
        "viscosity_Pa_s": 0.32e-3, "density_kg_m3": 784.0,
        "Tm_K": 178.5, "Tb_K": 329.2, "D_scale": 3.1,
        "Cs_ratio": {
            "nacl": 0.001,       # essentially insoluble
            "kcl": 0.001,        # essentially insoluble
            "sucrose": 0.001,    # essentially insoluble
            "caffeine": 0.55,    # 0.062 M vs 0.112 M
            "benzoic_acid": 125.0,  # 3.5 M vs 0.028 M
            "ibuprofen": 50000.0,   # ~5.0 M vs 1e-4 M
            "aspirin": 30.0,        # >=0.56 M vs 0.0185 M
            "caco3": 0.001,
        },
    },
    "dmso": {
        "name": "DMSO ((CH3)2SO)",
        "viscosity_Pa_s": 2.0e-3, "density_kg_m3": 1100.0,
        "Tm_K": 291.7, "Tb_K": 462.0, "D_scale": 0.5,
        "Cs_ratio": {
            "nacl": 0.013,       # 0.072 M vs 5.44 M (Unni 1979)
            "kcl": 0.005,        # low, below NaCl
            "sucrose": 1.0,      # qualitatively similar to water
            "caffeine": 0.22,    # ~0.025 M vs 0.112 M
            "benzoic_acid": 50.0,
            "ibuprofen": 38000.0,   # ~3.8 M, best of the organic solvents
            "aspirin": 30.0,        # >=0.56 M
            "caco3": 0.02,
        },
    },
    "hexane": {
        "name": "n-Hexane (C6H14)",
        "viscosity_Pa_s": 0.31e-3, "density_kg_m3": 659.0,
        "Tm_K": 177.8, "Tb_K": 341.9, "D_scale": 3.2,
        "Cs_ratio": {
            "nacl": 0.001,       # insoluble in hydrocarbons
            "kcl": 0.001,        # insoluble
            "sucrose": 0.001,    # insoluble
            "caffeine": 0.02,    # very low (no direct measurement available)
            "benzoic_acid": 3.6,   # 0.10 M vs 0.028 M
            "ibuprofen": 0.5,      # low
            "aspirin": 0.005,      # insoluble (NTP 50782: petroleum ether)
            "caco3": 0.001,
        },
    },
}

# ============================================================
#  SOLUTE LIBRARY
# ============================================================
SOLUTES = {
    "nacl": {
        "name": "Sodium chloride (NaCl)",
        "k_ref_sim": 5.0e-3, "Ea_J_mol": 20e3,
        "Cs_sim": 5.44, "dH_diss_J_mol": 3.9e3,
        "preferred_solvents": ["water", "dmso", "methanol"],
    },
    "kcl": {
        "name": "Potassium chloride (KCl)",
        "k_ref_sim": 4.0e-3, "Ea_J_mol": 25e3,
        "Cs_sim": 4.07, "dH_diss_J_mol": 17.2e3,
        "preferred_solvents": ["water"],
    },
    "sucrose": {
        "name": "Sucrose (C12H22O11)",
        "k_ref_sim": 5.0e-4, "Ea_J_mol": 5.4e3,
        "Cs_sim": 5.87, "dH_diss_J_mol": 5.4e3,
        "preferred_solvents": ["water"],
    },
    "caffeine": {
        "name": "Caffeine (C8H10N4O2)",
        "k_ref_sim": 2.0e-3, "Ea_J_mol": 20e3,
        "Cs_sim": 0.112, "dH_diss_J_mol": 20e3,
        "preferred_solvents": ["water", "ethanol", "dmso"],
    },
    "benzoic_acid": {
        "name": "Benzoic acid (C7H6O2)",
        "k_ref_sim": 3.0e-3, "Ea_J_mol": 22e3,
        "Cs_sim": 0.028, "dH_diss_J_mol": 22e3,
        "preferred_solvents": ["water", "ethanol", "acetone"],
    },
    "ibuprofen": {
        "name": "Ibuprofen (C13H18O2)",
        "k_ref_sim": 1.5e-3, "Ea_J_mol": 25e3,
        "Cs_sim": 1.0e-4, "dH_diss_J_mol": 25e3,
        "preferred_solvents": ["ethanol", "acetone", "dmso"],
    },
    "aspirin": {
        "name": "Aspirin (C9H8O4)",
        "k_ref_sim": 2.5e-3, "Ea_J_mol": 27e3,
        "Cs_sim": 0.0185, "dH_diss_J_mol": 27e3,
        "preferred_solvents": ["water", "ethanol", "acetone"],
    },
    "caco3": {
        "name": "Calcium carbonate (CaCO3)",
        "k_ref_sim": 5.0e-4, "Ea_J_mol": 45e3,
        "Cs_sim": 1.3e-4, "dH_diss_J_mol": -12e3,   # retrograde: less soluble as T rises
        "preferred_solvents": ["water"],
    },
}

# ------------------------------------------------------------
#  System config (populated by configure_system)
# ------------------------------------------------------------
CFG = {}
T_REF = 298.15
D_BASE_SIM = 5.0e-3      # reference sim-diffusivity (water @ 298 K)


def list_options():
    print("Available solvents:")
    for k, v in SOLVENTS.items():
        print(f"  {k:10s} - {v['name']}")
    print("\nAvailable solutes:")
    for k, v in SOLUTES.items():
        pref = ", ".join(v["preferred_solvents"])
        print(f"  {k:14s} - {v['name']}  (prefers: {pref})")


def _arrhenius_rate(k_ref, Ea, T, T_ref=T_REF):
    R = 8.314
    return k_ref * math.exp(-Ea / R * (1.0 / T - 1.0 / T_ref))


def _vant_hoff_Cs(Cs_ref, dH, T, T_ref=T_REF):
    R = 8.314
    return Cs_ref * math.exp(-dH / R * (1.0 / T - 1.0 / T_ref))


def configure_system(solute_key, solvent_key, T_K=298.15):
    """Populate the global CFG dict from a (solute, solvent, T) choice."""
    global CFG
    if solute_key not in SOLUTES:
        raise KeyError(f"Unknown solute '{solute_key}'. Options: {list(SOLUTES)}")
    if solvent_key not in SOLVENTS:
        raise KeyError(f"Unknown solvent '{solvent_key}'. Options: {list(SOLVENTS)}")

    solute  = SOLUTES[solute_key]
    solvent = SOLVENTS[solvent_key]

    if solvent_key not in solute["preferred_solvents"]:
        print(f"[warning] {solute['name']} is not particularly soluble in "
              f"{solvent['name']}.  Suggested: {solute['preferred_solvents']}.")
    if not (solvent["Tm_K"] + 1.0 < T_K < solvent["Tb_K"] - 1.0):
        print(f"[warning] T = {T_K} K lies outside {solvent['name']} liquid range "
              f"({solvent['Tm_K']:.1f} - {solvent['Tb_K']:.1f} K).")

    k_ref_T      = _arrhenius_rate(solute["k_ref_sim"], solute["Ea_J_mol"], T_K)
    D_eff        = D_BASE_SIM * solvent["D_scale"] * (T_K / T_REF)
    Cs_eff       = _vant_hoff_Cs(solute["Cs_sim"], solute["dH_diss_J_mol"], T_K)
    # Solubility is strongly solvent-dependent (literature: NaCl is ~5.4 M in
    # water but ~0.009 M in ethanol; benzoic acid is ~0.028 M in water but
    # ~3.6 M in ethanol -- the trend reverses between solute classes). Cs_sim
    # is the aqueous value; Cs_ratio expresses the solvent effect relative to
    # water and is what gives different solvents genuinely different behaviour.
    Cs_eff *= solvent.get("Cs_ratio", {}).get(solute_key, 1.0)
    k_attach_eff = 5.0e-5 * (D_eff / D_BASE_SIM)

    CFG = {
        "solute_key":   solute_key,
        "solvent_key":  solvent_key,
        "solute_name":  solute["name"],
        "solvent_name": solvent["name"],
        "T":            T_K,
        "k_ref":        k_ref_T,
        "Ea":           solute["Ea_J_mol"],
        "Cs":           Cs_eff,
        "D":            D_eff,
        "k_attach":     k_attach_eff,
        "viscosity":    solvent["viscosity_Pa_s"],
    }
    return CFG


# ---------- User parameters ----------
SEED = 0
rng = np.random.default_rng(SEED)

# Solid block
N_x, N_y, N_z = 14, 14, 6
spacing = 0.9
box = 20.0

# Time
dt = 1.0
n_steps = 600

# Concentration grid (on GPU)
gridN = 64
grid_dx = box / gridN

# Visualization parameters
solid_color     = (0.2, 0.4, 0.8)
inner_color     = (0.6, 0.6, 0.6)
dissolved_color = (1.0, 0.3, 0.2)

# ---------- Initial configuration ----------
xs = np.arange(N_x) * spacing
ys = np.arange(N_y) * spacing
zs = np.arange(N_z) * spacing
X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
solid_pos = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
center_offset = (box - (N_x - 1) * spacing) / 2.0
solid_pos += center_offset
Np = len(solid_pos)
state = np.zeros(Np, dtype=np.int32)

# Dissolved positions live on the GPU; the host copy is only built at the end
dissolved_host = np.empty((0, 3), dtype=np.float32)


def idx_to_grid(i):
    x = i // (N_y * N_z)
    y = (i // N_z) % N_y
    z = i % N_z
    return x, y, z


# Precompute surface mask (CPU)
surface_mask = np.zeros(Np, dtype=np.int32)
for i in range(Np):
    x, y, z = idx_to_grid(i)
    if x == 0 or x == N_x - 1 or y == 0 or y == N_y - 1 or z == 0 or z == N_z - 1:
        surface_mask[i] = 1
surface_idx = np.flatnonzero(surface_mask == 1).astype(np.int32)

# Mean smoothed bin count that the *un-dissolved* lattice alone contributes to
# the grid cell of a surface site. Every surface site is counted into its own
# cell and then spread over the 27 cells of the 3x3x3 smoothing kernel, so this
# is the irreducible "empty solution" reading that C_surf must have subtracted
# before it can be compared against a saturation concentration. It depends on
# gridN/spacing, so it is derived here rather than hardcoded.
def _solid_cell_baseline():
    p = solid_pos[surface_idx]
    cells = np.stack([
        np.clip((p[:, d] / grid_dx).astype(np.int64), 0, gridN - 1)
        for d in range(3)], axis=1)
    unique, counts = np.unique(cells, axis=0, return_counts=True)
    return float(counts.mean() / 27.0)


SOLID_CELL_BASELINE = _solid_cell_baseline()


# ============================================================
#  GPU kernels
# ============================================================

SMOOTH_TPB = 8
SMOOTH_SIDE = SMOOTH_TPB + 2          # tile + one-cell halo


@cuda.jit(device=True)
def _cell_xyz(x, y, z, grid_dx, gridN):
    """Cell coordinates of a point, clamped into the grid."""
    ix = int(x / grid_dx)
    iy = int(y / grid_dx)
    iz = int(z / grid_dx)
    if ix < 0: ix = 0
    if iy < 0: iy = 0
    if iz < 0: iz = 0
    if ix >= gridN: ix = gridN - 1
    if iy >= gridN: iy = gridN - 1
    if iz >= gridN: iz = gridN - 1
    return ix, iy, iz


@cuda.jit(device=True)
def _nearest_solid_site(x, y, z, center_offset, spacing, Nx, Ny, Nz):
    """Index of the closest lattice site of the solid block (box lattice)."""
    ix = int(math.floor((x - center_offset) / spacing + 0.5))
    iy = int(math.floor((y - center_offset) / spacing + 0.5))
    iz = int(math.floor((z - center_offset) / spacing + 0.5))
    if ix < 0: ix = 0
    if iy < 0: iy = 0
    if iz < 0: iz = 0
    if ix > Nx - 1: ix = Nx - 1
    if iy > Ny - 1: iy = Ny - 1
    if iz > Nz - 1: iz = Nz - 1
    return (ix * Ny + iy) * Nz + iz


@cuda.jit
def brownian_step_kernel(positions, n, rng_states, sigma, box):
    """Advance each dissolved particle by one Brownian step, reflect at walls."""
    i = cuda.grid(1)
    if i >= n:
        return
    # Built-in N(0,1) draws — no manual Box-Muller, no math.* calls
    gx = xoroshiro128p_normal_float32(rng_states, i)
    gy = xoroshiro128p_normal_float32(rng_states, i)
    gz = xoroshiro128p_normal_float32(rng_states, i)

    positions[i, 0] += sigma * gx
    positions[i, 1] += sigma * gy
    positions[i, 2] += sigma * gz

    # Reflecting boundaries
    for d in range(3):
        v = positions[i, d]
        if v < 0.0:
            positions[i, d] = -v
        elif v > box:
            positions[i, d] = 2.0 * box - v


@cuda.jit
def zero_grid_kernel(grid):
    """Set every cell of the 3D concentration grid to 0.0."""
    i, j, k = cuda.grid(3)
    if i < grid.shape[0] and j < grid.shape[1] and k < grid.shape[2]:
        grid[i, j, k] = 0.0


@cuda.jit
def bin_positions_kernel(positions, n, grid, gridN, grid_dx):
    """Atomically increment the grid cell containing each dissolved particle."""
    i = cuda.grid(1)
    if i >= n:
        return
    ix, iy, iz = _cell_xyz(positions[i, 0], positions[i, 1], positions[i, 2],
                           grid_dx, gridN)
    cuda.atomic.add(grid, (ix, iy, iz), 1.0)


@cuda.jit
def smooth_grid_kernel(grid_in, grid_out, gridN):
    """One-pass 3x3x3 box smoothing, reading a shared-memory halo tile."""
    i, j, k = cuda.grid(3)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z

    sh = cuda.shared.array(shape=(SMOOTH_SIDE, SMOOTH_SIDE, SMOOTH_SIDE),
                           dtype=np.float32)

    i0 = i - tx
    j0 = j - ty
    k0 = k - tz

    # Redundantly fill the halo tile so every thread reads only shared memory
    tid = tx + ty * SMOOTH_TPB + tz * SMOOTH_TPB * SMOOTH_TPB
    stride = SMOOTH_TPB ** 3
    area = SMOOTH_SIDE * SMOOTH_SIDE
    for s in range(tid, SMOOTH_SIDE ** 3, stride):
        a = s // area
        r = s - a * area
        b = r // SMOOTH_SIDE
        c = r - b * SMOOTH_SIDE
        gi = i0 + a - 1
        gj = j0 + b - 1
        gk = k0 + c - 1
        if 0 <= gi < gridN and 0 <= gj < gridN and 0 <= gk < gridN:
            sh[a, b, c] = grid_in[gi, gj, gk]
        else:
            sh[a, b, c] = 0.0
    cuda.syncthreads()

    if i >= gridN or j >= gridN or k >= gridN:
        return

    s = np.float32(0.0)
    for da in range(3):
        for db in range(3):
            for dc in range(3):
                s += sh[tx + da, ty + db, tz + dc]

    # In-range neighbours, matching the original skip-outside-bounds average
    cx = 3 - (1 if i == 0 else 0) - (1 if i == gridN - 1 else 0)
    cy = 3 - (1 if j == 0 else 0) - (1 if j == gridN - 1 else 0)
    cz = 3 - (1 if k == 0 else 0) - (1 if k == gridN - 1 else 0)
    grid_out[i, j, k] = s / (cx * cy * cz)


@cuda.jit
def gather_surface_kernel(solid_pos, surface_idx, n_s, grid, grid_dx, gridN,
                          out):
    """Concentration at every surface site of the solid block."""
    i = cuda.grid(1)
    if i >= n_s:
        return
    p = surface_idx[i]
    ix, iy, iz = _cell_xyz(solid_pos[p, 0], solid_pos[p, 1], solid_pos[p, 2],
                           grid_dx, gridN)
    out[i] = grid[ix, iy, iz]


@cuda.jit
def probe_kernel(positions, n, grid, grid_dx, gridN, center_offset, spacing,
                 Nx, Ny, Nz, out_c, out_site):
    """Concentration and nearest solid site for every dissolved particle."""
    i = cuda.grid(1)
    if i >= n:
        return
    x = positions[i, 0]; y = positions[i, 1]; z = positions[i, 2]
    ix, iy, iz = _cell_xyz(x, y, z, grid_dx, gridN)
    out_c[i] = grid[ix, iy, iz]
    out_site[i] = _nearest_solid_site(x, y, z, center_offset, spacing,
                                      Nx, Ny, Nz)


@cuda.jit
def append_kernel(dst, offset, src, m):
    """Write m new particles into the device buffer at row `offset`."""
    i = cuda.grid(1)
    if i >= m:
        return
    dst[offset + i, 0] = src[i, 0]
    dst[offset + i, 1] = src[i, 1]
    dst[offset + i, 2] = src[i, 2]


@cuda.jit
def compact_kernel(src, dst, keep_idx, k):
    """Gather the surviving particles to the front of a second buffer."""
    i = cuda.grid(1)
    if i >= k:
        return
    j = keep_idx[i]
    dst[i, 0] = src[j, 0]
    dst[i, 1] = src[j, 1]
    dst[i, 2] = src[j, 2]


@cuda.jit
def copy_rows_kernel(src, dst, n):
    i = cuda.grid(1)
    if i >= n:
        return
    dst[i, 0] = src[i, 0]
    dst[i, 1] = src[i, 1]
    dst[i, 2] = src[i, 2]


# ============================================================
#  CPU-side helpers
# ============================================================

def detach_step(surface_idx, C_surf, state):
    """Decide which surface solid particles dissolve this step (vectorized)."""
    if surface_idx.size == 0:
        return np.empty(0, dtype=np.int32)
    k  = CFG["k_ref"]
    Cs = CFG["Cs"]
    prob = k * (1.0 - np.minimum(C_surf / Cs, 1.0)) * dt
    np.clip(prob, 0.0, None, out=prob)
    draw = rng.random(prob.size)
    return surface_idx[(draw < prob) & (state[surface_idx] == 0)]


def reattach_mask(C_loc, sites, n):
    """Keep-mask for dissolved particles that re-attach to an empty surface site.

    Returns None when nothing is removed.
    """
    drop = np.zeros(n, dtype=bool)
    cand = C_loc > CFG["Cs"]
    if cand.any():
        cand &= rng.random(n) < CFG["k_attach"] * dt
        if cand.any():
            ss = sites[cand]
            valid = (state[ss] == 0) & (surface_mask[ss] == 1)
            drop[np.flatnonzero(cand)[valid]] = True
    if not drop.any():
        return None
    return ~drop


# ============================================================
#  Main simulation loop
# ============================================================

def run_simulation():
    """Run the dissolution simulation with the currently configured CFG."""
    global dissolved_host, rng

    if not CFG:
        raise RuntimeError(
            "configure_system(solute, solvent, T) must be called before "
            "run_simulation()")

    # Reset the per-run mutable state so that repeated calls in a single
    # process are independent and reproducible. `state` marks sites that have
    # already dissolved and detach_step refuses any site with state != 0, so
    # without this a second run inherits a partially dissolved solid. `rng`
    # likewise keeps advancing across calls. With both reset, consecutive runs
    # of the same configuration are bit-identical.
    state[:] = 0
    rng = np.random.default_rng(SEED)

    rng_states = None
    times    = []
    released = []
    released_count = 0

    grid_shape = (gridN, gridN, gridN)
    d_grid        = cuda.device_array(grid_shape, dtype=np.float32)
    d_grid_smooth = cuda.device_array(grid_shape, dtype=np.float32)

    sigma = math.sqrt(2.0 * CFG["D"] * dt)

    # smooth_grid_kernel bakes SMOOTH_TPB into its shared-memory halo tile, so
    # the launch block must match it. These were two independent hardcoded 8s;
    # desyncing them changes the smoothing silently, with no error raised.
    tpb3 = (SMOOTH_TPB, SMOOTH_TPB, SMOOTH_TPB)
    b3 = ((gridN + tpb3[0] - 1) // tpb3[0],
          (gridN + tpb3[1] - 1) // tpb3[1],
          (gridN + tpb3[2] - 1) // tpb3[2])
    n1 = 128

    # Concentration field the solid surface "sees".
    # Starts at zero, then carried forward from the GPU smoothing step.
    zero_grid_kernel[b3, tpb3](d_grid)
    zero_grid_kernel[b3, tpb3](d_grid_smooth)

    d_solid  = cuda.to_device(solid_pos.astype(np.float32))
    d_surf   = cuda.to_device(surface_idx)
    n_surf   = int(surface_idx.size)
    d_surf_C = cuda.device_array(max(n_surf, 1), dtype=np.float32)
    blk_surf = (n_surf + n1 - 1) // n1

    cap = max(Np, 1)
    d_pos = cuda.device_array((cap, 3), dtype=np.float32)
    d_alt = cuda.device_array((cap, 3), dtype=np.float32)
    d_probe_C    = None
    d_probe_site = None
    n = 0

    def ensure_capacity(need):
        nonlocal cap, d_pos, d_alt
        if need <= cap:
            return
        newcap = max(need, cap * 2)
        nd = cuda.device_array((newcap, 3), dtype=np.float32)
        na = cuda.device_array((newcap, 3), dtype=np.float32)
        if n > 0:
            copy_rows_kernel[(n + n1 - 1) // n1, n1](d_pos, nd, n)
            cuda.synchronize()
        d_pos, d_alt, cap = nd, na, newcap

    for step_idx in range(n_steps):
        # 1) Detachment decisions on CPU from the carried-over field.
        #    The surface cell a lattice site occupies also holds any solute
        #    released from that same cell, so its raw bin count is dominated by
        #    the solid itself and reads saturated before anything dissolves.
        #    SOLID_CELL_BASELINE is the mean smoothed count the un-dissolved
        #    lattice contributes to a surface cell; subtracting it leaves the
        #    dissolved concentration that Cs is meant to be compared against.
        if n_surf > 0:
            gather_surface_kernel[blk_surf, n1](
                d_solid, d_surf, n_surf, d_grid_smooth,
                grid_dx, gridN, d_surf_C)
            C_surf = np.maximum(
                d_surf_C.copy_to_host() - SOLID_CELL_BASELINE, 0.0
            ) / (grid_dx ** 3)
        else:
            C_surf = np.zeros(0, dtype=np.float32)
        detach_idxs = detach_step(surface_idx, C_surf, state)
        if detach_idxs.size > 0:
            m = int(detach_idxs.size)
            ensure_capacity(n + m)
            d_new = cuda.to_device(solid_pos[detach_idxs].astype(np.float32))
            append_kernel[(m + n1 - 1) // n1, n1](d_pos, n, d_new, m)
            n += m
            released_count += m
            state[detach_idxs] = 1

        # 2) Zero the GPU grid for this step's binning
        zero_grid_kernel[b3, tpb3](d_grid)

        # 3) Advance dissolved particles on GPU
        if n > 0:
            if rng_states is None or rng_states.size < n:
                rng_size = 256
                while rng_size < n:
                    rng_size *= 2
                rng_states = create_xoroshiro128p_states(
                    rng_size, seed=int(rng.integers(1, 1 << 30)))

            blocks = (n + n1 - 1) // n1

            brownian_step_kernel[blocks, n1](d_pos, n, rng_states, sigma, box)
            bin_positions_kernel[blocks, n1](d_pos, n, d_grid, gridN, grid_dx)
            smooth_grid_kernel[b3, tpb3](d_grid, d_grid_smooth, gridN)

            # 4) Re-adsorption probe: cell concentration + nearest solid site
            if d_probe_C is None or d_probe_C.size < n:
                d_probe_C    = cuda.device_array(n, dtype=np.float32)
                d_probe_site = cuda.device_array(n, dtype=np.int32)
            probe_kernel[blocks, n1](
                d_pos, n, d_grid_smooth, grid_dx, gridN,
                center_offset, spacing, N_x, N_y, N_z,
                d_probe_C, d_probe_site)
            # Same subtraction as C_surf: dissolved concentration only, so that
            # the re-adsorption test against Cs is not biased by the lattice.
            C_loc = np.maximum(
                d_probe_C.copy_to_host()[:n] - SOLID_CELL_BASELINE, 0.0
            ) / (grid_dx ** 3)
            keep = reattach_mask(C_loc, d_probe_site.copy_to_host()[:n], n)
            if keep is not None:
                keep_idx = np.flatnonzero(keep).astype(np.int32)
                n = int(keep_idx.size)
                if n > 0:
                    d_kidx = cuda.to_device(keep_idx)
                    compact_kernel[(n + n1 - 1) // n1, n1](
                        d_pos, d_alt, d_kidx, n)
                    d_pos, d_alt = d_alt, d_pos
        else:
            # Nothing in solution -> the field resets to zero
            zero_grid_kernel[b3, tpb3](d_grid_smooth)

        times.append(step_idx * dt)
        released.append(released_count / float(Np))

    grid_final = np.maximum(
        d_grid_smooth.copy_to_host() - SOLID_CELL_BASELINE, 0.0
    ) / (grid_dx ** 3)
    dissolved_host = d_pos.copy_to_host()[:n].copy()

    return times, released, state, dissolved_host, grid_final


# ============================================================
#  Driver — callable from main.py or from run.py
# ============================================================

def main_driver(solute_key="nacl", solvent_key="water", T_K=298.15,
                show_plot=True, save_path="dissolution_result.png"):
    """Configure the system, run the simulation, print stats, save & show plot."""
    list_options()
    configure_system(solute_key, solvent_key, T_K)

    print(f"\nRunning: {CFG['solute_name']} in {CFG['solvent_name']} at T = {T_K} K")
    print(f"  k_ref    = {CFG['k_ref']:.3e} / step")
    print(f"  Ea       = {CFG['Ea']/1000:.1f} kJ/mol")
    print(f"  Cs       = {CFG['Cs']:.4g}")
    print(f"  D (sim)  = {CFG['D']:.3e}")
    print(f"  k_attach = {CFG['k_attach']:.3e}")

    t0 = time.time()
    times, released, state_final, dissolved_final, grid_final = run_simulation()
    t1 = time.time()

    print(f"\nSimulation completed. Elapsed: {t1 - t0:.2f} s")
    print(f"Final dissolved fraction = {released[-1]:.4f} "
          f"({len(dissolved_final)} particles in solution)")

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')

    ax1.plot(times, released, '-o', markersize=3)
    ax1.set_xlabel('Time (step)')
    ax1.set_ylabel('Dissolved fraction')
    ax1.set_title(f"{CFG['solute_name']} in {CFG['solvent_name']}  (T = {T_K} K)")

    rem_mask = (state_final == 0)
    if rem_mask.sum() > 0:
        ax2.scatter(solid_pos[rem_mask, 0], solid_pos[rem_mask, 1], solid_pos[rem_mask, 2],
                    c=[solid_color], s=30)
    if dissolved_final.shape[0] > 0:
        ax2.scatter(dissolved_final[:, 0], dissolved_final[:, 1], dissolved_final[:, 2],
                    c=[dissolved_color], s=6)
    ax2.set_xlim(0, box); ax2.set_ylim(0, box); ax2.set_zlim(0, box)
    ax2.set_title('Final state (solid blue, dissolved red)')

    plt.tight_layout()

    try:
        fig.savefig(save_path, dpi=140)
        print(f"Saved figure -> {save_path}")
    except Exception as e:
        print(f"[warning] could not save figure: {e}")

    if show_plot:
        try:
            plt.show()
        except Exception as e:
            print(f"[warning] plt.show() failed (headless backend?): {e}")

    return times, released, state_final, dissolved_final, grid_final


# ============================================================
#  Entry point
# ============================================================

if __name__ == "__main__":
    # Pick any solute/solvent pair from the libraries above.
    main_driver(
        solute_key="nacl",
        solvent_key="water",
        T_K=298.15,
        show_plot=True,
        save_path="dissolution_result.png",
    )