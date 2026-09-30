# AGENTS.md

Monte-Carlo dissolution simulator (Nanjundiah-style) of a solid lattice in a
box, with Brownian diffusion of released particles and re-adsorption at empty
surface sites. Particle motion, concentration binning, and grid smoothing are
Numba CUDA kernels; the detachment/adsorption decisions are CPU-side.

Two files, flat layout, no package. `main.py` = library + engine + driver.
`run.py` = interactive terminal picker. No tests, no CI, no lint config, and git
has **zero commits**.

## Run it

A ready-to-use virtualenv lives in `.venv/` (created on disk, not in `/tmp`):

```bash
cd /home/ilgaz/Dissolution_CUDA
source .venv/bin/activate
python main.py     # or: python run.py
```

`activate` also runs `scripts/setup_cuda_home.sh`, which assembles the CUDA
toolkit tree and exports `CUDA_HOME` / `LD_LIBRARY_PATH`. Verified end-to-end
from a bare shell: 600 steps in ~2 s, final dissolved fraction 0.4660.

To rebuild from scratch (no `sudo`; the box is PEP 668 externally-managed and
has no `ensurepip`, so the venv is created `--without-pip` and then bootstrapped):

```bash
cd /home/ilgaz/Dissolution_CUDA
python3 -m venv --without-pip .venv
curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
.venv/bin/python -m pip install -r requirements.txt
source .venv/bin/activate
```

Things that will bite you here, all verified:

- Only `python3` exists; there is no `python`.
- **Do not let pip resolve numpy freely.** `numba-cuda` 0.30 calls
  `np.row_stack`, removed in numpy 2.5, and declares no upper bound — so a fresh
  `pip install` produces a combo that dies at import with
  `AttributeError: module 'numpy' has no attribute 'row_stack'`. `requirements.txt`
  pins `numpy<2.5` for this reason; don't relax it.
- There is no `nvcc` and no system CUDA toolkit, but the `nvidia-cuda-nvcc-cu12`
  wheel supplies `libnvvm.so`, which is all numba's JIT needs.
- Those wheels ship an **unversioned** `libnvvm.so`, while numba's finder
  (`numba.misc.findlib`) only matches `lib<name>.so.<digits>`. Without the
  versioned symlink that `setup_cuda_home.sh` creates, every kernel launch fails
  with `NvvmSupportError: libNVVM cannot be found`.
- `run.py` does `from main import ...`, so always run from the repo root.

`run.py` has no flag parsing — it prompts via `input()`. It needs at least one
line of stdin, so piping answers works and no TTY is required:
`printf '8\n1\n\nn\n\ny\n' | python run.py` runs CaCO3 in water and saves
`dissolution_result.png` (answers are: solute, solvent, temperature, show-plot,
save-path, start — blank accepts each default). Only *empty* stdin fails, with
`EOFError` at `run.py:87`. For anything scriptable, skip the prompts and call
`main.main_driver(...)` directly.

`state` (`main.py:207`) and `rng` (`main.py:178`) are reset at the top of
`run_simulation`, so consecutive runs of the same config are bit-identical
(both 0.465986) — run them in one process freely when testing. Do **not** rely
on that reset for anything else: `state` and `rng` are still module globals, and
`reattach_mask` reads `state` and `surface_mask` directly rather than taking
them as arguments, so mutating them mid-loop corrupts the run.

`CFG` is a global rewritten by `configure_system`; `run_simulation` raises
`RuntimeError` if it is empty, and is still not reentrant or thread-safe.

## Headless machines

`main.py` probes for a backend that will actually import (`_pick_backend`)
rather than calling `matplotlib.use("TkAgg")` in a `try/except`. This matters
because `use()` is lazy — it does not import the backend until the first figure
is created — so the old guard never fired and the crash surfaced much later at
the unguarded `plt.figure()`, *after* a full successful simulation. On Linux it
also checks for `DISPLAY`/`WAYLAND_DISPLAY`, because importing the Tk backend
succeeds with no display and then dies with `TclError` anyway. `MPLBACKEND` is
honored as an override.

`main_driver` now completes and writes its PNG on a headless box. If you
regress the backend probe, expect `ModuleNotFoundError: No module named
'tkinter'` or `TclError` at `plt.figure()` after the run has already finished —
not a broken simulation.

## Module-level mutable state

`state` (`main.py:207`) marks sites already dissolved and used to **never** be
reset, so a second `run_simulation()` in the same process started from a
partially dissolved solid: `detach_step` refuses any site with `state != 0`,
which dropped a repeat run from 0.4660 to 0.0434. `rng` (`main.py:178`) had the
same problem, advancing between calls. Both are now reset at the top of
`run_simulation` from `SEED`, making repeat runs bit-identical (0.465986 both).

If you add code that mutates `state` or `rng` between steps, you reintroduce
the bug: `reattach_mask` reads `state` and `surface_mask` as globals rather than
taking them as arguments, so mid-loop mutation silently corrupts the run.

`CFG` is a global rewritten by `configure_system`; `run_simulation` raises
`RuntimeError` if it is empty, and is still not reentrant or thread-safe.

## Silent-corruption hazards

- `SMOOTH_TPB`/`SMOOTH_SIDE` (`main.py:233`) are compile-time constants baked
  into `smooth_grid_kernel`'s shared-memory halo tile, and the launch block is
  now derived from them (`tpb3 = (SMOOTH_TPB, SMOOTH_TPB, SMOOTH_TPB)`). They
  used to be two independent hardcoded 8s; desyncing them silently produced a
  different field with no error (measured 0.4770 vs 0.4660). Keep the coupling
  if you retune the tile. Note `SMOOTH_TPB`/`SMOOTH_SIDE` are frozen into the
  kernel at first JIT, so retuning it only takes effect in a **fresh process** —
  editing the global mid-run re-creates the desync via numba's compile cache.
- Concentration is `count / grid_dx**3` (`main.py:514`, `main.py:553`), so
  `gridN` and `box` are **physics parameters, not just resolution knobs** —
  they scale absolute C against `CFG["Cs"]` and therefore the dissolution rate.
- The GPU RNG is reseeded from `rng.integers` whenever the particle count
  outgrows `rng_states` (`main.py:532-537`), so the Brownian trajectory depends
  on how the device buffer happened to grow. Don't treat it as a fixed stream.

## Simulation structure worth knowing before editing

- Per step, the order is: detach (CPU, from last step's field) → zero grid →
  Brownian move + atomic binning → 3x3x3 shared-mem smoothing → re-adsorption
  probe (CPU). Detachment deliberately reads the **previous** step's smoothed
  field, so the two grids ping-pong with a one-step lag. Reordering breaks it.
- Two device arrays: `d_grid` = raw counts for this step, `d_grid_smooth` =
  smoothed field carried to the next step. Don't collapse them.
- `d_pos`/`d_alt` double-buffer; compaction swaps them (`main.py:562`). Capacity
  growth reallocates both via `ensure_capacity`.
- Only host syncs are the per-step copies of `d_surf_C` and the probe arrays.
  The GPU work is not the bottleneck; the round-trips are.
- Libraries (`SOLVENTS`, `SOLUTES`) hold only reference values. Temperature
  enters via Arrhenius (`k`) and Van't Hoff (`Cs`) around `T_REF = 298.15`, plus
  `D = D_BASE_SIM * D_scale * (T/T_REF)`. `k_attach` is a hardcoded
  `5.0e-5 * (D/D_BASE)` (`main.py:159`), not a library field.
- `nacl` + water @ 298.15 K is the default in `main.py`'s `__main__` block.
  `caco3` has negative `dH_diss_J_mol` (retrograde solubility), which is
  correct per geochemistry: calcite is less soluble as temperature rises.

## Known limitation: the saturation term is quantisation-limited

**Do not interpret small differences in dissolved fraction as solvent
chemistry.** The 48-run sweep (8 solutes × 6 solvents) gives a spread of only
~0.13–0.49 regardless of solvent, and that is a real limitation, not noise-free
signal. Root cause, measured:

- One dissolved particle raises a cell's concentration by
  `1 / (grid_dx**3 * 27)` = **1.21** after the 3x3x3 smoothing. The field is
  effectively integer-counted, so `C_surf` can only take values 0, 1.21, 2.43…
- `Cs` for 5 of 8 solutes (ibuprofen 1e-4, CaCO3 1.3e-4, aspirin 0.0185,
  benzoic acid 0.028, caffeine 0.112) is **below that single-particle step**.
  For these, `min(C/Cs, 1)` is 0 or 1 and the interpolation term vanishes
  entirely — saturation becomes a binary flag set by one particle.
- Result: changing `Cs` across four orders of magnitude (ibuprofen is 12,000×
  more soluble in ethanol than water) moves the dissolved fraction by <0.02.
  Only the three high-`Cs` salts (NaCl 5.44, KCl 4.07, sucrose 5.87) sit in a
  range where the field can resolve saturation at all.

Fixing this properly needs a finer concentration representation (partial
occupancy / sub-count binning), not a tweak to `Cs` or `gridN`. Do not "fix" it
by tuning constants — it will look better and be no more physical.

`SOLID_CELL_BASELINE` exists because the surface cell a lattice site occupies
also holds any solute released from that cell, so its raw count reads saturated
before anything dissolves; it is derived from `gridN`/`spacing`, so it must stay
a derived value.

## Conventions

- Section banners: `# ==== Name ====` and `# ---- name ----`; kernels grouped
  under "GPU kernels", host logic under "CPU-side helpers". Keep this shape.
- The 1-D kernels use `n1 = 128` threads/block; the 3-D ones use 8x8x8.
- Geometry, time, and grid parameters are module-level constants
  (`N_x/N_y/N_z`, `spacing`, `box`, `dt`, `n_steps`, `gridN`) — not CLI args.
- `NumbaPerformanceWarning: Grid size 1 ... GPU under-utilization` spam during
  the first steps is expected: few particles means few blocks. Not an error.
- `dissolution_result.png` is a generated output, relative to CWD, and gets
  overwritten by any run from the repo root. `.gitignore` now covers it and
  `__pycache__/`.
