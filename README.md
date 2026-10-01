# Dissolution_CUDA

Monte-Carlo simulation of solid dissolution, following the Nanjundiah-style
lattice model: particles detach from a solid surface at a rate set by an
Arrhenius rate constant and local saturation, diffuse by Brownian motion, and
re-adsorb onto empty surface sites.

Particle motion, concentration binning, and grid smoothing are Numba CUDA
kernels; the detachment and re-adsorption decisions run on the CPU.

---
## Tested Machine Specs
Processor: 13th Gen Intel(R) Core(TM) i7-13700H (2.40 GHz)

RAM: 64 GB

Graphics Card: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)
## Quick start

A ready-to-use virtualenv is checked in at `.venv/`:

```bash
cd /home/ilgaz/Dissolution_CUDA
source .venv/bin/activate
python main.py     # library + engine + default driver
python run.py      # interactive terminal picker
```

`source` also runs `scripts/setup_cuda_home.sh`, which assembles a minimal CUDA
toolkit tree from the `nvidia-cuda-nvcc-cu12` wheel and exports `CUDA_HOME` /
`LD_LIBRARY_PATH`. No system CUDA install and no `sudo` are required.

Default run (NaCl in water, 298.15 K) takes ~2 s and writes
`dissolution_result.png`.

### Rebuilding the environment from scratch

```bash
python3 -m venv --without-pip .venv
curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
.venv/bin/python -m pip install -r requirements.txt
source .venv/bin/activate
```

Two traps, both verified on this machine:

- **Do not let pip resolve numpy freely.** `numba-cuda` 0.30 calls
  `np.row_stack`, removed in numpy 2.5, and declares no upper bound. A fresh
  `pip install` therefore produces a combination that dies at import with
  `AttributeError: module 'numpy' has no attribute 'row_stack'`. `requirements.txt`
  pins `numpy<2.5` for this reason.
- There is no `nvcc` and no system CUDA toolkit. The `nvidia-cuda-nvcc-cu12`
  wheel ships an **unversioned** `libnvvm.so`, while numba's `numba.misc.findlib`
  only matches `lib<name>.so.<digits>`. `setup_cuda_home.sh` creates the missing
  versioned symlink; without it every kernel launch fails with
  `NvvmSupportError: libNVVM cannot be found`.

### Scripting

`run.py` prompts via `input()` and needs at least one line of stdin, so piping
works without a TTY:

```bash
printf '8\n1\n\nn\n\ny\n' | python run.py
```

Answers are, in order: solute, solvent, temperature, show-plot, save-path,
start. Blank accepts each default. For anything scriptable, call
`main.main_driver(...)` directly rather than the prompts.

---

## Files

| File | Role |
| --- | --- |
| `main.py` | Library (`SOLUTES`, `SOLVENTS`), kernels, engine, plotting, driver |
| `run.py` | Interactive front end |
| `requirements.txt` | Pinned dependencies |
| `scripts/setup_cuda_home.sh` | Assembles the local CUDA toolkit tree |
| `AGENTS.md` | Detailed notes on structure, hazards, and known limitations |

---

## Physical model

Per step, the order is:

1. **Detach** (CPU) — read the *previous* step's smoothed field, apply
   `p = k · (1 − min(C/Cs, 1)) · dt`
2. **Zero** the bin grid
3. **Move + bin** (GPU) — Brownian step, then atomic scatter into `gridN³` cells
4. **Smooth** (GPU) — 3×3×3 box filter in shared memory
5. **Re-adsorb** (CPU) — probe local concentration, reattach to nearest empty site

Detachment deliberately reads the *previous* step's field, so the two grids
ping-pong with a one-step lag. Do not reorder this.

**Temperature** enters three ways, around `T_REF = 298.15 K`:

- `k` via Arrhenius
- `Cs` via Van 't Hoff
- `D = D_BASE_SIM · D_scale · (T/T_REF)`

**Solubility.** `Cs_sim` is the *aqueous* molar solubility. Each solvent carries a
`Cs_ratio` dict giving the ratio of its solubility to water's, so
`Cs_eff = Cs_sim · Cs_ratio[solute]`. See the data-quality section below for how
well each of these is sourced.

**Concentration** is `C = count / grid_dx³`, where a cell holds an integer count
of particles and `grid_dx = box / gridN = 20/64 = 0.3125`.

**`SOLID_CELL_BASELINE`.** The grid cell that a lattice site occupies also holds
any solute released from that same cell, so its raw count is dominated by the
solid and reads saturated before anything dissolves. Since every surface site
occupies its own cell and the 3×3×3 filter spreads that count over 27 cells, the
lattice contributes a mean of `1/27` to the cell reading. This is subtracted at
every concentration read site, and is derived from `gridN`/`spacing` rather than
hardcoded.

### A note on units

`C = count / grid_dx³` produces a number in *simulated* units, not mol/L. The
simulation assumes one sim length unit corresponds to one decimetre, which makes
the cell volume `0.0305 L` and places a single particle's contribution
numerically near a molar concentration — which is what lets real `Cs` values be
compared against the field at all. This is an assumption, not a derivation, and
it is the weakest link in the unit handling.

---

## Scientific literature

Everything below was used to set or sanity-check a number in `main.py`.
**Confidence is stated per value, because they are not equally well sourced.**

### Primary sources

| # | Source | Used for |
| --- | --- | --- |
| 1 | Pinho, S. P.; Macedo, E. A. "Solubility of NaCl, NaBr, and KCl in Water, Methanol, Ethanol, and Their Mixed Solvents." *J. Chem. Eng. Data* **2005**, *50* (1), 29–32. doi:[10.1021/je049922y](https://doi.org/10.1021/je049922y) | NaCl and KCl in water/methanol/ethanol, 298.15–348.15 K. Primary gravimetric measurements; ~530 citations. |
| 2 | Tully, G.; Hou, G.; Glennon, B. "Solubility of Benzoic Acid and Aspirin in Pure Solvents Using Focused Beam Reflective Measurement." *J. Chem. Eng. Data* **2016**, *61* (1), 594–601. doi:[10.1021/acs.jced.5b00746](https://doi.org/10.1021/acs.jced.5b00746) | Benzoic acid and aspirin in neat solvents, 275–327 K. The single best source for this model — covers both solutes across alcohol/ketone solvents. |
| 3 | "Solubility of Acetylsalicylic Acid in Ethanol, Acetone, Propylene Glycol, and 2-Propanol." *J. Chem. Eng. Data* **2007**, *53* (1), 256. doi:[10.1021/je7005693](https://doi.org/10.1021/je7005693) | Aspirin solvent ranking. Reports solubility highest in acetone over the whole 298–326 K range. |
| 4 | Acree, W. E. "IUPAC-NIST Solubility Data Series. 102. Solubility of Nonsteroidal Anti-inflammatory Drugs (NSAIDs) in Neat Organic Solvents and Organic Solvent Mixtures." *J. Phys. Chem. Ref. Data* **2014**, *43*, 023102. doi:[10.1063/1.4870081](https://doi.org/10.1063/1.4870081) | Curated compilation of 33 NSAIDs (incl. ibuprofen) in neat organic solvents, 1980–2014. |
| 5 | Garzón, L. C.; Martínez, F. "Temperature dependence of solubility for ibuprofen in some organic and aqueous solvents." *J. Solution Chem.* **2004**, *33* (11), 1379–1395. doi:[10.1007/s10953-004-9548-0](https://doi.org/10.1007/s10953-004-9548-0) | Ibuprofen in organic vs aqueous solvents; derived ΔH/ΔG/ΔS of solution. |
| 6 | "The solubility product controls the rate of calcite dissolution in pure water and seawater." *Chem. Sci.* **2024**. doi:[10.1039/D3SC04063A](https://doi.org/10.1039/D3SC04063A) | Calcite `Ksp = 3.3 × 10⁻⁹ M²` at 298 K, treated as well established in near-pure water. |
| 7 | "Measurement solubility of Acetylsalicylic Acid in water and alcohols." **1998**. | Aspirin in water, 1-octanol, ethanol, methanol, ethylene glycol, 298–330 K. |
| 8 | "Influence of Sodium Salicylate on Self-Aggregation and Caffeine Solubility in Water." *ACS Omega*. PMC9697389 | Measured caffeine solubility in water: **20.71 g / 1000 g H₂O at 298.15 K**. |

### Reference compilations

| # | Source | Used for |
| --- | --- | --- |
| 9 | PubChem (NIH), via the AQUASOL database of Yalkowsky, S. H.; Dannenfelser, R. M. | Ibuprofen **21 mg/L at 25 °C**; aspirin "1 g in 300 mL water at 25 °C" |
| 10 | ILO-WHO International Chemical Safety Cards, via PubChem | Sucrose **200 g/100 mL at 25 °C**; also "1 g dissolves in 0.5 mL water, 170 mL alcohol, ~100 mL methanol" |
| 11 | CRC-style aqueous solubility tables (25 °C, g/100 mL) | NaCl 36, KCl 35, sucrose ~201 |
| 12 | Wikipedia infoboxes, which aggregate the above | NaCl 360 g/L; KCl 35 g/100 mL; benzoic acid 3.44 g/L at 25 °C; caffeine; CaCO₃ 0.013 g/L at 25 °C |
| 13 | Cayman Chemical product sheet (ASA 70260) | Aspirin: ethanol 80 mg/mL, DMSO 41 mg/mL, PBS 2.7 mg/mL |
| 14 | *Introduction to Geochemistry* (carbonate equilibria) and Wikipedia | Calcite solubility **decreases** as temperature rises |

### Concentration rationale (not measured data)

- `D_scale` is a *Stokes–Einstein* proxy, `D ∝ 1/viscosity`, not a measured
  diffusion coefficient. Values track solvent viscosity well but were not fitted
  against experimental `D` data.
- The `1e-3` `Cs_ratio` entries for the salts in acetone/hexane are deliberate
  order-of-magnitude placeholders meaning "negligibly soluble". They are not
  measurements. They only need to keep `Cs` well below the single-particle
  concentration step (see the limitation below).

---

## Data quality

Aqueous values, with the conversion actually performed:

| Solute | Value in code | Literature | Conversion | Agreement |
| --- | --- | --- | --- | --- |
| CaCO₃ | 1.3 × 10⁻⁴ M | 0.013 g/L (25 °C) | / 100.09 | **exact** |
| Benzoic acid | 0.028 M | 3.44 g/L (25 °C) | / 122.12 | **exact** |
| Ibuprofen | 1.0 × 10⁻⁴ M | 21 mg/L (25 °C) | / 206.28 | **exact** |
| Aspirin | 0.0185 M | 3.3 g/L (25 °C) | / 180.16 | **exact** |
| Sucrose | 5.87 M | 200 g/100 mL (25 °C) | / 342.30 | **exact** |
| Caffeine | 0.112 M | 20.71 g/kg H₂O (298.15 K) | / 194.19 | ~5% high |
| NaCl | 5.44 M | 360 g/L (25 °C) | / 58.44 → **6.16 M** | **12% low** |
| KCl | 4.07 M | 350 g/L (25 °C) | / 74.55 → **4.69 M** | **13% low** |

### Open discrepancies

These are unresolved. They are listed rather than quietly patched.

1. **NaCl and KCl aqueous solubility are too low.** The code has 5.44 and 4.07 M;
   the standard tabulated values are ~6.16 and ~4.69 M. The code values appear to
   have come from an unsourced research summary. Correcting them would raise `Cs`
   for both salts, moving them further into the range where the saturation term
   can actually resolve anything (see below) — so this is worth doing, but it
   changes results and was left for an explicit decision.

2. **Sucrose in alcohols is ~2× too high.** The ICS line "1 g dissolves in 170 mL
   alcohol, ~100 mL methanol" gives 0.017 M in ethanol and 0.029 M in methanol
   (ratios 0.0029 and 0.005), versus 0.034 and 0.059 M in the code (ratios 0.0058
   and 0.010). Same order of magnitude, wrong by a factor of two.

3. **Aspirin data spread ~2.5×.** Cayman lists 80 mg/mL in ethanol (0.44 M);
   other sources give 200 mg/mL (1.11 M). The code uses 1.11 M, which matches the
   higher figure. DMSO at 41 mg/mL (0.228 M) implies a ratio of ~12, versus 30 in
   the code.

4. **NaCl in DMSO is unsourced.** No verifiable figure was found. See below.

5. **CaCO₃ `Cs` vs `Ksp`.** The code uses the *measured* solubility
   (1.3 × 10⁻⁴ M, from 0.013 g/L). Naive `Ksp = 4s³` with `Ksp = 3.3 × 10⁻⁹` would
   instead give 9.4 × 10⁻⁴ M. Both numbers are "correct" but measure different
   things: the measured value reflects carbonate protonation and air-equilibrated
   CO₂, which the simulation does not model. The measured value is the right
   choice here, but the discrepancy is real and unexplained.

### Citations that failed verification and were removed

An earlier draft of the `main.py` provenance comment cited two papers that
**could not be confirmed to exist**. They have been struck out rather than
retained:

- ~~*Jouyban et al., J. Chem. Eng. Data* 55 (2010) 5252 — ibuprofen in ethanol,
  1.2055 M.~~ No record of this paper could be found. The nearest real paper is
  Jouyban et al., *Chem. Pharm. Bull.* **58** (2), 219–224 (2010), on PEG 600 /
  ethanol / water mixtures, which reports a maximum ibuprofen solubility of
  3.2792 M — a different solvent system. The 1.2055 M ethanol figure in the code
  is therefore **unverified**; source [4] and [5] above should be consulted to
  replace it.
- ~~Unni et al., 1979 — NaCl in DMSO, 0.0775 M.~~ Repeated searches returned no
  such paper and no NaCl-in-DMSO solubility figure. The 0.0775 M value and its
  citation are both **unverified**. DMSO in the code is a placeholder.

A third citation, NTP 50782 for aspirin's insolubility in petroleum ether, is
plausible (it is a real NTP report on aspirin) but the specific solubility
statement was not confirmed. Treat the aspirin/hexane ratio as unverified.

---

## Known limitation: the saturation term is quantisation-limited

**Do not interpret small differences in dissolved fraction as solvent chemistry.**

The 48-combination sweep (8 solutes × 6 solvents) spans only ~0.13–0.49 dissolved
fraction, largely independent of solvent. This is a real structural limitation,
not noise.

The cause is arithmetic. One dissolved particle changes a cell's concentration by

```
1 / (grid_dx**3 * 27) = 1.214
```

because the 3×3×3 smoothing divides by 27. The field is effectively
integer-counted, so `C_surf` can only take the values 0, 1.21, 2.43, 3.64, …

For 5 of the 8 solutes, `Cs` is *below* that single-particle step:

| Solute | `Cs` | Particles needed to saturate one cell |
| --- | --- | --- |
| Ibuprofen | 1.0 × 10⁻⁴ | < 1 |
| CaCO₃ | 1.3 × 10⁻⁴ | < 1 |
| Aspirin | 0.0185 | < 1 |
| Benzoic acid | 0.028 | < 1 |
| Caffeine | 0.112 | < 1 |

For these, `min(C/Cs, 1)` is either 0 or 1 and the interpolation term vanishes
completely. Saturation degenerates into a binary flag set by whether one particle
happens to be in the cell. Measured directly: ibuprofen's surface cells are
**100% saturated in both water and ethanol**, so its detachment probability
becomes chemically independent.

The consequence is that a 12,000× solubility difference (ibuprofen in ethanol
vs water) moves the dissolved fraction by less than 0.02. Only the three
high-`Cs` salts — NaCl, KCl, sucrose — sit in a range where the field can
resolve saturation at all.

**Fixing this requires a finer concentration representation** (partial
occupancy, sub-count binning, or a continuous splat). It cannot be fixed by
adjusting `Cs`, `gridN`, or any other constant: that would produce a
better-looking and equally unphysical result.

---

## Reproducibility notes

- `state` and `rng` are reset from `SEED` at the top of `run_simulation`, so
  consecutive runs of the same configuration are bit-identical.
- `CFG` is a global rewritten by `configure_system`; `run_simulation` raises
  `RuntimeError` if it is empty. It is not reentrant or thread-safe.
- The GPU RNG is reseeded from `rng.integers` whenever the particle count
  outgrows the device RNG-state buffer, so the Brownian trajectory depends on how
  the buffer happened to grow. It is not a fixed stream.
- `gridN` and `box` are physics parameters, not resolution knobs: `C = count /
  grid_dx³` means they scale absolute `C` against `Cs` and therefore change the
  dissolution rate.
- `NumbaPerformanceWarning: Grid size 1 ... GPU under-utilization` during the
  first steps is expected — few particles means few blocks. Not an error.
- `dissolution_result.png` is a generated output, relative to the current
  directory, and is overwritten by any run from the repo root.
