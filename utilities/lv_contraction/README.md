# Left ventricle contraction coupled to a 0D circulation

Pipeline to simulate the contraction of a left ventricle (LV) segmented from
images, from end-diastole to end-systole and into relaxation, with
svMultiPhysics coupled to a 0D model of the circulation (svZeroDSolver). It can
also add a synthetic infarct.

The output is the displacement of every node of the mesh over time. It was set
up to provide ground-truth motion for testing image-based motion tracking: the
motion pattern is realistic, but the amplitudes are smaller than in vivo (see
[Limitations](#limitations)).

One config file per ventricle; the rest is automatic. For each ventricle,
`prepare_case.py`:
1. copies the mesh into a self-contained case folder,
2. generates the fiber directions with the single-ventricle Bayer method
   (`utilities/fiber_generation`, `FibGenBayerSingle`),
3. builds the cap that closes the cavity at the base (used to compute the
   cavity volume and flow),
4. optionally marks an infarct region,
5. writes the active stress curve, the svZeroDSolver configuration and the
   svMultiPhysics input file.

`postprocess.py` then computes the pressure-volume (PV) loop, EDV, ESV, EF and
peak pressure, and writes a file to open the results in ParaView.

## Requirements

- svMultiPhysics built from this branch:
  ```bash
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j 8
  ```
  The executable is `build/svMultiPhysics-build/bin/svmultiphysics`.
- svZeroDSolver (https://github.com/SimVascular/svZeroDSolver) built from
  source, for the coupling library
  `build/src/interface/libsvzero_interface.dylib` (`.so` on Linux).
- Python 3 with `numpy`, `scipy`, `pyvista` and `matplotlib`.
- MPI (`mpiexec`) to run in parallel.
- ParaView, to look at the results.

## Input mesh

For each ventricle:
- a linear tetrahedral volume mesh (`.vtu`) with `GlobalNodeID` and
  `GlobalElementID`,
- the endocardium, epicardium and base surfaces (`.vtp`) with `GlobalNodeID`
  and `GlobalElementID`, together covering the whole boundary of the mesh.
  The base is the flat cut of the wall at the top of the ventricle.

The endocardium must have exactly one open rim (where it meets the base). The
rim does not need to be planar: the cap is built from the rim nodes only.

The geometry is taken as the end-diastolic state (see
[Limitations](#limitations)). Meshes with very flat (sliver) tetrahedra can
slow down or stop the solver; if a run stalls, check the mesh quality first.

## Usage

1. Copy `case_config_PSTRAIN4.json` and edit it for the new ventricle: case
   name, output folder, mesh paths (relative to the config file, or absolute),
   `scale_factor` (to convert the mesh coordinates to metres: 1.0 if the mesh
   is in metres, 0.001 if it is in mm), paths to the executables and heart
   rate.

2. Prepare the case:
   ```bash
   python prepare_case.py case_config_<name>.json
   ```
   It prints the cavity volume of the mesh: check that it is close to the
   end-diastolic volume measured on the images. Use `--output-dir` to choose
   the case folder without editing the config, and `--skip-fibers` to reuse the
   fibers already in the case folder.

3. Run the simulation from inside the case folder:
   ```bash
   cd <case folder>
   mpiexec -n 5 <path to>/svmultiphysics solver.xml > run.log 2>&1
   ```
   With the default settings, this takes about one hour on 5 cores of a laptop
   for a mesh of about 23,000 nodes (93,000 tetrahedra). To stop a run cleanly, create a file named
   `STOP_SIM` in the case folder.

4. Postprocess:
   ```bash
   python postprocess.py <case folder> --check-3d-volume
   ```
   This prints EDV, ESV, EF and peak LV pressure, and writes:
   - `results/pv_loop.png` and `results/pv_loop.csv`: PV loop and pressures
     over time,
   - `results/results.pvd`: all the saved results as one time series.

   With `--check-3d-volume`, the cavity volume of the deformed mesh is compared
   with the volume from the 0D model (they should agree to a fraction of a mL).

## Results

Open `results/results.pvd` in ParaView. Time is in ms from end-diastole (the
mesh geometry); a result is saved every 10 ms (`timing.save_every_ms`).
End-systole is the time of minimum volume in `pv_loop.csv` (about 150-160 ms at
104 bpm). Each result has, at the nodes:

| Field | Meaning |
|---|---|
| `Displacement` | Displacement from the mesh geometry, **in metres** |
| `Velocity` | Velocity (m/ms) |
| `Jacobian` | Local volume ratio; about 1 everywhere (the wall is nearly incompressible) |
| `Cauchy_stress` | Stress (MPa) |
| `Fiber_direction`, `Fiber_stretch` | Deformed fiber direction and fiber stretch |

To see the motion, apply the *Warp By Vector* filter with `Displacement`
(scale factor 1) and play the animation. The point coordinates of the results
are those of the mesh scaled to metres; the deformed position of a node is its
coordinate plus `Displacement`.

## Infarct

Set `infarct.enabled` to `true` in the config. The elements whose centre lies
inside a sphere (`infarct.center`, `infarct.radius`) become infarct (scar):
they do not contract, and their passive stiffness is multiplied by
`infarct.stiffness_factor` (10 by default). The center and radius are in the
units of the mesh file (before `scale_factor`); to choose them, open the mesh in
ParaView and read the coordinates of a point on the wall (for example with
*Hover Points On*).

`prepare_case.py` prints the infarct volume as a percentage of the wall and
writes `mesh/domains.vtu`, with the cell array `DOMAIN_ID` (1: healthy, 2:
infarct), to check the region in ParaView. Everything else is identical to the
healthy case, so a healthy and an infarct case of the same ventricle can be
compared directly.

On the reference pig ventricle (PSTRAIN4) with an infarct of 12 % of the wall,
the wall thickens by about 18 % away from the infarct and thins by about 5 %
inside it (the scar is stretched by the cavity pressure), and EF drops from 28 %
to 24 %.

## Main parameters

All in the case config:

| Parameter | Meaning |
|---|---|
| `active_stress.peak_kPa` | Peak contraction strength. The parameter to adjust for a new ventricle if EF is far from the target; EF responds weakly (on PSTRAIN4, 250 kPa: EF 23 %, 450 kPa: EF 28 %) |
| `active_stress.cross_fiber_fraction` | Share of the contraction across the fibers, in the wall plane (default 0). 0.25 gave one more EF point and lost most of the twist |
| `infarct.*` | Synthetic infarct, see above |
| `timing.heart_rate_bpm` | Heart rate; sets the duration of systole |
| `timing.n_cycles` | Fraction (or number) of cardiac cycles to simulate; 0.5 covers systole and early relaxation |
| `fibers.*` | Helix (`alpha`) and transverse (`beta`) fiber angles at the endo and epi, in degrees |
| `material.*` | Passive stiffness (Holzapfel-Ogden, kPa) and volumetric penalty |
| `boundary_conditions.*` | Springs (MPa/m) and dampers (MPa*ms/m) that hold the ventricle, see [Model](#model) |
| `circulation.*` | Atrial pressure, arterial resistances, compliances and inertance, initial arterial pressure |

Change the boundary conditions, material and solver settings only with care:
the defaults are the result of the tests described below.

## Model

| Component | Choice |
|---|---|
| Reference geometry | The mesh geometry, taken as stress-free |
| Material | Holzapfel-Ogden (modified anisotropy), nearly incompressible (penalty 10 MPa) |
| Fibers | Rule-based, helix angle varying linearly from +60 deg (endo) to -60 deg (epi) |
| Contraction | Prescribed active tension along the fibers, uniform in space, sin^2 pulse in time |
| Pericardium | Spring and damper on the epicardium, acting only along the surface normal (100 MPa/m) |
| Base | Spring and damper on the base in all directions (1 MPa/m) |
| Circulation | Open loop of Brown et al. (2024): atrial pressure, mitral and aortic valves, proximal CRL and distal RC |
| Coupling | Implicit 3D-0D coupling; the 0D pressure acts on the endocardium |
| Elements and solver | Linear tetrahedra, `struct` equation, fsils linear solver |

The simulation is in units of m, ms, kg, with pressures and stresses in MPa.
The case config uses clinical units (mmHg, mL, s, kPa) and the conversion is
done by `prepare_case.py`.

The setup follows the svMultiPhysics test case
`tests/cases/electromechanics/ventricle_sv0D`, with a prescribed active stress
instead of electrophysiology.

References:
- Brown et al. (2024), A modular framework for implicit 3D-0D coupling in
  cardiac mechanics, CMAME 421:116764.
- Holzapfel & Ogden (2009), Phil. Trans. R. Soc. A 367:3445-3475.
- Bayer et al. (2012), Ann. Biomed. Eng. 40:2243-2254.
- Pfaller et al. (2019), The importance of the pericardium for cardiac
  biomechanics, Biomech. Model. Mechanobiol. 18:503-529.

## Limitations

Reference values below are for the pig ventricle PSTRAIN4 with the default
settings.

- **The motion is smaller than in vivo.** EF is about 28 % (healthy pigs:
  about 50-60 %), the base moves about 6 mm towards the apex (in vivo about
  10 mm) and the wall thickens by about 20 % (in vivo about 30-40 %). The
  motion pattern is realistic: the apex stays nearly still, the base moves
  towards it, the wall thickens and twists (about 18 deg between base and
  apex).
- **Only the LV is modelled, and springs hold it in place.** In the body the
  atria and great vessels pull on the base; here they are missing, and the cap
  that closes the cavity does not transmit the cavity pressure. The net force
  of the cavity pressure (about 14 N at peak) pushes the ventricle towards the
  apex, and only the springs resist it. The stiff pericardial spring keeps the
  apex still; the motion near the base is the least realistic part. The proper
  fix is a load on the base that balances the cavity pressure (as in Regazzoni
  et al. 2022), which svMultiPhysics does not have yet.
- **The mesh geometry is taken as stress-free.** In reality the ventricle is
  already loaded by the end-diastolic pressure. Here the simulation starts from
  the image geometry with zero pressure (see prestress below).
- **The infarct is synthetic**: a sphere with a sharp border, no border zone
  and no remodelling of the wall.
- **The fibers are rule-based**, not measured.
- **The contraction is uniform in space and time**: the whole wall contracts
  at once, with no electrical activation sequence.
- **The wall volume is not exactly conserved**: it shrinks by about 0.5 %,
  because of the linear tetrahedra with a moderate volumetric penalty (a
  larger penalty makes linear tetrahedra too stiff).
- **The circulation parameters are human values** (Brown et al. 2024); the
  peak LV pressure is about 75 mmHg, lower than in pigs.

## What was tried

So that these do not need to be tried again:

- **Prestress** (`use_prestress`): computes the stress at end-diastole so
  that the image geometry is in equilibrium with the end-diastolic pressure.
  It is implemented (`templates/prestress.xml`, `extract_prestress.py`) but
  converges too slowly with the fsils linear solver to be practical for now.
- **Quadratic tetrahedra** (`mesh.element_order: 2`, `make_quadratic_mesh.py`):
  they would allow a larger volumetric penalty without locking, but with fsils
  the linear solver became very slow at a 10 MPa penalty (stalls on a cluster).
- **PETSc linear solver** (`linear_solver: petsc`, `run_case_petsc.slurm`):
  svMultiPhysics' PETSc interface leaves the cap out of the 3D-0D coupling
  term (`petsc_impl.cpp`; the Trilinos interface includes it), so Newton
  stalls. Use fsils until that is fixed.
- **Pericardial spring stiffness**: with 10 MPa/m the whole ventricle slid
  about 4 mm towards the apex and the base barely moved relative to the apex
  (1.3 mm). A stiffness decreasing from the apex to zero towards the base
  (`epi_stiffness_free_height`, as in Strocchi et al. 2020) made it worse in a
  model of the LV alone (the ventricle got longer during systole), because it
  removes support against the cavity pressure. A uniform 100 MPa/m (as in the
  svMultiPhysics test case) gives the realistic pattern.
- **Base spring**: 0.1 MPa/m left the rim of the base floppy; 1 MPa/m is
  stable.
- **Contraction across the fibers** (`cross_fiber_fraction` 0.25): EF +1
  point, twist from 18 to 8 deg. Left off.
- **Stronger contraction**: 250 to 450 kPa raised EF from 23 % to 28 %; EF is
  also limited by the circulation model.

## Running on a cluster

`build_svzerod.slurm` builds svZeroDSolver, and `run_case_petsc.slurm` runs a
prepared case (it sets the path of the svZeroDSolver library on the cluster;
with `linear_solver: fsils` in the config, it runs with fsils even if
svMultiPhysics is built with PETSc). Edit the module and path lines at the top
for your cluster. `--restart` continues from the last restart file, with the
same number of MPI ranks; restarts reset the 0D model, so use them to debug,
not for results.

## Solver notes

- The residuals are small in absolute terms in these units: keep the linear
  solver `Absolute_tolerance` in `templates/solver.xml` small (`1e-15`),
  otherwise the linear solver stops early and Newton does not converge.
- `linear_solver_tolerance` (default 1e-6) is the relative tolerance of the
  linear solver; Newton does not need exact linear solves.
