#!/bin/bash
#Runs a case prepared by prepare_case.py with svMultiPhysics built with PETSc. Submit from inside the case folder:
#   cd <case folder> && sbatch <path to>/run_case_petsc.slurm
#The case can be prepared on another machine: this script writes the path of the svZeroDSolver coupling library
#on the cluster (built by build_svzerod.slurm) into solver.xml.
#
#Restart from the last restart file (results/stFile_last.bin, written every save_every_ms):
#   sbatch --nodes=<same> --ntasks=<same> <path to>/run_case_petsc.slurm --restart
#The restart file only works with the same number of MPI ranks as the run that wrote it (checked below).
#On restart svMultiPhysics restarts the 0D model from the initial conditions in svzerod_3Dcoupling.json and
#at 0D time zero, and overwrites results/svZeroD_data: the old file is kept as svZeroD_data.before_restart_<date>.
#Use restarts to debug a step, not for results that postprocess.py reads.
#SBATCH --job-name=lv_contraction
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=run_%j.out
#SBATCH --error=run_%j.err

module load foss/2024a VTK/9.3.1-foss-2024a PETSc/3.25.0-foss-2024a
set -euo pipefail

# ===== EDIT FOR YOUR CLUSTER =====================================
SVMP="$HOME/svMultiPhysics_petsc/build/svMultiPhysics-build/bin/svmultiphysics"
SVZEROD="$HOME/svZeroDSolver/build/src/interface/libsvzero_interface.so"
# =================================================================

[ -f "$SVZEROD" ] || { echo "svZeroDSolver library not found: $SVZEROD (see build_svzerod.slurm)"; exit 1; }
sed -i "s|<Shared_library>.*</Shared_library>|<Shared_library> $SVZEROD </Shared_library>|" solver.xml

#svMultiPhysics reads <Configuration_file> from the XML but does not pass it to PETSc, so the options go through
#the PETSC_OPTIONS environment variable, which PETSc reads at startup
if [ -f petsc_options.txt ]; then
    export PETSC_OPTIONS="$(grep -v '^\s*$' petsc_options.txt | tr '\n' ' ')"
    echo "PETSC_OPTIONS=$PETSC_OPTIONS"
fi

if [ -f prestress.xml ] && [ ! -f mesh/prestress.vtu ]; then
    echo "This case uses a prestress: run prestress.xml and extract_prestress.py first"; exit 1
fi

if [ "${1:-}" = "--restart" ]; then
    [ -f results/stFile_last.bin ] || { echo "No results/stFile_last.bin to restart from"; exit 1; }
    if [ -f results/n_ranks ] && [ "$(cat results/n_ranks)" != "$SLURM_NTASKS" ]; then
        echo "The restart file was written with $(cat results/n_ranks) ranks; submit with --ntasks=$(cat results/n_ranks)"
        exit 1
    fi
    [ -f results/svZeroD_data ] && cp results/svZeroD_data "results/svZeroD_data.before_restart_$(date +%Y%m%d_%H%M)"
    sed -i "s|<Continue_previous_simulation>.*</Continue_previous_simulation>|<Continue_previous_simulation> 1 </Continue_previous_simulation>|" solver.xml
else
    sed -i "s|<Continue_previous_simulation>.*</Continue_previous_simulation>|<Continue_previous_simulation> 0 </Continue_previous_simulation>|" solver.xml
    mkdir -p results
    echo "$SLURM_NTASKS" > results/n_ranks
fi

echo "start $(date) host=$(hostname) ranks=${SLURM_NTASKS} exe=$SVMP"
mpirun -np "$SLURM_NTASKS" "$SVMP" solver.xml
echo "done  $(date)"
