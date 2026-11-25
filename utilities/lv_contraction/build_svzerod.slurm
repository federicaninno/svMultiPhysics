#!/bin/bash
#Builds svZeroDSolver, whose interface library (libsvzero_interface.so) svMultiPhysics loads for the 3D-0D
#coupling, into $PREFIX.
#1. On the login node (needs internet), once:
#     git clone https://github.com/SimVascular/svZeroDSolver.git ~/svZeroDSolver
#2. From any folder:  sbatch path/to/build_svzerod.slurm
#   Library: ~/svZeroDSolver/build/src/interface/libsvzero_interface.so (used by run_case_petsc.slurm)
#If the configure step fails on a download (compute node without internet), run the same module loads and the
#cmake line below on the login node, then resubmit: the job skips cmake when the build folder is configured.
#SBATCH --job-name=build_svzerod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=build_svzerod_%j.out
#SBATCH --error=build_svzerod_%j.err

module load foss/2024a
module load CMake/3.29.3-GCCcore-13.3.0 2>/dev/null || module load CMake
#svZeroDSolver also builds its Python bindings, so CMake needs Python (interpreter and headers)
module load Python/3.12.3-GCCcore-13.3.0
set -euo pipefail

PREFIX="$HOME/svZeroDSolver"
[ -d "$PREFIX/src" ] || { echo "No source in $PREFIX: clone it on the login node first (see the header)"; exit 1; }
echo "svZeroDSolver commit $(git -C "$PREFIX" rev-parse HEAD)"

mkdir -p "$PREFIX/build"
cd "$PREFIX/build"
[ -f CMakeCache.txt ] || cmake .. -DCMAKE_BUILD_TYPE=Release
make -j "$SLURM_CPUS_PER_TASK"

LIB="$PREFIX/build/src/interface/libsvzero_interface.so"
[ -f "$LIB" ] || { echo "Build finished but $LIB is missing"; exit 1; }
echo "Built $LIB"
