#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
# SPDX-License-Identifier: BSD-3-Clause
# -*-coding:utf-8 -*-
"""Extract the prestress computed by prestress.xml for the contraction run.

In a prestress run the displacement is reset to zero at every time step, so
the displacement of each step measures how far the geometry is from
equilibrium. This script prints it for every step, checks that it has become
negligible, and writes the nodal stress of the last step (which includes the
accumulated prestress) to mesh/prestress.vtu, read by solver.xml.

Usage:
    python extract_prestress.py <case_dir> [--tolerance 1e-3]
"""

import argparse
import glob
import os

import numpy as np
import pyvista as pv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract the prestress for the contraction run.")
    parser.add_argument("case_dir", help="Case folder (with prestress/)")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="Maximum displacement of the last step relative to the first (default: %(default)s)")
    args = parser.parse_args()

    results = sorted(glob.glob(os.path.join(args.case_dir, "prestress", "result_*.vtu")))
    if not results:
        raise FileNotFoundError(f"No prestress results in {os.path.join(args.case_dir, 'prestress')}")

    print(" step   max displacement (mm)   relative to step 1")
    for path in results:
        result = pv.read(path)
        u_max = np.linalg.norm(result.point_data["Displacement"], axis=1).max()
        if path == results[0]:
            u_first = u_max
        print(f"{os.path.basename(path)[7:-4]:>5}   {1e3 * u_max:20.3e}   {u_max / u_first:18.3e}")

    if u_max / u_first > args.tolerance:
        raise RuntimeError(f"The prestress has not converged (relative displacement {u_max / u_first:.1e} > "
                           f"{args.tolerance:.0e}): increase prestress_time_steps and rerun prestress.xml")

    mesh = pv.read(os.path.join(args.case_dir, "mesh", "mesh-complete.mesh.vtu"))
    prestress = mesh.copy()
    prestress.clear_data()
    prestress.point_data["GlobalNodeID"] = mesh.point_data["GlobalNodeID"]
    prestress.point_data["Stress"] = result.point_data["Stress"]
    path = os.path.join(args.case_dir, "mesh", "prestress.vtu")
    prestress.save(path)
    print(f"\nPrestress from {os.path.basename(results[-1])} written to {path}")
