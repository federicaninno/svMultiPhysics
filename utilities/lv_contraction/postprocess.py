#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
# SPDX-License-Identifier: BSD-3-Clause
# -*-coding:utf-8 -*-
"""Postprocess a left ventricle contraction case prepared with prepare_case.py.

Reads the svZeroDSolver output of the coupled simulation (results/svZeroD_data)
and computes the LV pressure and volume. The volume is the reference cavity
volume minus the integral of the flow leaving the 3D LV through LV_IN.

Writes:
    - <results>/pv_loop.csv: time, LV volume, LV pressure, atrial and arterial pressures
    - <results>/pv_loop.png: PV loop of every cycle and the time traces
    - <results>/results.pvd: the saved results as one time series for ParaView
and prints EDV, ESV, EF and peak pressure per cycle.

With --check-3d-volume, the volume is also computed from the deformed 3D
mesh in the saved results, to check the volume obtained from the 0D flows.

Usage:
    python postprocess.py <case_dir> [--results-dir results] [--check-3d-volume]
"""

import argparse
import glob
import json
import os

import numpy as np

MMHG = 133.322e-6     # mmHg -> MPa
ML = 1.0e-6           # mL -> m^3


def read_svzerod_data(path):
    """Read the svZeroD_data output file.

    Args:
        path: Path to svZeroD_data.

    Returns:
        dict: Arrays of every svZeroD variable, keyed by name, plus 'step'.
    """
    with open(path) as f:
        names = f.readline().split()[1:]
        data = np.loadtxt(f, ndmin=2)
    out = {"step": data[:, 0].astype(int)}
    for i, name in enumerate(names):
        out[name] = data[:, i + 1]
    return out


def deformed_cavity_volumes(case_dir, results_dir, info):
    """Compute the cavity volume from the deformed 3D mesh of the saved results.

    Args:
        case_dir: Case folder.
        results_dir: Folder with the saved results (result_*.vtu).
        info: Case information written by prepare_case.py.

    Returns:
        tuple: (steps, volumes) arrays of saved time steps and volumes (mL).
    """
    import pyvista as pv

    mesh = pv.read(os.path.join(case_dir, "mesh", "mesh-complete.mesh.vtu"))
    endo = pv.read(os.path.join(case_dir, "mesh", "mesh-surfaces", "endo.vtp"))
    cap = pv.read(os.path.join(case_dir, "mesh", "mesh-surfaces", "endo_cap.vtp"))

    # Map the surface nodes to the volume mesh nodes through GlobalNodeID
    index = {gid: i for i, gid in enumerate(mesh.point_data["GlobalNodeID"])}
    endo_ids = np.array([index[g] for g in endo.point_data["GlobalNodeID"]])
    cap_ids = np.array([index[g] for g in cap.point_data["GlobalNodeID"]])
    scale = info["mesh_scale_factor"]

    steps, volumes = [], []
    for path in sorted(glob.glob(os.path.join(results_dir, "result_*.vtu"))):
        result = pv.read(path)
        x = scale * mesh.points + result.point_data["Displacement"]
        endo.points = x[endo_ids]
        cap.points = x[cap_ids]
        closed = (endo.extract_surface() + cap.extract_surface()).clean()
        steps.append(int(os.path.basename(path)[7:-4]))
        volumes.append(closed.volume / ML)

    return np.array(steps), np.array(volumes)


def write_pvd(results_dir, dt):
    """Write results.pvd, which ParaView opens as one time series of the saved results.

    Args:
        results_dir: Folder with the saved results (result_*.vtu).
        dt: Time step (ms); the time of each result is its time step times dt.

    Returns:
        str: Path to the .pvd file.
    """
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="Collection" version="0.1">',
             '  <Collection>']
    for path in sorted(glob.glob(os.path.join(results_dir, "result_*.vtu"))):
        name = os.path.basename(path)
        lines.append(f'    <DataSet timestep="{int(name[7:-4]) * dt:g}" file="{name}"/>')
    lines += ['  </Collection>', '</VTKFile>']
    path = os.path.join(results_dir, "results.pvd")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Postprocess a 3D-0D LV contraction case.")
    parser.add_argument("case_dir", help="Case folder (with case_info.json and results/)")
    parser.add_argument("--results-dir", default="results",
                        help="Results folder, relative to the case folder (default: %(default)s)")
    parser.add_argument("--check-3d-volume", action="store_true",
                        help="Also compute the volume from the deformed 3D mesh")
    args = parser.parse_args()

    with open(os.path.join(args.case_dir, "case_info.json")) as f:
        info = json.load(f)
    dt = info["time_step_ms"]
    period = info["period_ms"]
    results_dir = os.path.join(args.case_dir, args.results_dir)

    zd = read_svzerod_data(os.path.join(results_dir, "svZeroD_data"))
    # Row i of svZeroD_data is the state at the end of time step i + 1
    t = (zd["step"] + 1) * dt

    # LV_IN flow is positive out of the 3D LV, so V = V0 - int(Q) dt
    # (trapezoidal rule, starting from the initial flow of 0)
    q = np.concatenate([[0.0], zd["flow:LV_IN:lv_node"]])
    volume = info["reference_cavity_volume_mL"] - np.cumsum(0.5 * (q[1:] + q[:-1])) * dt / ML
    p_lv = zd["pressure:LV_IN:lv_node"] / MMHG
    p_atrium = zd["pressure:ATRIUM:MITRAL"] / MMHG
    p_arterial = zd["pressure:AORTIC:aorta_prox"] / MMHG

    np.savetxt(os.path.join(results_dir, "pv_loop.csv"),
               np.column_stack([t, volume, p_lv, p_atrium, p_arterial]), delimiter=",",
               header="time_ms,volume_mL,p_lv_mmHg,p_atrium_mmHg,p_arterial_mmHg", comments="")

    # Summary per cycle
    cycle = np.floor(t / period).astype(int)
    print(f"Case {info['case_name']}: period {period:.1f} ms, reference volume "
          f"{info['reference_cavity_volume_mL']:.1f} mL")
    print(" cycle   EDV (mL)   ESV (mL)   SV (mL)   EF (%)   max p_LV (mmHg)")
    for c in np.unique(cycle):
        k = cycle == c
        edv, esv = volume[k].max(), volume[k].min()
        print(f"  {c + 1:3d}   {edv:8.1f}   {esv:8.1f}   {edv - esv:7.1f}   {100 * (edv - esv) / edv:6.1f}"
              f"   {p_lv[k].max():10.1f}")

    if args.check_3d_volume:
        steps, v3d = deformed_cavity_volumes(args.case_dir, results_dir, info)
        v0d = np.interp(steps * dt, np.concatenate([[0.0], t]),
                        np.concatenate([[info["reference_cavity_volume_mL"]], volume]))
        print("\nVolume check (mL): deformed 3D mesh vs 0D flows")
        print(f"   max |difference| {np.abs(v3d - v0d).max():.2f} mL over {len(steps)} saved steps")

    print(f"\nSaved {write_pvd(results_dir, dt)} (open in ParaView; time in ms)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available: skipping the plot")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for c in np.unique(cycle):
            k = cycle == c
            axes[0].plot(volume[k], p_lv[k], label=f"cycle {c + 1}")
        if args.check_3d_volume:
            axes[0].plot(v3d, np.interp(steps * dt, t, p_lv), "k.", ms=3, label="3D mesh volume")
        axes[0].set_xlabel("LV volume (mL)")
        axes[0].set_ylabel("LV pressure (mmHg)")
        axes[0].set_title(f"{info['case_name']}: PV loop")
        axes[0].legend()

        axes[1].plot(t, p_lv, label="LV")
        axes[1].plot(t, p_atrium, label="atrium")
        axes[1].plot(t, p_arterial, label="aorta (proximal)")
        axes[1].set_xlabel("time (ms)")
        axes[1].set_ylabel("pressure (mmHg)")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "pv_loop.png"), dpi=150)
        print(f"\nSaved {os.path.join(results_dir, 'pv_loop.png')}")
