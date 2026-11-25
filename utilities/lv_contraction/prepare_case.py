#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
# SPDX-License-Identifier: BSD-3-Clause
# -*-coding:utf-8 -*-
"""Prepare a left ventricle contraction case coupled to a 0D circulation.

From a single-ventricle mesh (volume mesh plus endocardium, epicardium and
base surfaces) and a case config file, this script:
    1. copies the mesh into the case folder,
    2. generates the fibers with the single-ventricle Bayer method
       (utilities/fiber_generation),
    3. builds the endocardial cap that closes the cavity at the base,
    4. writes the active stress curve, the svZeroDSolver configuration and
       the svMultiPhysics input files of the prestress and contraction runs.

The mesh (image) geometry is taken as the end-diastolic configuration: the
prestress run computes the stress that balances the end-diastolic pressure in
that geometry, and the contraction run starts from there. The case folder is
self-contained; from inside it, run
    mpiexec -n <N> svmultiphysics prestress.xml
    python extract_prestress.py .
    mpiexec -n <N> svmultiphysics solver.xml
Model units are m, ms, kg, with pressures in MPa.

Usage:
    python prepare_case.py case_config.json [--skip-fibers]
"""

import argparse
import json
import os
import shutil
import string
import sys

import numpy as np
import pyvista as pv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "fiber_generation"))

from fiber_generation.laplace_solver import LaplaceSolver
from fiber_generation.fiber_generator import FibGenBayerSingle
from fiber_generation.surface_names import SurfaceName
from fiber_generation.surface_utils import generate_epi_apex

from make_quadratic_mesh import make_quadratic_volume, make_quadratic_surface


# Conversions from clinical units to model units (m, ms, kg, MPa)
MMHG = 133.322e-6                 # mmHg -> MPa
ML = 1.0e-6                       # mL -> m^3
KPA = 1.0e-3                      # kPa -> MPa
R_UNIT = MMHG * 1.0e3 / ML        # mmHg*s/mL -> MPa*ms/m^3
C_UNIT = ML / MMHG                # mL/mmHg -> m^3/MPa
L_UNIT = MMHG * 1.0e6 / ML        # mmHg*s^2/mL -> MPa*ms^2/m^3


def load_config(path):
    """Load the case config, resolving paths relative to the config file.

    Args:
        path: Path to the case config (.json).

    Returns:
        dict: Case configuration with absolute paths.
    """
    with open(path) as f:
        cfg = json.load(f)

    base_dir = os.path.dirname(os.path.abspath(path))
    resolve = lambda p: os.path.join(base_dir, os.path.expanduser(p))

    cfg["output_dir"] = resolve(cfg["output_dir"])
    for key in ["volume", "endo", "epi", "base"]:
        cfg["mesh"][key] = resolve(cfg["mesh"][key])
    for key in ["svmultiphysics", "svzerod_library"]:
        cfg["executables"][key] = resolve(cfg["executables"][key])

    return cfg


def copy_mesh(cfg, mesh_dir):
    """Copy the volume mesh and surfaces into the case folder with standard names.

    Args:
        cfg: Case configuration.
        mesh_dir: Destination mesh folder inside the case folder.

    Returns:
        dict: Paths of the copied mesh ('volume') and surfaces ('endo', 'epi', 'base').
    """
    os.makedirs(os.path.join(mesh_dir, "mesh-surfaces"), exist_ok=True)

    paths = {"volume": os.path.join(mesh_dir, "mesh-complete.mesh.vtu")}
    for key in ["endo", "epi", "base"]:
        paths[key] = os.path.join(mesh_dir, "mesh-surfaces", f"{key}.vtp")

    for key, dst in paths.items():
        src = cfg["mesh"][key]
        if not os.path.exists(src):
            raise FileNotFoundError(f"Mesh file for '{key}' not found: {src}")
        mesh = pv.read(src)
        if "GlobalNodeID" not in mesh.point_data:
            raise ValueError(f"{src} has no GlobalNodeID point data")
        shutil.copyfile(src, dst)

    return paths


def generate_fibers(cfg, paths, mesh_dir, svmultiphysics_exec):
    """Generate fiber and sheet directions with the single-ventricle Bayer method.

    Args:
        cfg: Case configuration.
        paths: Paths of the mesh and surfaces in the case folder.
        mesh_dir: Mesh folder inside the case folder.
        svmultiphysics_exec: svMultiPhysics executable (for the Laplace problems).
    """
    surface_paths = {
        SurfaceName.EPICARDIUM: paths["epi"],
        SurfaceName.BASE: paths["base"],
        SurfaceName.ENDOCARDIUM: paths["endo"],
        SurfaceName.EPICARDIUM_APEX: os.path.join(mesh_dir, "mesh-surfaces", "epi_apex.vtp"),
    }
    laplace_dir = os.path.join(mesh_dir, "fibers", "laplace")

    print("Generating EPICARDIUM_APEX surface...")
    generate_epi_apex(surface_paths)

    print("Running Laplace solver...")
    solver = LaplaceSolver(paths["volume"], surface_paths, svmultiphysics_exec + " ")
    laplace_results_file = solver.run("bayer_single", laplace_dir)
    if not os.path.exists(laplace_results_file):
        raise RuntimeError(f"Laplace solver failed: {laplace_results_file} not found")

    params = {
        "ALFA_END": cfg["fibers"]["alpha_endo"],
        "ALFA_EPI": cfg["fibers"]["alpha_epi"],
        "BETA_END": cfg["fibers"]["beta_endo"],
        "BETA_EPI": cfg["fibers"]["beta_epi"],
    }
    fib_gen = FibGenBayerSingle()
    fib_gen.load_laplace_results(laplace_results_file)
    fib_gen.generate_fibers(params)
    fib_gen.write_fibers(os.path.join(mesh_dir, "fibers"))


def min_area_triangulation(points, forbidden_chords=()):
    """Triangulate a closed 3D polygon with the minimum total area, without adding points.

    Dynamic programming over the polygon vertices (as in hole-filling methods,
    e.g. Liepa 2003). Every triangle (i, m, j) has i < m < j, so all triangles
    follow the winding of the polygon.

    Args:
        points: Array of shape (n, 3) with the polygon vertices, in loop order.
        forbidden_chords: Pairs of vertex indices that must not be connected
            (e.g. vertices already connected by an edge of the surrounding surface).

    Returns:
        np.ndarray: Array of shape (n - 2, 3) with the vertex indices of the triangles.
    """
    n = len(points)
    cost = np.zeros((n, n))
    best = np.zeros((n, n), dtype=int)

    penalty = np.zeros((n, n))
    for i, j in forbidden_chords:
        penalty[i, j] = penalty[j, i] = np.inf

    for gap in range(2, n):
        for i in range(n - gap):
            j = i + gap
            m = np.arange(i + 1, j)
            areas = 0.5 * np.linalg.norm(np.cross(points[m] - points[i], points[j] - points[i]), axis=1)
            total = cost[i, m] + cost[m, j] + areas + penalty[i, j]
            k = np.argmin(total)
            cost[i, j] = total[k]
            best[i, j] = m[k]

    if np.isinf(cost[0, n - 1]):
        raise RuntimeError("No triangulation of the rim avoids the forbidden chords")

    triangles = []
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        m = best[i, j]
        triangles.append((i, m, j))
        stack += [(i, m), (m, j)]

    return np.array(triangles)


def make_endo_cap(endo_path, cap_path):
    """Build the cap that closes the endocardial cavity at the base.

    The cap triangulates the open rim of the endocardium using only the rim
    nodes, so every cap node is a mesh node (svMultiPhysics matches them by
    GlobalNodeID). The cell normals point into the cavity, consistently with
    the endocardial face normals.

    Args:
        endo_path: Path to the endocardial surface (.vtp).
        cap_path: Output path of the cap surface (.vtp).

    Returns:
        PyVista PolyData of the cap.
    """
    endo = pv.read(endo_path)
    rim = endo.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                     manifold_edges=False, non_manifold_edges=False)
    if rim.n_points == 0:
        raise ValueError(f"The endocardium {endo_path} is closed: there is no rim to cap")
    if rim.connectivity().point_data["RegionId"].max() > 0:
        raise ValueError(f"The endocardium {endo_path} has more than one open rim")

    # Order the rim nodes along the loop
    lines = rim.lines.reshape(-1, 3)[:, 1:]
    neighbors = {i: [] for i in range(rim.n_points)}
    for i, j in lines:
        neighbors[i].append(j)
        neighbors[j].append(i)
    loop = [0, neighbors[0][0]]
    while len(loop) < rim.n_points:
        prev, curr = loop[-2], loop[-1]
        loop.append(neighbors[curr][0] if neighbors[curr][0] != prev else neighbors[curr][1])

    points = rim.points[loop]
    global_node_ids = rim.point_data["GlobalNodeID"][loop].astype(np.int32)

    # Chords between rim nodes that are already connected by an endocardial
    # edge would make the closed surface non-manifold
    loop_index = {gid: k for k, gid in enumerate(global_node_ids)}
    endo_ids = endo.point_data["GlobalNodeID"]
    forbidden_chords = set()
    for tri in endo.faces.reshape(-1, 4)[:, 1:]:
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            i, j = loop_index.get(endo_ids[a]), loop_index.get(endo_ids[b])
            if i is not None and j is not None and abs(i - j) not in (1, len(loop) - 1):
                forbidden_chords.add((i, j))

    # The rim is in general not planar (and may fold over itself when
    # projected), so triangulate it in 3D with the minimum-area triangulation
    triangles = min_area_triangulation(points, forbidden_chords)

    # All triangles follow the loop winding: flip them together so that the
    # normals point into the cavity
    normals = np.cross(points[triangles[:, 1]] - points[triangles[:, 0]],
                       points[triangles[:, 2]] - points[triangles[:, 0]])
    into_cavity = endo.points.mean(axis=0) - points.mean(axis=0)
    if normals.sum(axis=0) @ into_cavity < 0.0:
        triangles = triangles[:, ::-1]
        normals = -normals
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    cap = pv.PolyData(points, faces=np.hstack([np.full((len(triangles), 1), 3), triangles]))
    cap.point_data["GlobalNodeID"] = global_node_ids
    cap.cell_data["Normals"] = normals
    cap.save(cap_path)

    return cap


def cavity_volume(endo_path, cap, scale_factor=1.0):
    """Compute the volume enclosed by the endocardium and its cap.

    Args:
        endo_path: Path to the endocardial surface (.vtp).
        cap: PyVista PolyData of the cap.
        scale_factor: Factor converting the mesh coordinates to metres.

    Returns:
        float: Cavity volume (m^3).
    """
    closed = (pv.read(endo_path).extract_surface() + cap.extract_surface()).clean()
    return closed.volume * scale_factor**3


def write_domains(cfg, volume_path, domain_path):
    """Mark the infarct elements and write the domain file read by svMultiPhysics.

    Elements whose centroid lies inside the infarct sphere get DOMAIN_ID 2,
    the others (healthy myocardium) DOMAIN_ID 1.

    Args:
        cfg: Case configuration.
        volume_path: Path to the volume mesh in the case folder.
        domain_path: Output path of the domain file (.vtu).

    Returns:
        float: Fraction of the wall volume in the infarct.
    """
    infarct = cfg["infarct"]
    mesh = pv.read(volume_path)
    centers = mesh.cell_centers().points
    inside = np.linalg.norm(centers - np.asarray(infarct["center"]), axis=1) < infarct["radius"]
    if not inside.any():
        raise ValueError("The infarct sphere contains no element: check infarct center and radius "
                         "(mesh coordinates, same units as the mesh file)")

    domains = mesh.copy()
    domains.clear_data()
    domains.cell_data["DOMAIN_ID"] = np.where(inside, 2, 1).astype(np.int32)
    domains.save(domain_path)

    volumes = mesh.compute_cell_sizes(length=False, area=False)["Volume"]
    return volumes[inside].sum() / volumes.sum()


def write_epi_spring(cfg, epi_path, base_path, spring_path):
    """Write the epicardial spring stiffness, decreasing from the apex to zero towards the base.

    The stiffness is epi_stiffness at the apex and decreases linearly with the
    height along the long axis (apex to base centroid) to zero at
    epi_stiffness_free_height (a fraction of that height), so that the base
    can descend while the apex stays in place.

    Args:
        cfg: Case configuration.
        epi_path: Path to the epicardial surface (.vtp).
        base_path: Path to the base surface (.vtp).
        spring_path: Output path (.vtp): the epicardium with Stiffness and Damping point data.

    Returns:
        float: Fraction of the epicardial nodes with a nonzero stiffness.
    """
    bcs = cfg["boundary_conditions"]
    epi = pv.read(epi_path)
    points = np.asarray(epi.points, dtype=float)
    base_center = np.asarray(pv.read(base_path).points, dtype=float).mean(axis=0)
    apex = points[np.argmax(np.linalg.norm(points - base_center, axis=1))]
    long_axis = base_center - apex
    # Elementwise products instead of @, which prints spurious warnings with numpy on macOS
    height = ((points - apex) * long_axis).sum(axis=1) / (long_axis * long_axis).sum()

    free_height = bcs["epi_stiffness_free_height"]
    stiffness = bcs["epi_stiffness"] * np.clip(1.0 - height / free_height, 0.0, 1.0)
    epi.point_data["Stiffness"] = stiffness
    epi.point_data["Damping"] = np.full(epi.n_points, float(bcs["epi_damping"]))
    epi.save(spring_path)
    return np.mean(stiffness > 0.0)


def write_active_stress(path, cfg, period, n_samples=500, n_fourier=50):
    """Write the active tension curve, periodic over one cardiac cycle.

    The tension follows a sin^2 pulse between the start and the end of the
    active phase, and is zero otherwise.

    Args:
        path: Output path of the temporal values file.
        cfg: Case configuration.
        period: Cardiac period (ms).
        n_samples: Number of time samples over one period.
        n_fourier: Number of Fourier coefficients used by svMultiPhysics.
    """
    timing = cfg["timing"]
    t = np.linspace(0.0, period, n_samples)
    phase = (t - timing["active_stress_start"] * period) / (timing["active_stress_duration"] * period)
    tension = cfg["active_stress"]["peak_kPa"] * KPA * np.sin(np.pi * phase)**2
    tension[(phase < 0.0) | (phase > 1.0)] = 0.0

    with open(path, "w") as f:
        f.write(f"{n_samples} {n_fourier}\n")
        for ti, si in zip(t, tension):
            f.write(f"{ti:.6f} {si:.8e}\n")


def atrial_pressure(cfg, period, dt):
    """Compute the atrial pressure over one cardiac cycle.

    Raised-cosine atrial systole on top of a constant baseline, as in Eq. (65)
    of Brown et al. (2024).

    Args:
        cfg: Case configuration.
        period: Cardiac period (ms).
        dt: Sampling interval (ms).

    Returns:
        tuple: (t, p) lists of times (ms) and pressures (MPa).
    """
    timing = cfg["timing"]
    circ = cfg["circulation"]
    t = np.linspace(0.0, period, int(round(period / dt)) + 1)
    phase = (t - timing["atrial_systole_start"] * period) / (timing["atrial_systole_duration"] * period)
    p = circ["atrial_pressure_amplitude_mmHg"] * 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))
    p[(phase < 0.0) | (phase > 1.0)] = 0.0
    p += circ["atrial_pressure_baseline_mmHg"]
    return t.tolist(), (p * MMHG).tolist()


def write_svzerod_config(path, cfg, period, dt):
    """Write the svZeroDSolver configuration of the open-loop circulation.

    Topology (Brown et al. 2024): atrium -[mitral]- LV -[aortic]- p_p -Rp,Lp- p_d -Rd- Pd,
    with Cp at p_p and Cd at p_d. The 3D LV couples through the LV_IN block.

    Args:
        path: Output path of the configuration (.json).
        cfg: Case configuration.
        period: Cardiac period (ms).
        dt: Time step of the 3D simulation (ms).
    """
    circ = cfg["circulation"]
    p0 = circ["initial_arterial_pressure_mmHg"] * MMHG
    p_ed = cfg["end_diastolic_pressure_mmHg"] * MMHG
    t_atrium, p_atrium = atrial_pressure(cfg, period, dt=2.0)

    valve_params = {"Rmax": 1.0e6, "Rmin": 1.0e3, "Steepness": 1.0e4}
    config = {
        "description": {
            "description of case": f"{cfg['case_name']}: 3D LV coupled to the open-loop circulation of Brown et al. (2024), CMAME 421:116764.",
            "units": "m, ms, kg; pressure in MPa; flow in m^3/ms",
        },
        "simulation_parameters": {
            "coupled_simulation": True,
            "number_of_time_pts": 101,
            "external_step_size": dt,
            "output_all_cycles": True,
            "steady_initial": False,
            "absolute_tolerance": 1e-11,
            "maximum_nonlinear_iterations": 10000,
        },
        "initial_condition": {
            # The simulation starts at end-diastole: LV at the end-diastolic
            # pressure, arteries at the initial (diastolic) pressure
            "pressure:LV_IN:lv_node": p_ed,
            "pressure:lv_node:J_LV": p_ed,
            "pressure:MITRAL:J_LV": p_ed,
            "pressure:J_LV:AORTIC": p_ed,
            "pressure:AORTIC:aorta_prox": p0,
            "pressure:aorta_prox:DISTAL": p0,
            "pressure_c:DISTAL": p0,
        },
        "boundary_conditions": [
            {
                "bc_name": "ATRIUM",
                "bc_type": "PRESSURE",
                "bc_values": {"t": t_atrium, "P": p_atrium},
            },
            {
                "bc_name": "DISTAL",
                "bc_type": "RCR",
                "bc_values": {
                    "Rp": 0.0,
                    "C": circ["Cd_mL_per_mmHg"] * C_UNIT,
                    "Rd": circ["Rd_mmHg_s_per_mL"] * R_UNIT,
                    "Pd": circ["Pd_mmHg"] * MMHG,
                },
            },
        ],
        "external_solver_coupling_blocks": [
            {
                "name": "LV_IN",
                "type": "FLOW",
                "location": "inlet",
                "connected_block": "lv_node",
                "periodic": False,
                "values": {"Q": [0.0, 0.0], "t": [0.0, 1.0]},
            }
        ],
        "vessels": [
            {
                "vessel_id": 0,
                "vessel_length": 1.0,
                "vessel_name": "lv_node",
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"R_poiseuille": 0.0, "C": 0.0, "L": 0.0},
            },
            {
                "boundary_conditions": {"outlet": "DISTAL"},
                "vessel_id": 1,
                "vessel_length": 1.0,
                "vessel_name": "aorta_prox",
                "zero_d_element_type": "BloodVesselCRL",
                "zero_d_element_values": {
                    "R_poiseuille": circ["Rp_mmHg_s_per_mL"] * R_UNIT,
                    "C": circ["Cp_mL_per_mmHg"] * C_UNIT,
                    "L": circ["Lp_mmHg_s2_per_mL"] * L_UNIT,
                },
            },
        ],
        "junctions": [
            {
                "junction_name": "J_LV",
                "junction_type": "NORMAL_JUNCTION",
                "inlet_blocks": ["lv_node"],
                "outlet_blocks": [],
            }
        ],
        "valves": [
            {
                "type": "ValveTanh",
                "name": "MITRAL",
                "params": {**valve_params, "upstream_block": "ATRIUM", "downstream_block": "J_LV"},
            },
            {
                "type": "ValveTanh",
                "name": "AORTIC",
                "params": {**valve_params, "upstream_block": "J_LV", "downstream_block": "aorta_prox"},
            },
        ],
    }

    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def write_solver_xml(template_name, path, cfg, period, dt, n_time_steps):
    """Write an svMultiPhysics input file from a template in templates/.

    Args:
        template_name: Template file name (e.g. 'solver.xml' or 'prestress.xml').
        path: Output path of the input file (.xml).
        cfg: Case configuration.
        period: Cardiac period (ms).
        dt: Time step (ms).
        n_time_steps: Number of time steps.
    """
    timing = cfg["timing"]
    bcs = cfg["boundary_conditions"]

    if cfg.get("infarct", {}).get("enabled", False):
        domain_line = "  <Domain_file_path> mesh/domains.vtu </Domain_file_path> <!-- 1: healthy, 2: infarct -->"
    else:
        domain_line = "  <!-- Single material: no infarct -->"

    if bcs.get("epi_stiffness_free_height") is not None:
        epi_spring = ("    <!-- Stiffness decreasing from the apex to zero towards the base, from the config -->\n"
                      "    <Spatial_values_file_path> mesh/mesh-surfaces/epi_spring.vtp </Spatial_values_file_path>")
    else:
        epi_spring = (f"    <Stiffness> {bcs['epi_stiffness']} </Stiffness> <!-- MPa/m -->\n"
                      f"    <Damping> {bcs['epi_damping']} </Damping> <!-- MPa*ms/m -->")

    # The prestress run is quasi-static (no inertia) and has no active tension
    if template_name == "prestress.xml":
        materials = materials_xml(cfg, density=0.0, active=False)
    else:
        materials = materials_xml(cfg, density=1000.0, active=True)

    if cfg["use_prestress"]:
        prestress_line = ("  <!-- Prestress balancing the end-diastolic pressure, from extract_prestress.py -->\n"
                          "  <Prestress_file_path> mesh/prestress.vtu </Prestress_file_path>")
    else:
        prestress_line = "  <!-- No prestress: the mesh (image) geometry is stress-free -->"

    # fsils (diagonal preconditioner) works for linear and quadratic tetrahedra
    # (slowly for quadratic ones). PETSc, with the solver set in
    # petsc_options.txt, leaves the cap out of the 3D-0D coupling term in
    # svMultiPhysics, so Newton stalls with the cap.
    if cfg.get("linear_solver", "fsils") == "petsc":
        linear_algebra = ('    <Linear_algebra type="petsc" >\n'
                          '      <Preconditioner> petsc-jacobi </Preconditioner>\n'
                          '      <Configuration_file> petsc_options.txt </Configuration_file>\n'
                          '    </Linear_algebra>')
    else:
        linear_algebra = ('    <Linear_algebra type="fsils" >\n'
                          '      <Preconditioner> fsils </Preconditioner>\n'
                          '    </Linear_algebra>')

    values = {
        "n_time_steps": n_time_steps,
        "time_step": dt,
        "prestress_line": prestress_line,
        "linear_algebra": linear_algebra,
        "linear_solver_tolerance": cfg.get("linear_solver_tolerance", 1e-6),
        "surface_ext": "vtu" if cfg["mesh"].get("element_order", 1) == 2 else "vtp",
        "prestress_time_step": cfg.get("prestress_time_step_ms", 10.0),
        "save_increment": max(1, int(round(timing["save_every_ms"] / dt))),
        "restart_increment": max(1, int(round(timing["save_every_ms"] / dt))),
        "mesh_scale_factor": cfg["mesh"]["scale_factor"],
        "svzerod_library": cfg["executables"]["svzerod_library"],
        "domain_line": domain_line,
        "materials": materials,
        "epi_spring": epi_spring,
        "base_stiffness": bcs["base_stiffness"],
        "base_damping": bcs["base_damping"],
        "end_diastolic_pressure": cfg["end_diastolic_pressure_mmHg"] * MMHG,
    }

    with open(os.path.join(HERE, "templates", template_name)) as f:
        template = string.Template(f.read())
    with open(path, "w") as f:
        f.write(template.substitute(values))


def material_xml(cfg, density, active, stiffness_factor=1.0, indent="  "):
    """Write the material block of the struct equation (or of one of its domains).

    Args:
        cfg: Case configuration.
        density: Density (kg/m^3).
        active: If True, add the prescribed active tension along the fibers.
        stiffness_factor: Factor scaling the passive stiffnesses (a, a4f, a4s, afs).
        indent: Indentation of the block.

    Returns:
        str: XML lines.
    """
    mat = cfg["material"]
    stiffness = lambda key: f"{mat[key] * KPA * stiffness_factor:.6g}"   # MPa
    lines = [
        f"<Density> {density} </Density> <!-- kg/m^3 -->",
        "<Viscosity model=\"Potential\">",
        "  <Value> 0.1 </Value> <!-- MPa*ms -->",
        "</Viscosity>",
        "",
        "<Constitutive_model type=\"HolzapfelOgden-ModifiedAnisotropy\">",
        f"  <a> {stiffness('a_kPa')} </a>     <!-- MPa -->",
        f"  <b> {mat['b']} </b>",
        f"  <a4f> {stiffness('a4f_kPa')} </a4f> <!-- MPa -->",
        f"  <b4f> {mat['b4f']} </b4f>",
        f"  <a4s> {stiffness('a4s_kPa')} </a4s> <!-- MPa -->",
        f"  <b4s> {mat['b4s']} </b4s>",
        f"  <afs> {stiffness('afs_kPa')} </afs> <!-- MPa -->",
        f"  <bfs> {mat['bfs']} </bfs>",
        "  <k> 100.0 </k>",
        "</Constitutive_model>",
        "<Dilational_penalty_model> Quadratic </Dilational_penalty_model>",
        f"<Penalty_parameter> {mat['penalty_kPa'] * KPA:.6g} </Penalty_parameter> <!-- MPa -->",
    ]
    if active:
        # Share of the tension across the fibers, in the wall plane (the sheet
        # direction is transmural); svMultiPhysics requires the shares to sum to 1
        cross = cfg["active_stress"].get("cross_fiber_fraction", 0.0)
        lines += [
            "",
            "<!-- Prescribed active tension, periodic over one cycle -->",
            "<Active_stress>",
            "  <Model> UniformUnsteady </Model>",
            "  <Directional_distribution>",
            f"    <Fiber_direction> {1.0 - cross:.6g} </Fiber_direction>",
            "    <Sheet_direction> 0.0 </Sheet_direction>",
            f"    <Sheet_normal_direction> {cross:.6g} </Sheet_normal_direction>",
            "  </Directional_distribution>",
            "  <UniformUnsteady>",
            "    <Ramp> false </Ramp>",
            "    <Temporal_values_file_path> active_stress.dat </Temporal_values_file_path>",
            "  </UniformUnsteady>",
            "</Active_stress>",
        ]
    return "\n".join(indent + line if line else "" for line in lines)


def materials_xml(cfg, density, active):
    """Write the materials of the struct equation: one block, or one per domain with an infarct.

    Args:
        cfg: Case configuration.
        density: Density (kg/m^3).
        active: If True, the healthy myocardium contracts.

    Returns:
        str: XML lines.
    """
    infarct = cfg.get("infarct", {})
    if not infarct.get("enabled", False):
        return material_xml(cfg, density, active)

    return ("  <!-- Healthy myocardium (DOMAIN_ID 1 in mesh/domains.vtu) -->\n"
            "  <Domain id=\"1\" >\n"
            "    <Equation> struct </Equation>\n"
            f"{material_xml(cfg, density, active, indent='    ')}\n"
            "  </Domain>\n\n"
            "  <!-- Infarct (DOMAIN_ID 2): no active tension, stiffer passive material (scar) -->\n"
            "  <Domain id=\"2\" >\n"
            "    <Equation> struct </Equation>\n"
            f"{material_xml(cfg, density, False, infarct['stiffness_factor'], indent='    ')}\n"
            "  </Domain>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a 3D-0D LV contraction case.")
    parser.add_argument("config", help="Case config file (.json)")
    parser.add_argument("--skip-fibers", action="store_true",
                        help="Reuse the fibers already in the case folder")
    parser.add_argument("--output-dir", default=None,
                        help="Case folder, overriding output_dir in the config")
    parser.add_argument("--svzerod-library", default=None,
                        help="svZeroDSolver coupling library written in solver.xml, overriding the config "
                             "(e.g. its path on the cluster where the case will run)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.svzerod_library:
        cfg["executables"]["svzerod_library"] = args.svzerod_library
    case_dir = os.path.abspath(args.output_dir) if args.output_dir else cfg["output_dir"]
    mesh_dir = os.path.join(case_dir, "mesh")
    os.makedirs(case_dir, exist_ok=True)

    period = 60.0e3 / cfg["timing"]["heart_rate_bpm"]   # ms
    dt = cfg["timing"]["time_step_ms"]

    # Without prestress the mesh geometry is stress-free, so the simulation
    # must start with no pressure in the LV, and the mitral valve closed
    if not cfg["use_prestress"] and cfg["end_diastolic_pressure_mmHg"] != 0.0:
        raise ValueError("Without prestress the mesh geometry is stress-free: "
                         "set end_diastolic_pressure_mmHg to 0, or use_prestress to true")
    if cfg["circulation"]["atrial_pressure_baseline_mmHg"] > cfg["end_diastolic_pressure_mmHg"]:
        print("Warning: the atrial pressure at t = 0 is above the initial LV pressure, "
              "so the LV fills before contracting")

    print(f"Preparing case '{cfg['case_name']}' in {case_dir}")
    paths = copy_mesh(cfg, mesh_dir)

    if args.skip_fibers:
        for name in ["fiber.vtu", "sheet.vtu"]:
            if not os.path.exists(os.path.join(mesh_dir, "fibers", name)):
                raise FileNotFoundError(f"--skip-fibers given, but {name} is not in {mesh_dir}/fibers")
    else:
        generate_fibers(cfg, paths, mesh_dir, cfg["executables"]["svmultiphysics"])

    cap = make_endo_cap(paths["endo"], os.path.join(mesh_dir, "mesh-surfaces", "endo_cap.vtp"))
    volume = cavity_volume(paths["endo"], cap, cfg["mesh"]["scale_factor"])
    print(f"Cavity volume of the reference configuration: {volume / ML:.1f} mL")

    # Quadratic elements: the fibers and the cap are computed on the linear
    # mesh, then the mesh, surfaces and fibers are converted to tet10. Corner
    # nodes keep their GlobalNodeID, so the cap (on the rim corner nodes) and
    # the linear .vtp surfaces (used by postprocess.py) stay valid.
    if cfg["mesh"].get("element_order", 1) == 2:
        mesh = pv.read(paths["volume"])
        quad, edge_node = make_quadratic_volume(mesh)
        for key in ["endo", "epi", "base"]:
            surface = make_quadratic_surface(pv.read(paths[key]), mesh, quad, edge_node)
            surface.save(os.path.join(mesh_dir, "mesh-surfaces", f"{key}.vtu"))
        for name in ["fiber.vtu", "sheet.vtu"]:
            path = os.path.join(mesh_dir, "fibers", name)
            fibers = quad.copy()
            fibers.clear_data()
            fibers.cell_data["FIB_DIR"] = pv.read(path).cell_data["FIB_DIR"]
            fibers.save(path)
        quad.save(paths["volume"])
        print(f"Converted to quadratic tetrahedra: {quad.n_points} nodes")

    if cfg["boundary_conditions"].get("epi_stiffness_free_height") is not None:
        if cfg["mesh"].get("element_order", 1) == 2:
            raise NotImplementedError("epi_stiffness_free_height needs linear tetrahedra (element_order 1): "
                                      "svMultiPhysics reads the spring from a .vtp face file")
        fraction = write_epi_spring(cfg, paths["epi"], paths["base"],
                                    os.path.join(mesh_dir, "mesh-surfaces", "epi_spring.vtp"))
        print(f"Epicardial spring: nonzero on {100 * fraction:.0f} % of the epicardial nodes "
              f"(mesh/mesh-surfaces/epi_spring.vtp)")

    if cfg.get("infarct", {}).get("enabled", False):
        fraction = write_domains(cfg, paths["volume"], os.path.join(mesh_dir, "domains.vtu"))
        print(f"Infarct: {100 * fraction:.1f} % of the wall volume (mesh/domains.vtu)")

    if cfg.get("linear_solver", "fsils") == "petsc":
        with open(os.path.join(case_dir, "petsc_options.txt"), "w") as f:
            f.write("\n".join(cfg["petsc_options"]) + "\n")

    write_active_stress(os.path.join(case_dir, "active_stress.dat"), cfg, period)
    write_svzerod_config(os.path.join(case_dir, "svzerod_3Dcoupling.json"), cfg, period, dt)
    if cfg["use_prestress"]:
        write_solver_xml("prestress.xml", os.path.join(case_dir, "prestress.xml"), cfg, period, dt,
                         cfg["prestress_time_steps"])
    write_solver_xml("solver.xml", os.path.join(case_dir, "solver.xml"), cfg, period, dt,
                     int(round(cfg["timing"]["n_cycles"] * period / dt)))

    # Information needed by extract_prestress.py and postprocess.py
    with open(os.path.join(case_dir, "case_info.json"), "w") as f:
        json.dump({"case_name": cfg["case_name"],
                   "period_ms": period,
                   "time_step_ms": dt,
                   "n_cycles": cfg["timing"]["n_cycles"],
                   "end_diastolic_pressure_mmHg": cfg["end_diastolic_pressure_mmHg"],
                   "reference_cavity_volume_mL": volume / ML,
                   "mesh_scale_factor": cfg["mesh"]["scale_factor"],
                   "infarct": cfg.get("infarct", {"enabled": False})}, f, indent=2)

    exe = cfg["executables"]["svmultiphysics"]
    print(f"\nDone. To run the simulation:\n   cd {case_dir}")
    if cfg["use_prestress"]:
        print(f"   mpiexec -n <N> {exe} prestress.xml\n"
              f"   python {os.path.join(HERE, 'extract_prestress.py')} .")
    print(f"   mpiexec -n <N> {exe} solver.xml")
