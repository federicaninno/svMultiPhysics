#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
# SPDX-License-Identifier: BSD-3-Clause
# -*-coding:utf-8 -*-
"""Convert a linear tetrahedral mesh (tet4) and its surfaces to quadratic (tet10).

A node is added at the midpoint of every edge (the geometry stays the same).
The input mesh is not modified: the quadratic mesh is written to a new folder
with the same layout:
    <output>/mesh-complete.mesh.vtu        quadratic tetrahedra (VTK type 24)
    <output>/mesh-surfaces/<name>.vtu      quadratic triangles (VTK type 22)
    <output>/<fiber files>                 same cell data on the quadratic mesh
Surfaces are written as .vtu because .vtp files cannot hold quadratic
triangles; svMultiPhysics reads both. Corner nodes keep their GlobalNodeID,
new nodes are numbered after them, and GlobalElementID is unchanged.

Usage:
    python make_quadratic_mesh.py <input mesh folder> <output folder> [--fibers fiber.vtu sheet.vtu]
"""

import argparse
import glob
import os

import numpy as np
import pyvista as pv

# Corner pairs of the edge nodes, in VTK order
TET10_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
TRI6_EDGES = [(0, 1), (1, 2), (2, 0)]


def make_quadratic_volume(mesh):
    """Add a node at the midpoint of every edge of a tet4 mesh.

    Args:
        mesh: PyVista UnstructuredGrid with linear tetrahedra and GlobalNodeID.

    Returns:
        tuple: (quadratic mesh, dict mapping a sorted pair of corner indices to its edge node index).
    """
    if set(np.unique(mesh.celltypes)) != {pv.CellType.TETRA}:
        raise ValueError("The input mesh must contain only linear tetrahedra")
    tets = mesh.cells_dict[pv.CellType.TETRA]

    edge_node = {}
    points = [mesh.points]
    n = mesh.n_points
    new_points = []
    for a, b in TET10_EDGES:
        for i, j in zip(tets[:, a], tets[:, b]):
            key = (min(i, j), max(i, j))
            if key not in edge_node:
                edge_node[key] = n + len(new_points)
                new_points.append(0.5 * (mesh.points[i] + mesh.points[j]))
    points.append(np.array(new_points))

    cells = np.empty((len(tets), 10), dtype=np.int64)
    cells[:, :4] = tets
    for k, (a, b) in enumerate(TET10_EDGES):
        cells[:, 4 + k] = [edge_node[(min(i, j), max(i, j))] for i, j in zip(tets[:, a], tets[:, b])]

    quad = pv.UnstructuredGrid(np.hstack([np.full((len(cells), 1), 10), cells]).ravel(),
                               np.full(len(cells), pv.CellType.QUADRATIC_TETRA, dtype=np.uint8),
                               np.vstack(points))
    gid = mesh.point_data["GlobalNodeID"]
    quad.point_data["GlobalNodeID"] = np.concatenate(
        [gid, gid.max() + 1 + np.arange(len(new_points))]).astype(gid.dtype)
    quad.cell_data["GlobalElementID"] = mesh.cell_data["GlobalElementID"]
    return quad, edge_node


def make_quadratic_surface(surface, mesh, quad, edge_node):
    """Convert a linear triangle surface of the mesh to quadratic triangles.

    Args:
        surface: PyVista PolyData with linear triangles, GlobalNodeID and GlobalElementID.
        mesh: The linear volume mesh.
        quad: The quadratic volume mesh.
        edge_node: Edge node indices from make_quadratic_volume().

    Returns:
        PyVista UnstructuredGrid with quadratic triangles.
    """
    index = {g: i for i, g in enumerate(mesh.point_data["GlobalNodeID"])}
    tris = surface.faces.reshape(-1, 4)[:, 1:]
    corners = np.vectorize(index.get)(surface.point_data["GlobalNodeID"][tris])

    nodes = np.empty((len(tris), 6), dtype=np.int64)
    nodes[:, :3] = corners
    for k, (a, b) in enumerate(TRI6_EDGES):
        nodes[:, 3 + k] = [edge_node[(min(i, j), max(i, j))] for i, j in zip(corners[:, a], corners[:, b])]

    # Renumber the surface nodes locally
    used, local = np.unique(nodes, return_inverse=True)
    local = local.reshape(nodes.shape)
    out = pv.UnstructuredGrid(np.hstack([np.full((len(local), 1), 6), local]).ravel(),
                              np.full(len(local), pv.CellType.QUADRATIC_TRIANGLE, dtype=np.uint8),
                              quad.points[used])
    out.point_data["GlobalNodeID"] = quad.point_data["GlobalNodeID"][used]
    out.cell_data["GlobalElementID"] = surface.cell_data["GlobalElementID"]
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a tet4 mesh and its surfaces to tet10.")
    parser.add_argument("input_dir", help="Folder with mesh-complete.mesh.vtu and mesh-surfaces/*.vtp")
    parser.add_argument("output_dir", help="New folder for the quadratic mesh")
    parser.add_argument("--fibers", nargs="*", default=[],
                        help="Fiber files (cell data on the linear mesh), relative to input_dir")
    args = parser.parse_args()

    if os.path.abspath(args.input_dir) == os.path.abspath(args.output_dir):
        raise ValueError("The output folder must differ from the input folder")
    os.makedirs(os.path.join(args.output_dir, "mesh-surfaces"), exist_ok=True)

    mesh = pv.read(os.path.join(args.input_dir, "mesh-complete.mesh.vtu"))
    quad, edge_node = make_quadratic_volume(mesh)
    quad.save(os.path.join(args.output_dir, "mesh-complete.mesh.vtu"))
    print(f"Volume mesh: {mesh.n_points} -> {quad.n_points} nodes, {quad.n_cells} quadratic tetrahedra")

    for path in sorted(glob.glob(os.path.join(args.input_dir, "mesh-surfaces", "*.vtp"))):
        name = os.path.splitext(os.path.basename(path))[0]
        surface = make_quadratic_surface(pv.read(path), mesh, quad, edge_node)
        surface.save(os.path.join(args.output_dir, "mesh-surfaces", f"{name}.vtu"))
        print(f"Surface {name}: {surface.n_cells} quadratic triangles")

    for name in args.fibers:
        fibers = pv.read(os.path.join(args.input_dir, name))
        if fibers.n_cells != quad.n_cells:
            raise ValueError(f"{name} has {fibers.n_cells} cells, the mesh has {quad.n_cells}")
        out = quad.copy()
        out.clear_data()
        out.cell_data["FIB_DIR"] = fibers.cell_data["FIB_DIR"]
        out.save(os.path.join(args.output_dir, name))
        print(f"Fibers {name}: copied to the quadratic mesh")
