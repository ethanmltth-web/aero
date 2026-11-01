"""Load, build, and normalize triangle models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

SAMPLE_NAMES = ["Wing", "Sphere", "Brick", "Cylinder", "Cone"]


def naca4_profile(m: float, p: float, t: float, count: int, chord: float):
    """Closed airfoil outline as x and z coordinates, leading edge first."""
    beta = np.linspace(0.0, np.pi, count)
    x = chord * (1.0 - np.cos(beta)) / 2.0
    xc = np.clip(x / chord, 0.0, 1.0)
    yt = 5.0 * t * chord * (
        0.2969 * np.sqrt(xc + 1e-16)
        - 0.1260 * xc
        - 0.3516 * xc**2
        + 0.2843 * xc**3
        - 0.1015 * xc**4
    )
    yt[0] = 0.0
    yt[-1] = 0.0
    yc = np.zeros_like(x)
    slope = np.zeros_like(x)
    if m > 1e-8 and 0.02 < p < 0.98:
        left = xc < p
        yc[left] = (m / p**2) * (2 * p * xc[left] - xc[left] ** 2) * chord
        slope[left] = (2 * m / p**2) * (p - xc[left])
        right = ~left
        yc[right] = (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * xc[right] - xc[right] ** 2) * chord
        slope[right] = (2 * m / (1 - p) ** 2) * (p - xc[right])
    theta = np.arctan(slope)
    xu = x - yt * np.sin(theta)
    zu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    zl = yc - yt * np.cos(theta)
    px = np.concatenate([xu, xl[-2:0:-1]])
    pz = np.concatenate([zu, zl[-2:0:-1]])
    return px, pz


def wing_mesh() -> pv.PolyData:
    """A cambered wing: chord along X, span along Y, thickness along Z."""
    chord = 0.52
    span = 1.75
    sections = 28
    px, pz = naca4_profile(0.02, 0.4, 0.12, 72, chord)
    px = px - (float(px.min()) + float(px.max())) / 2.0
    m = len(px)
    ys = np.linspace(-span / 2.0, span / 2.0, sections)
    points = np.empty((sections * m, 3), dtype=np.float64)
    for s, y in enumerate(ys):
        block = points[s * m : (s + 1) * m]
        block[:, 0] = px
        block[:, 1] = y
        block[:, 2] = pz
    faces = []
    for s in range(sections - 1):
        for i in range(m):
            i2 = (i + 1) % m
            a = s * m + i
            b = s * m + i2
            c = (s + 1) * m + i2
            d = (s + 1) * m + i
            faces.append((a, b, c))
            faces.append((a, c, d))
    for s, outward in ((0, False), (sections - 1, True)):
        base = s * m
        for i in range(1, m - 1):
            if outward:
                faces.append((base, base + i, base + i + 1))
            else:
                faces.append((base, base + i + 1, base + i))
    tris = np.asarray(faces, dtype=np.int64)
    flat = np.column_stack([np.full(len(tris), 3, dtype=np.int64), tris]).ravel()
    return pv.PolyData(points, faces=flat)


def sample_mesh(name: str) -> pv.PolyData:
    key = name.strip().lower()
    if key == "wing":
        return wing_mesh()
    if key == "sphere":
        return pv.Sphere(radius=0.5, theta_resolution=64, phi_resolution=64)
    if key == "brick":
        return pv.Cube(x_length=1.0, y_length=0.46, z_length=0.30)
    if key == "cylinder":
        return pv.Cylinder(center=(0, 0, 0), direction=(0, 1, 0), radius=0.22, height=1.2, resolution=64, capping=True)
    if key == "cone":
        # Tip points into the oncoming wind (toward -X).
        return pv.Cone(center=(0, 0, 0), direction=(-1, 0, 0), height=1.15, radius=0.32, resolution=64, capping=True)
    raise ValueError("Unknown sample shape: " + name)


def _flatten_block(data) -> pv.PolyData:
    if isinstance(data, pv.MultiBlock):
        parts = []

        def walk(block):
            if block is None:
                return
            if isinstance(block, pv.MultiBlock):
                for child in block:
                    walk(child)
                return
            parts.append(block)

        walk(data)
        if not parts:
            raise ValueError("This file has no geometry.")
        surfaces = []
        for part in parts:
            if isinstance(part, pv.PolyData):
                surfaces.append(part)
            else:
                surfaces.append(part.extract_surface())
        mesh = surfaces[0] if len(surfaces) == 1 else pv.merge(surfaces, merge_points=False)
    elif isinstance(data, pv.PolyData):
        mesh = data
    else:
        mesh = data.extract_surface()
    mesh = mesh.triangulate().clean()
    if mesh.n_points < 3 or mesh.n_cells < 1:
        raise ValueError("This file does not contain a surface.")
    return mesh


def _from_meshio(path: str) -> pv.PolyData:
    import meshio

    loaded = meshio.read(path)
    points = np.asarray(loaded.points, dtype=np.float64)[:, :3]
    blocks = []
    for cell_block in loaded.cells:
        kind = cell_block.type
        data = np.asarray(cell_block.data)
        if data.size == 0:
            continue
        if kind == "triangle":
            blocks.append(np.column_stack([np.full(len(data), 3), data[:, :3]]))
        elif kind == "quad":
            blocks.append(np.column_stack([np.full(len(data), 3), data[:, 0], data[:, 1], data[:, 2]]))
            blocks.append(np.column_stack([np.full(len(data), 3), data[:, 0], data[:, 2], data[:, 3]]))
    if not blocks:
        raise ValueError("No triangles or quads were found in this file.")
    faces = np.vstack(blocks).astype(np.int64).ravel()
    mesh = pv.PolyData(np.ascontiguousarray(points), faces=faces)
    mesh = mesh.triangulate().clean()
    if mesh.n_cells < 1:
        raise ValueError("This file does not contain a surface.")
    return mesh


def load_mesh(path: str) -> pv.PolyData:
    """Read a model file. The result is not scaled yet."""
    path = str(path)
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    errors = []
    try:
        return _flatten_block(pv.read(path))
    except Exception as exc:
        errors.append(str(exc))
    try:
        return _from_meshio(path)
    except Exception as exc:
        errors.append(str(exc))
    detail = errors[-1] if errors else "unknown error"
    raise ValueError("Could not read this file as a 3D model. Try STL or OBJ.\n" + detail)


def normalize_mesh(mesh: pv.PolyData, length: float = 1.0) -> pv.PolyData:
    """Center the model and scale its longest side to `length` meters."""
    mesh = mesh.copy(deep=True)
    points = np.asarray(mesh.points)
    if points.size == 0 or not np.isfinite(points).all():
        keep = np.isfinite(points).all(axis=1) if points.size else np.array([], dtype=bool)
        if keep.size == 0 or not np.any(keep):
            raise ValueError("This model has no usable points.")
        mesh = mesh.extract_points(keep, adjacent_cells=True).extract_surface().triangulate().clean()
    center = np.array(mesh.center, dtype=np.float64)
    mesh.translate(-center, inplace=True)
    bounds = np.array(mesh.bounds, dtype=np.float64).reshape(3, 2)
    size = float(np.max(bounds[:, 1] - bounds[:, 0]))
    if size < 1e-9:
        raise ValueError("This model is empty or flat in a way that cannot be shown.")
    factor = float(length) / size
    try:
        mesh.scale(factor, inplace=True, point=(0.0, 0.0, 0.0))
    except TypeError:
        mesh.scale(factor, inplace=True)
    return mesh


def _decimate_to(mesh: pv.PolyData, max_cells: int) -> pv.PolyData:
    if mesh.n_cells <= max_cells:
        return mesh
    reduction = 1.0 - (max_cells / float(mesh.n_cells))
    reduction = float(min(0.98, max(0.0, reduction)))
    try:
        smaller = mesh.decimate(reduction)
    except Exception:
        return mesh
    if smaller is None or smaller.n_cells < 1:
        return mesh
    return smaller.triangulate().clean()


def _normalize_rows(normals: np.ndarray) -> np.ndarray:
    normals = np.array(normals, dtype=np.float64, copy=True)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    bad = length[:, 0] < 1e-8
    normals[bad] = np.array([0.0, 0.0, 1.0])
    length[bad] = 1.0
    normals /= np.maximum(length, 1e-8)
    return normals


def _triangle_geometry(mesh: pv.PolyData):
    faces = np.asarray(mesh.faces)
    if faces.size < 4 or faces.size % 4 != 0:
        raise ValueError("The model could not be turned into triangles.")
    faces = faces.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("The model could not be turned into triangles.")
    idx = faces[:, 1:]
    pts = np.asarray(mesh.points, dtype=np.float64)
    a = pts[idx[:, 0]]
    b = pts[idx[:, 1]]
    c = pts[idx[:, 2]]
    return idx, a, b, c


def areas_of(mesh: pv.PolyData) -> np.ndarray:
    _idx, a, b, c = _triangle_geometry(mesh)
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def cell_centers_of(mesh: pv.PolyData) -> np.ndarray:
    _idx, a, b, c = _triangle_geometry(mesh)
    return (a + b + c) / 3.0


def _cell_normals_from_triangles(mesh: pv.PolyData) -> np.ndarray:
    _idx, a, b, c = _triangle_geometry(mesh)
    normals = np.cross(b - a, c - a)
    return _normalize_rows(normals)


def surface_normals(mesh: pv.PolyData):
    """Point normals and cell normals, same winding, unit length."""
    point_n = None
    cell_n = None
    try:
        pn = np.asarray(mesh.point_normals, dtype=np.float64)
        cn = np.asarray(mesh.cell_normals, dtype=np.float64)
        if len(pn) == mesh.n_points and len(cn) == mesh.n_cells:
            point_n = pn
            cell_n = cn
    except Exception:
        point_n = None
        cell_n = None
    if point_n is None or cell_n is None:
        cell_n = _cell_normals_from_triangles(mesh)
        idx, _a, _b, _c = _triangle_geometry(mesh)
        acc = np.zeros((mesh.n_points, 3), dtype=np.float64)
        for column in range(3):
            np.add.at(acc, idx[:, column], cell_n)
        point_n = acc
    return _normalize_rows(point_n), _normalize_rows(cell_n)


def prepare_for_view(mesh: pv.PolyData, max_cells: int = 1_200_000) -> pv.PolyData:
    mesh = mesh.triangulate().clean()
    if mesh.n_cells < 1 or mesh.n_points < 3:
        raise ValueError("This model does not contain a surface.")
    mesh = _decimate_to(mesh, max_cells)
    try:
        mesh.compute_normals(
            inplace=True,
            cell_normals=True,
            point_normals=True,
            auto_orient_normals=True,
            consistent_normals=True,
            split_vertices=False,
        )
    except TypeError:
        try:
            mesh.compute_normals(
                inplace=True,
                cell_normals=True,
                point_normals=True,
                auto_orient_normals=True,
                consistent_normals=True,
            )
        except Exception:
            mesh.compute_normals(inplace=True, cell_normals=True, point_normals=True)
    except Exception:
        mesh.compute_normals(inplace=True, cell_normals=True, point_normals=True)
    return mesh
