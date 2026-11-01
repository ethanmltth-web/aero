"""Physics checks for the preview solver. No window required."""

from __future__ import annotations

import time

import numpy as np

from aero_app.meshes import (
    SAMPLE_NAMES,
    areas_of,
    cell_centers_of,
    normalize_mesh,
    prepare_for_view,
    sample_mesh,
    surface_normals,
)
from aero_app.solver import (
    compute_flow,
    compute_occupancy,
    downstream_shadow,
    solve_potential,
    trace_streamlines,
)


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_uniform_flow() -> None:
    n = 24
    solid = np.zeros((n, n, n), dtype=bool)
    origin = np.array([-1.0, -1.0, -1.0])
    spacing = np.full(3, 2.0 / (n - 1))
    wind = np.array([1.25, 0.0, 0.0])
    velocity = solve_potential(solid, origin, spacing, wind, iterations=20, omega=1.2)
    error = float(np.max(np.abs(velocity - wind)))
    print(f"uniform flow max error {error:.3e}")
    _expect(error < 1e-4, f"empty tunnel should stay uniform, error {error}")


def test_sphere_speeds_up_on_the_side() -> None:
    n = 36
    half = 1.6
    origin = np.array([-half, -half, -half])
    spacing = np.full(3, (2 * half) / (n - 1))
    xs = origin[0] + np.arange(n) * spacing[0]
    x, y, z = np.meshgrid(xs, xs, xs, indexing="ij")
    radius = 0.36
    solid = (x**2 + y**2 + z**2) <= radius**2
    wind = np.array([1.0, 0.0, 0.0])
    velocity = solve_potential(solid, origin, spacing, wind, iterations=90, omega=1.35)
    speed = np.linalg.norm(velocity, axis=-1)

    def at(px, py, pz):
        i = int(np.clip(np.rint((px - origin[0]) / spacing[0]), 0, n - 1))
        j = int(np.clip(np.rint((py - origin[1]) / spacing[1]), 0, n - 1))
        k = int(np.clip(np.rint((pz - origin[2]) / spacing[2]), 0, n - 1))
        return float(speed[i, j, k]), bool(solid[i, j, k])

    gap = 2.0 * float(spacing[0])
    nose, nose_solid = at(-radius - gap, 0.0, 0.0)
    side, side_solid = at(0.0, radius + gap, 0.0)
    print(f"sphere nose speed {nose:.3f} solid {nose_solid} | side speed {side:.3f} solid {side_solid}")
    _expect(not nose_solid and not side_solid, "sample points landed inside the sphere")
    _expect(side > nose + 0.05, f"side should be faster than the nose ({side:.3f} vs {nose:.3f})")


def test_shadow_goes_downstream() -> None:
    solid = np.zeros((20, 12, 12), dtype=bool)
    solid[6:9, 4:8, 4:8] = True
    origin = np.zeros(3)
    spacing = np.ones(3)
    shadow = downstream_shadow(solid, origin, spacing, np.array([1.0, 0.0, 0.0]), steps=10)
    xs = np.arange(20)

    def center_of_mass(mask):
        weight = mask.sum(axis=(1, 2))
        return float(np.sum(xs * weight) / (weight.sum() + 1e-9))

    solid_x = center_of_mass(solid)
    shadow_x = center_of_mass(shadow > 0.2)
    print(f"shadow center {shadow_x:.2f} vs body center {solid_x:.2f}")
    _expect(shadow_x > solid_x, "wake shadow should sit downstream of the body")


def test_samples_and_wing_lift() -> None:
    for name in SAMPLE_NAMES:
        mesh = prepare_for_view(normalize_mesh(sample_mesh(name)))
        print(f"sample {name}: {mesh.n_cells} triangles, open edges {mesh.n_open_edges}")
        _expect(mesh.n_cells > 10, name)

    mesh = prepare_for_view(normalize_mesh(sample_mesh("Wing")))
    started = time.perf_counter()
    occ = compute_occupancy(mesh, 40)
    print(f"occupancy {time.perf_counter() - started:.2f}s  solid {occ['solid'].mean():.2%}")
    origin = occ["origin"]
    spacing = occ["spacing"]
    solid = occ["solid"]
    ix = int(np.clip(np.rint((0.0 - origin[0]) / spacing[0]), 0, solid.shape[0] - 1))
    iy = int(np.clip(np.rint((0.0 - origin[1]) / spacing[1]), 0, solid.shape[1] - 1))
    blocked = bool(solid[ix, iy].any())
    print(f"wing blocks a vertical ray: {blocked}  column solids {int(solid[ix, iy].sum())}")
    _expect(blocked, "the wing should block airflow through its center")
    _expect(not bool(solid[1, 1, 1]), "the tunnel corner should stay open")

    point_n, cell_n = surface_normals(mesh)
    pitch = np.deg2rad(8.0)
    wind = np.array([np.cos(pitch), 0.0, np.sin(pitch)]) * 8.0
    started = time.perf_counter()
    result = compute_flow(
        solid,
        origin,
        spacing,
        wind,
        wake=0.45,
        rho=1.225,
        points=np.asarray(mesh.points),
        point_normals=point_n,
        cell_centers=cell_centers_of(mesh),
        cell_normals=cell_n,
        cell_areas=areas_of(mesh),
        iterations=50,
    )
    print(f"solve {time.perf_counter() - started:.2f}s")
    print("report", {key: round(float(value), 4) for key, value in result["report"].items()})
    _expect(result["report"]["lift"] > 0.0, "a wing at +8 degrees should make upward lift")
    _expect(result["report"]["drag"] > 0.0, "wake should add some drag")
    _expect(result["report"]["peak"] > 1.02, "air should speed up somewhere around the wing")
    paths, _speeds = trace_streamlines(
        result["velocity"], solid, origin, spacing, wind, steps=40
    )
    print("streamline seeds", paths.shape[0])
    _expect(paths.shape[0] > 4, "streamlines should find a path through the tunnel")


def main() -> None:
    test_uniform_flow()
    test_sphere_speeds_up_on_the_side()
    test_shadow_goes_downstream()
    test_samples_and_wing_lift()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
