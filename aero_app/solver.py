"""Preview airflow.

The base field is potential flow: air speeds up where it squeezes around the
model and slows at the nose. A wake control then slows the smoke behind the
model and adds a simple base-drag suction on rear-facing surfaces.

This is a preview for comparing shape and angle. It is not a Navier–Stokes solver.
"""

from __future__ import annotations

import numpy as np

QUALITY = {
    "Fast": (32, 40),
    "Medium": (44, 58),
    "Detailed": (56, 76),
}


def flow_basis(wind_hat: np.ndarray):
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(up, wind_hat))) > 0.92:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, wind_hat)
    right = right / np.linalg.norm(right)
    up = np.cross(wind_hat, right)
    up = up / np.linalg.norm(up)
    return right, up


def _shift_axis(field: np.ndarray, amount: float, axis: int) -> np.ndarray:
    if abs(amount) < 1e-8:
        return field
    n = field.shape[axis]
    src = np.arange(n, dtype=np.float64) - amount
    i0 = np.floor(src).astype(np.int32)
    frac = (src - i0).astype(np.float64)
    i1 = i0 + 1

    def take(indices):
        ok = (indices >= 0) & (indices < n)
        safe = np.clip(indices, 0, n - 1)
        values = np.take(field, safe, axis=axis)
        shape = [1] * field.ndim
        shape[axis] = n
        return values, ok.reshape(shape)

    v0, ok0 = take(i0)
    v1, ok1 = take(i1)
    shape = [1] * field.ndim
    shape[axis] = n
    tt = frac.reshape(shape)
    mixed = v0 * (1.0 - tt) + v1 * tt
    mixed = np.where(ok0 & (ok1 | (tt == 0)), mixed, 0.0)
    return mixed


def _shift_zero(field: np.ndarray, shift: np.ndarray) -> np.ndarray:
    out = field
    for axis, amount in enumerate(shift):
        out = _shift_axis(out, float(amount), axis)
    return out


def _blur(field: np.ndarray) -> np.ndarray:
    padded = np.pad(field, 1, mode="edge")
    center = padded[1:-1, 1:-1, 1:-1]
    total = center * 6.0
    total = total + padded[:-2, 1:-1, 1:-1] + padded[2:, 1:-1, 1:-1]
    total = total + padded[1:-1, :-2, 1:-1] + padded[1:-1, 2:, 1:-1]
    total = total + padded[1:-1, 1:-1, :-2] + padded[1:-1, 1:-1, 2:]
    return total / 12.0


def _wind_coord(shape, origin, spacing, wind_hat) -> np.ndarray:
    xs = origin[0] + np.arange(shape[0]) * spacing[0]
    ys = origin[1] + np.arange(shape[1]) * spacing[1]
    zs = origin[2] + np.arange(shape[2]) * spacing[2]
    return (
        wind_hat[0] * xs[:, None, None]
        + wind_hat[1] * ys[None, :, None]
        + wind_hat[2] * zs[None, None, :]
    )


def downstream_shadow(solid, origin, spacing, wind_hat, steps: int = 16) -> np.ndarray:
    solid = np.asarray(solid, dtype=bool)
    if not np.any(solid):
        return np.zeros(solid.shape, dtype=np.float32)
    wind_hat = np.asarray(wind_hat, dtype=np.float64)
    wind_hat = wind_hat / (np.linalg.norm(wind_hat) + 1e-12)
    spacing = np.asarray(spacing, dtype=np.float64)
    step = wind_hat / spacing
    step = step / (np.linalg.norm(step) + 1e-12)
    acc = solid.astype(np.float64)
    shadow = np.zeros_like(acc)
    for _ in range(int(steps)):
        acc = _shift_zero(acc, step)
        acc = _blur(acc) * 0.92
        np.maximum(shadow, acc, out=shadow)
    shadow[solid] = 0.0
    coord = _wind_coord(solid.shape, np.asarray(origin, dtype=np.float64), spacing, wind_hat)
    front = float(np.quantile(coord[solid], 0.35))
    shadow[coord < front] = 0.0
    shadow[shadow < 0.05] = 0.0
    peak = float(shadow.max())
    if peak > 1e-6:
        shadow /= peak
    return shadow.astype(np.float32)


def _deriv(phi: np.ndarray, solid: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    plus = np.roll(phi, -1, axis=axis)
    minus = np.roll(phi, 1, axis=axis)
    plus = np.where(np.roll(solid, -1, axis=axis), phi, plus)
    minus = np.where(np.roll(solid, 1, axis=axis), phi, minus)
    deriv = (plus - minus) / (2.0 * spacing)

    left = [slice(None)] * 3
    left_in = [slice(None)] * 3
    left[axis] = 0
    left_in[axis] = 1
    left_t = tuple(left)
    left_in_t = tuple(left_in)
    neighbor = np.where(solid[left_in_t], phi[left_t], phi[left_in_t])
    deriv[left_t] = (neighbor - phi[left_t]) / spacing

    right = [slice(None)] * 3
    right_in = [slice(None)] * 3
    right[axis] = -1
    right_in[axis] = -2
    right_t = tuple(right)
    right_in_t = tuple(right_in)
    neighbor = np.where(solid[right_in_t], phi[right_t], phi[right_in_t])
    deriv[right_t] = (phi[right_t] - neighbor) / spacing
    deriv[solid] = 0.0
    return deriv


def _solve_once(solid, origin, spacing, wind, iterations: int, omega: float):
    shape = solid.shape
    xs = origin[0] + np.arange(shape[0]) * spacing[0]
    ys = origin[1] + np.arange(shape[1]) * spacing[1]
    zs = origin[2] + np.arange(shape[2]) * spacing[2]
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
    freestream = wind[0] * x + wind[1] * y + wind[2] * z
    phi = freestream.copy()
    boundary = np.zeros(shape, dtype=bool)
    boundary[0] = True
    boundary[-1] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    boundary[:, :, 0] = True
    boundary[:, :, -1] = True
    update = (~solid) & (~boundary)
    for step in range(int(iterations)):
        xm = np.where(np.roll(solid, 1, 0), phi, np.roll(phi, 1, 0))
        xp = np.where(np.roll(solid, -1, 0), phi, np.roll(phi, -1, 0))
        ym = np.where(np.roll(solid, 1, 1), phi, np.roll(phi, 1, 1))
        yp = np.where(np.roll(solid, -1, 1), phi, np.roll(phi, -1, 1))
        zm = np.where(np.roll(solid, 1, 2), phi, np.roll(phi, 1, 2))
        zp = np.where(np.roll(solid, -1, 2), phi, np.roll(phi, -1, 2))
        average = (xm + xp + ym + yp + zm + zp) / 6.0
        nxt = (1.0 - omega) * phi + omega * average
        if step % 8 == 7 and np.max(np.abs(nxt - phi)[update]) < 1e-4:
            phi = np.where(update, nxt, phi)
            phi[boundary] = freestream[boundary]
            break
        phi = np.where(update, nxt, phi)
        phi[boundary] = freestream[boundary]
        if not np.isfinite(phi).all() or float(np.max(np.abs(phi))) > 1e5:
            return None
    velocity = np.empty(shape + (3,), dtype=np.float64)
    velocity[..., 0] = _deriv(phi, solid, float(spacing[0]), 0)
    velocity[..., 1] = _deriv(phi, solid, float(spacing[1]), 1)
    velocity[..., 2] = _deriv(phi, solid, float(spacing[2]), 2)
    velocity[solid] = 0.0
    if not np.isfinite(velocity).all():
        return None
    return velocity


def solve_potential(solid, origin, spacing, wind, iterations: int = 60, omega: float = 1.45):
    solid = np.asarray(solid, dtype=bool)
    origin = np.asarray(origin, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    wind = np.asarray(wind, dtype=np.float64)
    velocity = _solve_once(solid, origin, spacing, wind, iterations, omega)
    if velocity is None:
        velocity = _solve_once(solid, origin, spacing, wind, iterations, 1.0)
    if velocity is None:
        velocity = np.zeros(solid.shape + (3,), dtype=np.float64)
        velocity[..., 0] = wind[0]
        velocity[..., 1] = wind[1]
        velocity[..., 2] = wind[2]
        velocity[solid] = 0.0
    return velocity


def in_domain(points, origin, spacing, shape) -> np.ndarray:
    idx = (np.asarray(points, dtype=np.float64) - origin) / spacing
    upper = np.asarray(shape, dtype=np.float64) - 1.0
    return np.all((idx >= 0.0) & (idx <= upper), axis=1)


def solid_mask_at(points, solid, origin, spacing) -> np.ndarray:
    idx = np.rint((np.asarray(points, dtype=np.float64) - origin) / spacing).astype(np.int32)
    shape = np.asarray(solid.shape, dtype=np.int32)
    inside = np.all((idx >= 0) & (idx < shape), axis=1)
    flags = np.zeros(len(idx), dtype=bool)
    if np.any(inside):
        safe = np.clip(idx, 0, shape - 1)
        picked = safe[inside]
        flags[inside] = solid[picked[:, 0], picked[:, 1], picked[:, 2]]
    return flags


def sample_velocity(velocity, origin, spacing, points, wind, solid=None) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    wind = np.asarray(wind, dtype=np.float64)
    nx, ny, nz, _ = velocity.shape
    idx = (points - origin) / spacing
    x = idx[:, 0]
    y = idx[:, 1]
    z = idx[:, 2]
    x0 = np.clip(np.floor(x).astype(np.int32), 0, nx - 2)
    y0 = np.clip(np.floor(y).astype(np.int32), 0, ny - 2)
    z0 = np.clip(np.floor(z).astype(np.int32), 0, nz - 2)
    xd = np.clip(x - x0, 0.0, 1.0)[:, None]
    yd = np.clip(y - y0, 0.0, 1.0)[:, None]
    zd = np.clip(z - z0, 0.0, 1.0)[:, None]
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1
    c000 = velocity[x0, y0, z0]
    c100 = velocity[x1, y0, z0]
    c010 = velocity[x0, y1, z0]
    c110 = velocity[x1, y1, z0]
    c001 = velocity[x0, y0, z1]
    c101 = velocity[x1, y0, z1]
    c011 = velocity[x0, y1, z1]
    c111 = velocity[x1, y1, z1]
    c00 = c000 * (1.0 - xd) + c100 * xd
    c01 = c001 * (1.0 - xd) + c101 * xd
    c10 = c010 * (1.0 - xd) + c110 * xd
    c11 = c011 * (1.0 - xd) + c111 * xd
    c0 = c00 * (1.0 - yd) + c10 * yd
    c1 = c01 * (1.0 - yd) + c11 * yd
    sampled = c0 * (1.0 - zd) + c1 * zd
    outside = ~in_domain(points, origin, spacing, (nx, ny, nz))
    if np.any(outside):
        sampled = sampled.copy()
        sampled[outside] = wind
    if solid is not None:
        blocked = solid_mask_at(points, solid, origin, spacing)
        if np.any(blocked):
            sampled = sampled.copy()
            sampled[blocked] = 0.0
    return sampled


def _outside_speed(points, normals, velocity, solid, origin, spacing, wind) -> np.ndarray:
    """Fastest air along the outward normal, just outside the blocked cells."""
    dx = float(np.mean(spacing))
    best = np.zeros(len(points), dtype=np.float64)
    found = np.zeros(len(points), dtype=bool)
    for scale in (1.6, 2.4, 3.4, 4.6):
        sample_points = points + normals * (scale * dx)
        vel = sample_velocity(velocity, origin, spacing, sample_points, wind, solid)
        speed = np.linalg.norm(vel, axis=1)
        ok = in_domain(sample_points, origin, spacing, solid.shape) & ~solid_mask_at(
            sample_points, solid, origin, spacing
        )
        take = ok & (~found | (speed > best))
        best[take] = speed[take]
        found |= ok
    return best


def _rear_cp(cp, outward, wind_hat, wake: float) -> np.ndarray:
    rear = np.clip(outward @ wind_hat, 0.0, 1.0)
    return np.clip(cp - 0.85 * float(wake) * rear, -3.5, 1.2)


def sample_vector_glyphs(velocity, solid, origin, spacing, wind_speed, mode: str, stride: int):
    nx, ny, nz, _ = velocity.shape
    stride = max(1, int(stride))
    xs = np.arange(0, nx, stride)
    ys = np.arange(0, ny, stride)
    zs = np.arange(0, nz, stride)
    if mode == "side":
        ys = np.array([ny // 2])
    elif mode == "top":
        zs = np.array([nz // 2])
    elif mode == "front":
        xs = np.array([nx // 2])
    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    xi = grid_x.ravel()
    yi = grid_y.ravel()
    zi = grid_z.ravel()
    blocked = solid[xi, yi, zi]
    vec = velocity[xi, yi, zi]
    speed = np.linalg.norm(vec, axis=1)
    keep = (~blocked) & (speed > 0.05 * max(float(wind_speed), 1e-6))
    xi, yi, zi, vec, speed = xi[keep], yi[keep], zi[keep], vec[keep], speed[keep]
    if len(xi) > 1800:
        step = int(np.ceil(len(xi) / 1800))
        xi, yi, zi = xi[::step], yi[::step], zi[::step]
        vec, speed = vec[::step], speed[::step]
    origin = np.asarray(origin, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    points = np.column_stack(
        [
            origin[0] + xi * spacing[0],
            origin[1] + yi * spacing[1],
            origin[2] + zi * spacing[2],
        ]
    )
    ratio = speed / max(float(wind_speed), 1e-6)
    direction = vec / (speed[:, None] + 1e-9)
    length = ratio * float(np.mean(spacing)) * max(stride, 1) * 1.45
    return points, direction, length, ratio


def trace_streamlines(velocity, solid, origin, spacing, wind, steps: int = 110):
    wind = np.asarray(wind, dtype=np.float64)
    speed = float(np.linalg.norm(wind))
    if speed < 1e-8:
        return np.zeros((0, steps, 3), np.float32), np.zeros((0, steps), np.float32)
    hat = wind / speed
    origin = np.asarray(origin, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    shape = velocity.shape[:3]
    half = spacing * (np.array(shape) - 1) * 0.5
    center = origin + half
    right, up = flow_basis(hat)
    reach = float(np.min(half))
    inlet = center - hat * (reach * 0.8)
    # Keep seeds in the center plane so the side view reads as one airflow picture.
    across = np.linspace(-0.03, 0.03, 2) * reach
    vertical = np.linspace(-0.42, 0.52, 8) * reach
    rr, uu = np.meshgrid(across, vertical, indexing="ij")
    seeds = inlet + rr.ravel()[:, None] * right + uu.ravel()[:, None] * up
    keep = in_domain(seeds, origin, spacing, shape) & ~solid_mask_at(seeds, solid, origin, spacing)
    seeds = seeds[keep]
    if len(seeds) == 0:
        return np.zeros((0, steps, 3), np.float32), np.zeros((0, steps), np.float32)
    dt = 0.48 * float(np.mean(spacing)) / speed
    cap = 0.42 * float(np.mean(spacing))
    paths = np.empty((len(seeds), steps, 3), dtype=np.float32)
    speeds = np.empty((len(seeds), steps), dtype=np.float32)
    pos = seeds.copy()
    for frame in range(steps):
        vel = sample_velocity(velocity, origin, spacing, pos, wind, solid)
        mag = np.linalg.norm(vel, axis=1)
        paths[:, frame] = pos
        speeds[:, frame] = mag
        step = vel * dt
        scale = np.minimum(1.0, cap / np.maximum(mag * dt, 1e-12))
        pos = pos + step * scale[:, None]
    return paths, speeds


def grid_for_mesh(mesh, n: int):
    bounds = np.array(mesh.bounds, dtype=np.float64).reshape(3, 2)
    center = (bounds[:, 0] + bounds[:, 1]) * 0.5
    half = (bounds[:, 1] - bounds[:, 0]) * 0.5
    reach = float(np.max(half) + 0.8)
    origin = center - reach
    step = (2.0 * reach) / (n - 1)
    spacing = np.array([step, step, step], dtype=np.float64)
    return origin, spacing


def _vtk_distances(mesh, points: np.ndarray) -> np.ndarray:
    import pyvista as pv

    cloud = pv.PolyData(np.ascontiguousarray(points, dtype=np.float64))
    sampled = cloud.compute_implicit_distance(mesh)
    return np.asarray(sampled.point_data["implicit_distance"], dtype=np.float64).reshape(-1)


def _stamp(shell: np.ndarray, points: np.ndarray, origin, spacing) -> None:
    if len(points) == 0:
        return
    idx = np.rint((points - origin) / spacing).astype(np.int32)
    shape = np.asarray(shell.shape, dtype=np.int32)
    valid = np.all((idx >= 0) & (idx < shape), axis=1)
    idx = idx[valid]
    if len(idx) == 0:
        return
    shell[idx[:, 0], idx[:, 1], idx[:, 2]] = True


def _barycentric(a, b, c, divisions: int) -> np.ndarray:
    ii, jj = np.indices((divisions + 1, divisions + 1))
    mask = (ii + jj) <= divisions
    i = ii[mask].astype(np.float64) / divisions
    j = jj[mask].astype(np.float64) / divisions
    w = 1.0 - i - j
    pts = (
        w[None, :, None] * a[:, None, :]
        + i[None, :, None] * b[:, None, :]
        + j[None, :, None] * c[:, None, :]
    )
    return np.ascontiguousarray(pts.reshape(-1, 3))


def _splat_surface(shell, mesh, origin, spacing) -> None:
    faces = np.asarray(mesh.faces)
    if faces.size < 4:
        return
    faces = faces.reshape(-1, 4)[:, 1:]
    pts = np.asarray(mesh.points, dtype=np.float64)
    a = pts[faces[:, 0]]
    b = pts[faces[:, 1]]
    c = pts[faces[:, 2]]
    step = float(np.min(spacing)) * 0.48
    longest = np.maximum(
        np.linalg.norm(b - a, axis=1),
        np.maximum(np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)),
    )
    divisions = np.clip(np.ceil(longest / max(step, 1e-6)).astype(np.int32), 1, 32)
    _stamp(shell, pts, origin, spacing)
    _stamp(shell, (a + b + c) / 3.0, origin, spacing)
    for count in sorted(set(divisions.tolist())):
        chosen = divisions == count
        if not np.any(chosen):
            continue
        _stamp(shell, _barycentric(a[chosen], b[chosen], c[chosen], int(count)), origin, spacing)


def _solids_from_shell(shell: np.ndarray) -> np.ndarray:
    air = np.zeros(shell.shape, dtype=bool)
    air[0] = ~shell[0]
    air[-1] = ~shell[-1]
    air[:, 0] = air[:, 0] | ~shell[:, 0]
    air[:, -1] = air[:, -1] | ~shell[:, -1]
    air[:, :, 0] = air[:, :, 0] | ~shell[:, :, 0]
    air[:, :, -1] = air[:, :, -1] | ~shell[:, :, -1]
    air[shell] = False
    limit = 4 * max(shell.shape)
    for _ in range(limit):
        padded = np.pad(air, 1, mode="constant", constant_values=False)
        grown = (
            padded[1:-1, 1:-1, 1:-1]
            | padded[:-2, 1:-1, 1:-1]
            | padded[2:, 1:-1, 1:-1]
            | padded[1:-1, :-2, 1:-1]
            | padded[1:-1, 2:, 1:-1]
            | padded[1:-1, 1:-1, :-2]
            | padded[1:-1, 1:-1, 2:]
        )
        grown[shell] = False
        if np.array_equal(grown, air):
            break
        air = grown
    return ~air


def _decimate_solver_mesh(mesh):
    if mesh.n_cells <= 80000:
        return mesh.triangulate().clean()
    reduction = float(min(0.98, max(0.0, 1.0 - 80000 / float(mesh.n_cells))))
    try:
        smaller = mesh.decimate(reduction)
    except Exception:
        smaller = mesh
    if smaller is None or smaller.n_cells < 1:
        smaller = mesh
    return smaller.triangulate().clean()


def compute_occupancy(mesh, n: int) -> dict:
    """Mark grid cells that the model blocks. Thin and open shapes still block air."""
    n = int(n)
    origin, spacing = grid_for_mesh(mesh, n)
    xs = origin[0] + np.arange(n) * spacing[0]
    ys = origin[1] + np.arange(n) * spacing[1]
    zs = origin[2] + np.arange(n) * spacing[2]
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
    query = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    work = _decimate_solver_mesh(mesh)
    band = 0.72 * float(np.min(spacing))
    shell = np.zeros((n, n, n), dtype=bool)
    try:
        dist = _vtk_distances(work, query).reshape((n, n, n))
        near = np.abs(dist) < band
        if np.isfinite(dist).all() and not np.allclose(dist, 0.0) and float(near.mean()) < 0.35:
            shell = near
    except Exception:
        shell = np.zeros((n, n, n), dtype=bool)
    splat = np.zeros((n, n, n), dtype=bool)
    _splat_surface(splat, work, origin, spacing)
    shell = shell | splat
    if float(shell.mean()) > 0.45:
        shell = splat
    solid = _solids_from_shell(shell)
    return {
        "solid": solid,
        "origin": origin,
        "spacing": spacing,
        "shape": solid.shape,
    }


def compute_flow(
    solid,
    origin,
    spacing,
    wind,
    wake,
    rho,
    points,
    point_normals,
    cell_centers,
    cell_normals,
    cell_areas,
    iterations=60,
):
    solid = np.asarray(solid, dtype=bool)
    origin = np.asarray(origin, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    wind = np.asarray(wind, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    point_normals = np.asarray(point_normals, dtype=np.float64)
    cell_centers = np.asarray(cell_centers, dtype=np.float64)
    cell_normals = np.asarray(cell_normals, dtype=np.float64)
    cell_areas = np.asarray(cell_areas, dtype=np.float64)
    speed = float(np.linalg.norm(wind))
    if speed < 1e-6:
        wind = np.array([1.0, 0.0, 0.0])
        speed = 1.0
    wind_hat = wind / speed
    potential = solve_potential(solid, origin, spacing, wind, iterations=iterations)

    dx = float(np.mean(spacing)) * 2.4
    plus = sample_velocity(potential, origin, spacing, points + point_normals * dx, wind, solid)
    minus = sample_velocity(potential, origin, spacing, points - point_normals * dx, wind, solid)
    sign = -1.0 if np.linalg.norm(minus, axis=1).mean() > np.linalg.norm(plus, axis=1).mean() else 1.0
    outward_points = point_normals * sign
    outward_cells = cell_normals * sign

    point_speed = _outside_speed(points, outward_points, potential, solid, origin, spacing, wind)
    cp_points = np.clip(1.0 - (point_speed / speed) ** 2, -4.0, 1.2)
    cp_points = _rear_cp(cp_points, outward_points, wind_hat, wake)
    cp_points = np.nan_to_num(cp_points, nan=0.0, posinf=1.2, neginf=-3.5)

    cell_speed = _outside_speed(cell_centers, outward_cells, potential, solid, origin, spacing, wind)
    cp_cells = np.clip(1.0 - (cell_speed / speed) ** 2, -4.0, 1.2)
    cp_cells = _rear_cp(cp_cells, outward_cells, wind_hat, wake)
    cp_cells = np.nan_to_num(cp_cells, nan=0.0, posinf=1.2, neginf=-3.5)

    dynamic = 0.5 * float(rho) * speed * speed
    force = np.sum((-cp_cells[:, None] * outward_cells) * cell_areas[:, None], axis=0) * dynamic
    up = np.array([0.0, 0.0, 1.0])
    lift_dir = up - wind_hat * float(np.dot(up, wind_hat))
    if np.linalg.norm(lift_dir) < 1e-8:
        lift_dir = np.array([0.0, 1.0, 0.0])
    else:
        lift_dir = lift_dir / np.linalg.norm(lift_dir)
    side_dir = np.cross(wind_hat, lift_dir)
    side_dir = side_dir / (np.linalg.norm(side_dir) + 1e-12)
    drag = float(np.dot(force, wind_hat))
    lift = float(np.dot(force, lift_dir))
    side = float(np.dot(force, side_dir))
    area_wind = 0.5 * float(np.sum(np.abs(outward_cells @ wind_hat) * cell_areas))
    area_up = 0.5 * float(np.sum(np.abs(outward_cells @ lift_dir) * cell_areas))
    area_ref = max(area_wind, area_up, 1e-8)
    shown = np.linalg.norm(potential, axis=-1)
    fluid = ~solid
    peak = float(shown[fluid].max() / speed) if np.any(fluid) else 0.0

    shadow = downstream_shadow(solid, origin, spacing, wind_hat)
    factor = np.clip(1.0 - float(wake) * shadow, 0.32, 1.0)
    factor[solid] = 0.0
    velocity = (potential * factor[..., None]).astype(np.float32)

    report = {
        "lift": lift,
        "drag": drag,
        "side": side,
        "cl": lift / (dynamic * area_ref),
        "cd": drag / (dynamic * area_ref),
        "peak": peak,
        "area": area_ref,
    }
    return {
        "velocity": velocity,
        "solid": solid,
        "origin": origin,
        "spacing": spacing,
        "wind_hat": wind_hat.astype(np.float64),
        "wind_speed": speed,
        "cp": cp_points.astype(np.float64),
        "report": report,
    }
