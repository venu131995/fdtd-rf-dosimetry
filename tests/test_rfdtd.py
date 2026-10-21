import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rfdtd import mie  # noqa: E402
from rfdtd.dipole import cube_average_max  # noqa: E402
from rfdtd.sources import Phasor, PlaneWave  # noqa: E402
from rfdtd.sphere import centre_fields  # noqa: E402
from rfdtd.yee import C0, EPS0, ETA0, Grid  # noqa: E402


def test_mie_small_sphere_limit():
    k = 2 * np.pi * 1e6 / C0
    E = mie.internal_field(np.array([[0.0, 0.0, 0.0], [1e-5, 2e-5, -1e-5]]), 1e-4, k, np.sqrt(4.0 + 0j))
    assert np.allclose(np.abs(E[:, 0]), 3 / (4 + 2), rtol=1e-6) and np.allclose(np.abs(E[:, 1:]), 0, atol=1e-6)


def test_mie_absorption_equals_internal_dissipation():
    f, a, er, s = 900e6, 0.05, 41.5, 0.97
    w = 2 * np.pi * f
    k = w / C0
    m = np.sqrt(er + 1j * s / (w * EPS0))
    _, _, cabs = mie.cross_sections(k * a, m, k)
    # midpoint rule in spherical coordinates
    nr, nt, npz = 40, 40, 40
    r = (np.arange(nr) + 0.5) / nr * a
    th = (np.arange(nt) + 0.5) / nt * np.pi
    ph = (np.arange(npz) + 0.5) / npz * 2 * np.pi
    R, T, P = np.meshgrid(r, th, ph, indexing="ij")
    pts = np.column_stack([(R * np.sin(T) * np.cos(P)).ravel(), (R * np.sin(T) * np.sin(P)).ravel(), (R * np.cos(T)).ravel()])
    E = mie.internal_field(pts, a, k, m)
    dV = (R ** 2 * np.sin(T)).ravel() * (a / nr) * (np.pi / nt) * (2 * np.pi / npz)
    p = 0.5 * s * np.sum(np.sum(np.abs(E) ** 2, 1) * dV)
    assert abs(p / (cabs / (2 * ETA0)) - 1) < 0.01


def test_tfsf_plane_wave_stays_inside_the_box():
    n, dx, f = 50, 0.01, 1e9
    g = Grid((n, n, n), dx, npml=8)
    g.edge_coeffs()
    b0, b1 = 12, 28
    pw = PlaneWave(g, (b0, b1, b0, b1, b0, b1), f)
    ph = Phasor(2 * np.pi * f, ["Ex", "Ey", "Ez"])
    steps = int(8 / f / g.dt)
    for s in range(steps):
        pw.step_1d_h(); g.update_h(); pw.correct_h(); g.update_e(); pw.correct_e(); pw.step_1d_e()
        if s >= steps - int(2 / f / g.dt):
            ph.add(g, pw.t)
    E = centre_fields(g, ph.get("Ex"), ph.get("Ey"), ph.get("Ez"))
    inside = np.abs(E[b0 + 2:b1 - 2, b0 + 2:b1 - 2, b0 + 2:b1 - 2, 0]).mean()
    outside = np.abs(E[b1 + 3:n - 9, b0:b1, b0:b1]).max()
    assert abs(inside - 1.0) < 0.02 and outside < 0.01


def test_cube_average_respects_medium_boundary():
    f = np.full((10, 10, 10), np.nan)
    f[2:8, 2:8, 2:8] = 1.0
    f[4, 4, 4] = 28.0
    # 3-cubes fully inside the medium: the best one contains the hot cell -> (26 + 28) / 27
    assert np.isclose(cube_average_max(f, 3), (26 + 28) / 27)
