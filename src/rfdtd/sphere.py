"""FDTD of a plane wave on a homogeneous lossy sphere, for comparison with Mie theory."""
import numpy as np

from .sources import Phasor, PlaneWave
from .yee import Grid


def centre_fields(g, Ex, Ey, Ez):
    """Average the edge phasors onto cell centres (i+1/2, j+1/2, k+1/2)."""
    ex = 0.25 * (Ex + np.roll(Ex, -1, 1) + np.roll(Ex, -1, 2) + np.roll(np.roll(Ex, -1, 1), -1, 2))
    ey = 0.25 * (Ey + np.roll(Ey, -1, 0) + np.roll(Ey, -1, 2) + np.roll(np.roll(Ey, -1, 0), -1, 2))
    ez = 0.25 * (Ez + np.roll(Ez, -1, 0) + np.roll(Ez, -1, 1) + np.roll(np.roll(Ez, -1, 0), -1, 1))
    return np.stack([ex, ey, ez], -1)


def run(f, a_r, eps_r, sigma, dx, margin=0.035, npml=10, periods=25, avg_periods=3, verbose=False):
    n = int(np.ceil((2 * a_r + 2 * margin) / dx)) + 2 * npml
    g = Grid((n, n, n), dx, npml)
    c = (n / 2.0) * dx                                        # sphere centre (a cell corner)
    idx = (np.arange(n) + 0.5) * dx - c                       # cell-centre coordinates
    X, Y, Z = np.meshgrid(idx, idx, idx, indexing="ij")
    inside = X ** 2 + Y ** 2 + Z ** 2 <= a_r ** 2
    g.eps[inside] = eps_r
    g.sig[inside] = sigma
    g.edge_coeffs()
    b0, b1 = npml + 3, n - npml - 4
    pw = PlaneWave(g, (b0, b1, b0, b1, b0, b1), f)
    ph = Phasor(2 * np.pi * f, ["Ex", "Ey", "Ez"])
    T = 1.0 / f
    n_steps = int(periods * T / g.dt)
    start = n_steps - int(avg_periods * T / g.dt)
    for s in range(n_steps):
        pw.step_1d_h()
        g.update_h()
        pw.correct_h()
        g.update_e()
        pw.correct_e()
        pw.step_1d_e()
        if s >= start:
            ph.add(g, pw.t)
        if verbose and s % 500 == 0:
            print(s, n_steps, flush=True)
    E = centre_fields(g, ph.get("Ex"), ph.get("Ey"), ph.get("Ez"))
    pts = np.column_stack([X[inside], Y[inside], Z[inside]])
    p_abs = 0.5 * sigma * np.sum(np.abs(E[inside]) ** 2) * dx ** 3
    return dict(pts=pts, E=E[inside], p_abs=p_abs, n=n, steps=n_steps, cells_inside=int(inside.sum()))
