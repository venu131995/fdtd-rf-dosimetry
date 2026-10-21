"""Total-field / scattered-field plane wave (x-polarised, travelling +z) and phasor extraction.

The incident field comes from a 1-D Yee grid with the same dx and dt, so its numerical dispersion matches the 3-D grid
along z exactly. Index conventions follow yee.py: Ex[i,j,k] at (i+1/2, j, k), Hy[i,j,k] at (i+1/2, j, k+1/2),
Hz[i,j,k] at (i+1/2, j+1/2, k), Ez[i,j,k] at (i, j, k+1/2). The total-field box spans E nodes i0..i1, j0..j1,
k0..k1; the corrections below are the standard consistency terms on its six faces.
"""
import numpy as np

from .yee import EPS0, MU0


class PlaneWave:
    def __init__(self, g, box, f, E0=1.0, ramp_periods=3.0):
        self.g, self.box, self.w, self.E0 = g, box, 2 * np.pi * f, E0
        self.ramp = ramp_periods / f
        self.n1 = g.nz + 3000
        self.e = np.zeros(self.n1)
        self.h = np.zeros(self.n1)
        self.off = 5                      # 1-D index of 3-D z index 0 is off
        self.t = 0.0

    def _src(self):
        r = min(1.0, self.t / self.ramp)
        r = r * r * (3 - 2 * r)           # smooth ramp
        return self.E0 * r * np.sin(self.w * self.t)

    def step_1d_h(self):
        g = self.g
        self.h[:-1] -= g.dt / (MU0 * g.dx) * (self.e[1:] - self.e[:-1])

    def step_1d_e(self):
        g = self.g
        self.e[1:] -= g.dt / (EPS0 * g.dx) * (self.h[1:] - self.h[:-1])
        self.e[0] = self._src()
        self.t += g.dt

    def einc(self, k):
        return self.e[k + self.off]

    def hinc(self, k):
        return self.h[k + self.off]

    def correct_h(self):
        g = self.g
        i0, i1, j0, j1, k0, k1 = self.box
        kc = g.dt / (MU0 * g.dx)
        g.Hy[i0:i1, j0:j1 + 1, k0 - 1] += kc * self.einc(k0)
        g.Hy[i0:i1, j0:j1 + 1, k1] -= kc * self.einc(k1)
        ks = np.arange(k0, k1 + 1)
        g.Hz[i0:i1, j0 - 1, k0:k1 + 1] -= kc * self.einc(ks)[None, :]
        g.Hz[i0:i1, j1, k0:k1 + 1] += kc * self.einc(ks)[None, :]

    def correct_e(self):
        g = self.g
        i0, i1, j0, j1, k0, k1 = self.box
        g.Ex[i0:i1, j0:j1 + 1, k0] += g.cb["x"][i0:i1, j0:j1 + 1, k0] * self.hinc(k0 - 1)
        g.Ex[i0:i1, j0:j1 + 1, k1] -= g.cb["x"][i0:i1, j0:j1 + 1, k1] * self.hinc(k1)
        ks = np.arange(k0, k1)
        g.Ez[i0, j0:j1 + 1, k0:k1] -= g.cb["z"][i0, j0:j1 + 1, k0:k1] * self.hinc(ks)[None, :]
        g.Ez[i1, j0:j1 + 1, k0:k1] += g.cb["z"][i1, j0:j1 + 1, k0:k1] * self.hinc(ks)[None, :]


class Phasor:
    """Running DFT at angular frequency w: X = (2/T_acc) sum x(t) exp(i w t) dt, so x(t) = Re(X exp(-i w t))."""

    def __init__(self, w, names):
        self.w = w
        self.acc = {n: 0.0 for n in names}
        self.T = 0.0

    def add(self, g, t):
        ph = np.exp(1j * self.w * t) * g.dt
        for n in self.acc:
            self.acc[n] = self.acc[n] + getattr(g, n) * ph
        self.T += g.dt

    def get(self, n):
        return 2.0 * self.acc[n] / self.T
