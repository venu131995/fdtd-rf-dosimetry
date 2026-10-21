"""3-D finite-difference time-domain solver (Yee 1966) for lossy, non-magnetic media.

* Uniform cubic grid, cell size dx, Courant number 0.99/sqrt(3).
* Field arrays carry the usual staggering: E components on cell edges, H on face centres. Every array is allocated
  with the full (nx, ny, nz) shape for simplicity; the last plane of a staggered component is unused.
* Lossy update (semi-implicit conductivity term) with eps_r and sigma averaged onto each E edge.
* Absorbing boundary: convolutional PML (Roden & Gedney 2000) with polynomial grading of sigma and kappa = 1 and
  a small CFS alpha, on all six faces.
* Sources: a total-field / scattered-field box for an x-polarised plane wave travelling in +z (1-D auxiliary grid
  with the same dispersion), or a soft voltage source in an Ez edge (dipole feed).
* Steady-state phasors by a running discrete Fourier transform over the last periods.
"""
import numpy as np

C0 = 299792458.0
EPS0 = 8.8541878128e-12
MU0 = 4e-7 * np.pi
ETA0 = np.sqrt(MU0 / EPS0)


class Grid:
    def __init__(self, shape, dx, npml=10):
        self.nx, self.ny, self.nz = shape
        self.dx = dx
        self.dt = 0.99 * dx / (C0 * np.sqrt(3.0))
        self.npml = npml
        self.eps = np.ones(shape)
        self.sig = np.zeros(shape)
        z = lambda: np.zeros(shape)  # noqa: E731
        self.Ex, self.Ey, self.Ez, self.Hx, self.Hy, self.Hz = z(), z(), z(), z(), z(), z()
        self._pml()

    # ---------------------------------------------------------------- materials
    def edge_coeffs(self):
        """Lossy update coefficients for each E component, from cell properties averaged onto the edges."""
        e, s = self.eps, self.sig

        def avg(a, axes):
            out = a.copy()
            for ax in axes:
                out = 0.5 * (out + np.roll(out, 1, axis=ax))
            return out
        self.ca, self.cb = {}, {}
        for comp, axes in (("x", (1, 2)), ("y", (0, 2)), ("z", (0, 1))):
            ee, ss = avg(e, axes), avg(s, axes)
            k = ss * self.dt / (2 * EPS0 * ee)
            self.ca[comp] = (1 - k) / (1 + k)
            self.cb[comp] = self.dt / (EPS0 * ee * self.dx) / (1 + k)

    # ---------------------------------------------------------------- CPML
    def _pml(self):
        n, m, R0 = self.npml, 3, 1e-6
        smax = -(m + 1) * np.log(R0) / (2 * ETA0 * n * self.dx)
        amax = 0.05 * 2 * np.pi * 1e9 * EPS0

        def profile(N, half):
            d = np.zeros(N)
            pos = np.arange(N) + (0.5 if half else 0.0)
            depth = np.maximum(np.maximum(n - pos, pos - (N - 1 - n)), 0.0) / n
            s = smax * depth ** m
            a = amax * (1 - depth)
            b = np.exp(-(s + a) * self.dt / EPS0)
            c = np.where(s > 0, s * (b - 1) / (s + a + 1e-30), 0.0)
            return b, c
        self.pb, self.pc = {}, {}
        for ax, N in zip("xyz", (self.nx, self.ny, self.nz)):
            for half in (False, True):
                b, c = profile(N, half)
                shp = [1, 1, 1]
                shp["xyz".index(ax)] = N
                self.pb[(ax, half)] = b.reshape(shp)
                self.pc[(ax, half)] = c.reshape(shp)
        z = lambda: np.zeros((self.nx, self.ny, self.nz))  # noqa: E731
        self.psi = {k: z() for k in ("Exy", "Exz", "Eyx", "Eyz", "Ezx", "Ezy", "Hxy", "Hxz", "Hyx", "Hyz", "Hzx", "Hzy")}

    # ---------------------------------------------------------------- updates
    @staticmethod
    def _d(a, ax, forward):
        """Difference along ax: forward a[i+1]-a[i] (for curl E -> H) or backward a[i]-a[i-1] (curl H -> E)."""
        out = np.zeros_like(a)
        sl = [slice(None)] * 3
        if forward:
            sl[ax] = slice(0, -1)
            s2 = list(sl)
            s2[ax] = slice(1, None)
            out[tuple(sl)] = a[tuple(s2)] - a[tuple(sl)]
        else:
            sl[ax] = slice(1, None)
            s2 = list(sl)
            s2[ax] = slice(0, -1)
            out[tuple(sl)] = a[tuple(sl)] - a[tuple(s2)]
        return out

    def update_h(self):
        k = self.dt / (MU0 * self.dx)
        P, b, c = self.psi, self.pb, self.pc
        dEz_y, dEy_z = self._d(self.Ez, 1, True), self._d(self.Ey, 2, True)
        dEx_z, dEz_x = self._d(self.Ex, 2, True), self._d(self.Ez, 0, True)
        dEy_x, dEx_y = self._d(self.Ey, 0, True), self._d(self.Ex, 1, True)
        P["Hxy"] = b[("y", True)] * P["Hxy"] + c[("y", True)] * dEz_y
        P["Hxz"] = b[("z", True)] * P["Hxz"] + c[("z", True)] * dEy_z
        P["Hyz"] = b[("z", True)] * P["Hyz"] + c[("z", True)] * dEx_z
        P["Hyx"] = b[("x", True)] * P["Hyx"] + c[("x", True)] * dEz_x
        P["Hzx"] = b[("x", True)] * P["Hzx"] + c[("x", True)] * dEy_x
        P["Hzy"] = b[("y", True)] * P["Hzy"] + c[("y", True)] * dEx_y
        self.Hx -= k * (dEz_y - dEy_z + P["Hxy"] - P["Hxz"])
        self.Hy -= k * (dEx_z - dEz_x + P["Hyz"] - P["Hyx"])
        self.Hz -= k * (dEy_x - dEx_y + P["Hzx"] - P["Hzy"])

    def update_e(self):
        P, b, c = self.psi, self.pb, self.pc
        dHz_y, dHy_z = self._d(self.Hz, 1, False), self._d(self.Hy, 2, False)
        dHx_z, dHz_x = self._d(self.Hx, 2, False), self._d(self.Hz, 0, False)
        dHy_x, dHx_y = self._d(self.Hy, 0, False), self._d(self.Hx, 1, False)
        P["Exy"] = b[("y", False)] * P["Exy"] + c[("y", False)] * dHz_y
        P["Exz"] = b[("z", False)] * P["Exz"] + c[("z", False)] * dHy_z
        P["Eyz"] = b[("z", False)] * P["Eyz"] + c[("z", False)] * dHx_z
        P["Eyx"] = b[("x", False)] * P["Eyx"] + c[("x", False)] * dHz_x
        P["Ezx"] = b[("x", False)] * P["Ezx"] + c[("x", False)] * dHy_x
        P["Ezy"] = b[("y", False)] * P["Ezy"] + c[("y", False)] * dHx_y
        self.Ex = self.ca["x"] * self.Ex + self.cb["x"] * (dHz_y - dHy_z + P["Exy"] - P["Exz"])
        self.Ey = self.ca["y"] * self.Ey + self.cb["y"] * (dHx_z - dHz_x + P["Eyz"] - P["Eyx"])
        self.Ez = self.ca["z"] * self.Ez + self.cb["z"] * (dHy_x - dHx_y + P["Ezx"] - P["Ezy"])
