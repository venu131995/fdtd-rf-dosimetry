"""Reference dipole above a flat phantom (the IEEE 1528 system-validation set-up) at 900 MHz.

Thin-wire dipole along y (a line of zero Ey edges, length ~149 mm) with a one-cell gap feed in which Ey is set to the
source voltage over the gap. The dipole axis is 15 mm above the liquid surface, with a 2 mm low-loss shell between.
Liquid: head target eps_r 41.5, sigma 0.97 S/m (3GPP TS 34.114), density 1000 kg/m3, 0.6 x 0.4 wavelengths wide and
150 mm deep. The feed current is the loop integral of H around the feed edge; SAR is scaled to 1 W accepted power and
averaged over 1 g and 10 g cubes that lie entirely in the liquid.
"""
import numpy as np

from .sources import Phasor
from .sphere import centre_fields
from .yee import C0, Grid


def run(dx, f=900e6, length=0.149, spacing=0.015, shell=0.002, eps_shell=3.7, eps_liq=41.5, sig_liq=0.97,
        depth=0.150, margin=0.04, npml=10, periods=20, avg_periods=3, verbose=False):
    lam = C0 / f
    Lx, Ly = 0.4 * lam, 0.6 * lam
    nx = int(np.ceil((Lx + 2 * margin) / dx)) + 2 * npml
    ny = int(np.ceil((Ly + 2 * margin) / dx)) + 2 * npml
    nz = int(np.ceil((depth + shell + spacing + 2 * margin) / dx)) + 2 * npml
    g = Grid((nx, ny, nz), dx, npml)
    ic, jc = nx // 2, ny // 2
    k_liq_top = npml + int(round(margin / dx)) + int(round(depth / dx))       # first cell index above the liquid
    k_shell = int(round(shell / dx))
    k_dip = k_liq_top + int(round(spacing / dx))                              # E-node plane of the dipole axis
    hx, hy = int(round(Lx / dx / 2)), int(round(Ly / dx / 2))
    liq = (slice(ic - hx, ic + hx), slice(jc - hy, jc + hy), slice(k_liq_top - int(round(depth / dx)), k_liq_top))
    g.eps[liq] = eps_liq
    g.sig[liq] = sig_liq
    g.eps[ic - hx:ic + hx, jc - hy:jc + hy, k_liq_top:k_liq_top + k_shell] = eps_shell
    g.edge_coeffs()
    half = int(round(length / dx / 2))
    wire = (ic, slice(jc - half, jc + half + 1), k_dip)        # Ey edges from jc-half .. jc+half (feed at jc)
    w = 2 * np.pi * f
    ramp = 3.0 / f
    ph = Phasor(w, ["Ex", "Ey", "Ez"])
    vph, iph = 0.0 + 0j, 0.0 + 0j
    T = 1.0 / f
    n_steps = int(periods * T / g.dt)
    start = n_steps - int(avg_periods * T / g.dt)
    t = 0.0
    for s in range(n_steps):
        g.update_h()
        g.update_e()
        t += g.dt
        r = min(1.0, t / ramp)
        v = r * r * (3 - 2 * r) * np.sin(w * t)                # 1 V amplitude
        g.Ey[wire] = 0.0
        g.Ey[ic, jc, k_dip] = -v / dx
        if s >= start:
            ph.add(g, t)
            # feed current: loop of H around the Ey edge at (ic, jc+1/2, k_dip); H at t - dt/2, phase-corrected below
            # Ampere: I_y = loop integral = dx (dHx/dz - dHz/dx) around the edge; positive along +y, the direction in
            # which the source drives current through the gap when it delivers power
            i_loop = dx * (g.Hx[ic, jc, k_dip] - g.Hx[ic, jc, k_dip - 1] - g.Hz[ic, jc, k_dip] + g.Hz[ic - 1, jc, k_dip])
            e = np.exp(1j * w * (t - 0.5 * g.dt)) * g.dt
            iph += i_loop * e
            vph += v * np.exp(1j * w * t) * g.dt
        if verbose and s % 500 == 0:
            print(s, n_steps, flush=True)
    Tacc = ph.T
    V, I = 2 * vph / Tacc, 2 * iph / Tacc
    Z = V / I
    P = 0.5 * np.real(V * np.conj(I))
    E = centre_fields(g, ph.get("Ex"), ph.get("Ey"), ph.get("Ez"))
    sar = g.sig * np.sum(np.abs(E) ** 2, -1) / (2 * 1000.0) / P            # W/kg per W accepted
    sar_liq = np.where(g.sig > 0, sar, np.nan)
    out = dict(dx_mm=1e3 * dx, cells=nx * ny * nz, steps=n_steps, Z_in=[float(Z.real), float(Z.imag)], P_acc_per_V2=float(P))
    for mass, label in ((1e-3, "1g"), (1e-2, "10g")):
        side = (mass / 1000.0) ** (1 / 3)
        nc = max(1, int(round(side / dx)))
        out[f"psSAR_{label}"] = float(cube_average_max(sar_liq, nc))
        out[f"cube_{label}_mm"] = 1e3 * nc * dx
    out["peak_sar"] = float(np.nanmax(sar_liq))
    return out


def cube_average_max(field, n):
    """Maximum over all n x n x n cubes lying entirely in the medium (NaN outside) of the cube mean."""
    a = np.nan_to_num(field, nan=0.0)
    m = np.isfinite(field).astype(float)
    def box(x):
        c = x.cumsum(0).cumsum(1).cumsum(2)
        c = np.pad(c, ((1, 0), (1, 0), (1, 0)))
        return (c[n:, n:, n:] - c[:-n, n:, n:] - c[n:, :-n, n:] - c[n:, n:, :-n]
                + c[:-n, :-n, n:] + c[:-n, n:, :-n] + c[n:, :-n, :-n] - c[:-n, :-n, :-n])
    s, cnt = box(a), box(m)
    full = cnt >= n ** 3 - 0.5
    return (s[full] / n ** 3).max()
