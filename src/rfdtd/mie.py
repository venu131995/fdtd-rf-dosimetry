"""Mie theory for a homogeneous lossy sphere in an x-polarised plane wave travelling +z (Bohren & Huffman 1983,
"Absorption and Scattering of Light by Small Particles", sections 4.3-4.4, exp(-i w t) convention).

Internal field E1 = sum_n E_n (c_n M_o1n^(1) - i d_n N_e1n^(1)), E_n = i^n E0 (2n+1) / (n(n+1)), with the vector
harmonics built on j_n(m k r). Non-magnetic sphere (mu1 = mu): with x = k a and m the complex refractive index,
c_n = [j_n(x) xi_n'(x) - xi_n(x) psi_n'(x)/x ...] written below through Riccati-Bessel functions.
Also returns the absorption cross-section from a_n, b_n.
"""
import numpy as np
from scipy.special import spherical_jn, spherical_yn


def _riccati(n, z):
    j = spherical_jn(n, z)
    jd = spherical_jn(n, z, derivative=True)
    return j, z * jd + j            # z j_n(z) derivative = j + z j'


def coefficients(x, m, nmax):
    n = np.arange(1, nmax + 1)
    mx = m * x
    jx, djx = spherical_jn(n, x), spherical_jn(n, x, derivative=True)
    yx, dyx = spherical_yn(n, x), spherical_yn(n, x, derivative=True)
    hx, dhx = jx + 1j * yx, djx + 1j * dyx
    jm, djm = spherical_jn(n, mx), spherical_jn(n, mx, derivative=True)
    psi_x, dpsi_x = x * jx, jx + x * djx                   # Riccati-Bessel psi = x j_n
    xi_x, dxi_x = x * hx, hx + x * dhx                     # xi = x h_n
    psi_m, dpsi_m = mx * jm, jm + mx * djm
    a = (m * psi_m * dpsi_x - psi_x * dpsi_m) / (m * psi_m * dxi_x - xi_x * dpsi_m)
    b = (psi_m * dpsi_x - m * psi_x * dpsi_m) / (psi_m * dxi_x - m * xi_x * dpsi_m)
    # internal coefficients (B&H 4.52-4.53 with mu1 = mu)
    c = (jx * dxi_x - hx * dpsi_x) / (jm * dxi_x - hx * dpsi_m)
    d = (m * jx * dxi_x - m * hx * dpsi_x) / (m * m * jm * dxi_x - hx * dpsi_m)
    return n, a, b, c, d


def cross_sections(x, m, k, nmax=None):
    nmax = nmax or int(x + 4 * x ** (1 / 3) + 10)
    n, a, b, _, _ = coefficients(x, m, nmax)
    cext = 2 * np.pi / k ** 2 * np.sum((2 * n + 1) * np.real(a + b))
    csca = 2 * np.pi / k ** 2 * np.sum((2 * n + 1) * (np.abs(a) ** 2 + np.abs(b) ** 2))
    return cext, csca, cext - csca


def internal_field(pts, a_r, k, m, E0=1.0, nmax=None):
    """Complex Cartesian E inside the sphere (radius a_r) at points (p, 3), sphere centred at the origin."""
    x = k * a_r
    nmax = nmax or int(abs(m) * x + 4 * (abs(m) * x) ** (1 / 3) + 15)
    n, _, _, c, d = coefficients(x, m, nmax)
    r = np.linalg.norm(pts, axis=1)
    r = np.maximum(r, 1e-12)
    ct = np.clip(pts[:, 2] / r, -1, 1)
    st = np.sqrt(1 - ct ** 2)
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    cp, sp = np.cos(phi), np.sin(phi)
    rho = m * k * r
    # angular functions pi_n, tau_n (B&H 4.47)
    P = np.zeros((nmax + 1, len(r)))
    T = np.zeros((nmax + 1, len(r)))
    P[1] = 1.0
    for nn in range(2, nmax + 1):
        P[nn] = (2 * nn - 1) / (nn - 1) * ct * P[nn - 1] - nn / (nn - 1) * P[nn - 2]
    for nn in range(1, nmax + 1):
        T[nn] = nn * ct * P[nn] - (nn + 1) * P[nn - 1]
    Er = np.zeros(len(r), complex)
    Et = np.zeros(len(r), complex)
    Ep = np.zeros(len(r), complex)
    for idx, nn in enumerate(n):
        En = (1j ** nn) * E0 * (2 * nn + 1) / (nn * (nn + 1))
        z = spherical_jn(nn, rho)
        dz = z + rho * spherical_jn(nn, rho, derivative=True)        # d(rho z_n)/drho
        pn, tn = P[nn], T[nn]
        # M_o1n = cos(phi) pi z  e_theta - sin(phi) tau z  e_phi
        Mt, Mp = cp * pn * z, -sp * tn * z
        # N_e1n = cos(phi) n(n+1) sin(theta) pi z/rho e_r + cos(phi) tau dz/rho e_theta - sin(phi) pi dz/rho e_phi
        Nr = cp * nn * (nn + 1) * st * pn * z / rho
        Nt = cp * tn * dz / rho
        Np = -sp * pn * dz / rho
        Er += En * (-1j * d[idx] * Nr)
        Et += En * (c[idx] * Mt - 1j * d[idx] * Nt)
        Ep += En * (c[idx] * Mp - 1j * d[idx] * Np)
    ex = Er * st * cp + Et * ct * cp - Ep * sp
    ey = Er * st * sp + Et * ct * sp + Ep * cp
    ez = Er * ct - Et * st
    return np.column_stack([ex, ey, ez])
