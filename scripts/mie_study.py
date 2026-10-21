"""FDTD vs Mie: head-tissue sphere (radius 50 mm) in a 1 V/m plane wave at 900 and 1800 MHz, three cell sizes.
Tissue: 3GPP TS 34.114 head-phantom targets (900 MHz: eps_r 41.5, sigma 0.97 S/m; 1800 MHz: 40.0, 1.40 S/m).
Writes results/mie_study.json."""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rfdtd import mie, sphere  # noqa: E402
from rfdtd.yee import C0, EPS0, ETA0  # noqa: E402

A_R = 0.05
CASES = [(900e6, 41.5, 0.97, [0.005, 0.00333, 0.0025]), (1800e6, 40.0, 1.40, [0.0025, 0.00167])]


def main():
    out = []
    for f, er, s, dxs in CASES:
        w = 2 * np.pi * f
        k = w / C0
        m = np.sqrt(er + 1j * s / (w * EPS0))
        _, _, cabs = mie.cross_sections(k * A_R, m, k)
        p_mie = cabs / (2 * ETA0)
        for dx in dxs:
            t0 = time.time()
            r = sphere.run(f, A_R, er, s, dx)
            Em = mie.internal_field(r["pts"], A_R, k, m)
            rr = np.linalg.norm(r["pts"], axis=1)
            core = rr < A_R - 2 * dx
            mf, mm = np.linalg.norm(np.abs(r["E"]), axis=1), np.linalg.norm(np.abs(Em), axis=1)
            # SAR (rho = 1000 kg/m3) peak and its location error
            sar_f, sar_m = s * mf ** 2 / 2000, s * mm ** 2 / 2000
            row = dict(f_MHz=f / 1e6, dx_mm=1e3 * dx, cells=r["n"] ** 3, steps=r["steps"],
                       rel_l2_E_core=float(np.linalg.norm(mf[core] - mm[core]) / np.linalg.norm(mm[core])),
                       rel_l2_E_all=float(np.linalg.norm(mf - mm) / np.linalg.norm(mm)),
                       p_abs_fdtd=float(r["p_abs"]), p_abs_mie=float(p_mie), p_abs_rel_err=float(r["p_abs"] / p_mie - 1),
                       peak_sar_fdtd=float(sar_f.max()), peak_sar_mie=float(sar_m.max()), minutes=(time.time() - t0) / 60)
            out.append(row)
            print(json.dumps(row), flush=True)
            np.savez_compressed(ROOT / "results" / f"sphere_{int(f / 1e6)}MHz_{int(round(1e4 * dx))}.npz",
                                pts=r["pts"], E=r["E"], Emie=Em)
    (ROOT / "results" / "mie_study.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
