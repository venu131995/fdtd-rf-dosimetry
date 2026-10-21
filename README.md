# 3-D FDTD for RF Dosimetry, Verified against Mie Theory and the IEEE 1528 Validation Dipole

A finite-difference time-domain solver written from scratch, checked against an exact solution and against the
standard system-validation set-up used by SAR laboratories, with the remaining error stated.

## In short

| Question | Answer |
|---|---|
| Does the solver reproduce an exact solution? | Head-tissue sphere (radius 50 mm; 3GPP TS 34.114 head targets: eps_r 41.5, sigma 0.97 S/m at 900 MHz; 40.0, 1.40 S/m at 1800 MHz) in a 1 V/m plane wave, against Mie theory: interior field **3.7 %, 2.9 %, 2.4 %** relative L2 error at 5, 3.3 and 2.5 mm cells (900 MHz); **2.7 %** at 1800 MHz and 2.5 mm. Absorbed power +6.1 %, +5.3 %, **+3.9 %** (900 MHz); +1.7 % (1800 MHz). |
| Why not faster convergence? | The sphere is staircased onto the cubic grid, a first-order error at the surface where the field is strongest; conformal or sub-cell averaging would be the next step. |
| IEEE 1528 system validation | Dipole of length 149-150 mm with a one-cell gap feed, axis 15 mm above a flat phantom (2 mm shell, eps_r 3.7; head liquid 41.5 / 0.97 S/m; 0.4 x 0.6 wavelength, 150 mm deep), 900 MHz, normalised to 1 W accepted power: **1 g peak spatial-average SAR 10.93 W/kg** against the 10.8 W/kg target (**+1.2 %**, within the 10 % validation tolerance), 10 g 6.87 W/kg, input impedance 48.6 - j9.8 Ohm (3 mm cells; 5 mm cells give 9.78 W/kg). |
| Is the plane-wave source clean? | Total-field/scattered-field leakage below 1 % of the incident field (test). |

## Solver (`src/rfdtd`)

* **Yee scheme** on a uniform cubic grid, Courant number 0.99/sqrt(3); lossy, non-magnetic media with eps_r and
  sigma averaged onto each edge and a semi-implicit conductivity update.
* **Convolutional PML** (Roden and Gedney 2000) on all faces, polynomial grading, small CFS alpha.
* **Total-field/scattered-field** plane wave fed by a 1-D auxiliary Yee grid with the same dx and dt, so its
  numerical dispersion matches the 3-D grid along the propagation axis.
* **Phasors** by a running DFT over the last periods; fields averaged from edges to cell centres.
* **Mie theory** (Bohren and Huffman 1983): internal field from the vector spherical harmonics with the c_n, d_n
  coefficients, and the absorption cross-section from a_n, b_n.
* **Dipole**: thin wire as zeroed E edges, gap source, feed current from the loop integral of H around the gap,
  accepted power 0.5 Re(V I*), SAR = sigma |E|^2 / (2 rho) with rho = 1000 kg/m3, cube averages over cubes fully
  inside the liquid (1 g: 10 mm cube; 10 g: 21.5 mm, rounded to whole cells).

## Verification (`pytest tests`, 4 tests)

| Check | Result |
|---|---|
| Mie internal field in the small-sphere limit, 3/(eps_r + 2) | 1e-6 |
| Mie absorption cross-section vs volume integral of sigma |E|^2 / 2 of the Mie internal field | within 1 % |
| TF/SF plane wave in an empty grid: amplitude inside, leakage outside | 1 +- 2 %; below 1 % |
| Cube averaging only over cubes inside the medium | exact |

## Limitations

* Staircased geometry (first-order surface error), uniform grid, pure Python/NumPy (a 1.1 M-cell dipole run takes
  15 minutes on one core); a GPU or C++ kernel and sub-cell averaging are the obvious extensions.
* The wire dipole is thinner than the 3.6 mm reference dipole; the cube averaging follows the spirit of
  IEC/IEEE 62209-1528 but not every detail (no extrapolation to the phantom surface).
* The 10 g target of IEEE 1528 is not compared here, because I could not verify its value from a primary source.

## Run

    pip install numpy scipy pytest
    python scripts/mie_study.py        # sphere vs Mie, about 40 min on one core
    python scripts/dipole_study.py     # IEEE 1528 dipole, 3 / 2.5 / 2 mm cells
    pytest tests

Results: `results/mie_study.json`, `results/dipole_study.json`.
