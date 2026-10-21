"""IEEE 1528 validation dipole over a flat phantom at 900 MHz, three cell sizes. Writes results/dipole_study.json."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rfdtd import dipole  # noqa: E402

if __name__ == "__main__":
    out = []
    for dx in (0.003, 0.0025, 0.002):
        t = time.time()
        r = dipole.run(dx)
        r["minutes"] = (time.time() - t) / 60
        out.append(r)
        print(json.dumps(r), flush=True)
        (ROOT / "results" / "dipole_study.json").write_text(json.dumps(out, indent=1))
