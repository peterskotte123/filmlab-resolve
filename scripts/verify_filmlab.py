"""End-to-end check: real spektrafilm development (re-timed) vs film LUT -> FilmLab.dctl (compiled) -> print LUT.

usage: verify_filmlab.py LUT_DIR
Out-of-sample colours (different seed from the fit). Errors in dE2000 on Rec.709 display light.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import colour
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fit_processes import de, grid_params, pipeline, run, samples  # noqa: E402
from spektrafilm_lut_creator.color_spaces import decode_cctf  # noqa: E402

root = Path(__file__).resolve().parent.parent
lut_dir = Path(sys.argv[1])
STOCKS = {"kodak_vision3_50d": ("V50D", "Vision3 50D", "50d"), "kodak_vision3_200t": ("V200T", "Vision3 200T", "200t"),
          "kodak_vision3_250d": ("V250D", "Vision3 250D", "250d"), "kodak_vision3_500t": ("V500T", "Vision3 500T", "500t")}
SETTINGS = [(0, 1.0), (1, 1.0), (-1, 1.0), (2, 0.5), (-2, 0.0), (0.5, 1.25), (1.5, 0.75), (0, 0.0), (0, 1.5)]
tet = colour.algebra.table_interpolation_tetrahedral

img = samples(1500, seed=7)
gray = np.full((1, 1, 3), 0.18, np.float32)
di = colour.models.oetf_DaVinciIntermediate(img)

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    for film, (enum, folder, tag) in STOCKS.items():
        film_lut = colour.read_LUT(str(lut_dir / folder / f"{tag}_film.cube"))
        print_lut = colour.read_LUT(str(lut_dir / folder / f"{tag}_2383_print_rec709.cube"))
        np.save(td / "code.npy", film_lut.apply(di, interpolator=tet).astype(np.float32))
        base = pipeline(film)
        mid0 = run(base, gray, "cmy_film")[0]
        for push, inter in SETTINGS:
            var = pipeline(film, **grid_params(push, inter))
            target = run(var, img, "cmy_film") - run(var, gray, "cmy_film")[0] + mid0
            real = run(base, target.reshape(-1, 1, 3).astype(np.float32), "rgb_out", inject="cmy_film")
            subprocess.run([sys.executable, "-I", str(root / "scripts/dctl_harness.py"), str(root / "dctl/FilmLab.dctl"),
                            td / "code.npy", td / "dev.npy", f"stock={enum}", f"push={push}", f"inter={inter}"], check=True)
            out = print_lut.apply(np.load(td / "dev.npy"), interpolator=tet)
            model = decode_cctf(np.clip(out, 0, 1), "Rec.709").reshape(-1, 3)
            err = de(real, model)
            print(f"{folder:13s} push {push:+.1f} inter {inter:.2f}: dE2000 median {np.median(err):.2f}  "
                  f"p95 {np.percentile(err, 95):.2f}  max {err.max():.2f}")
