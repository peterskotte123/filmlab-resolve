"""Render ColorChecker x exposure + grey ramp through 500T -> FilmLab settings -> 2383 (Rec.709), check push colour
bias against Kodak's description, and measure the HDR print LUTs in nits.

usage: verify_looks.py LUT_DIR OUT.png
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import colour
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parent.parent
lut_dir, out_png = Path(sys.argv[1]), Path(sys.argv[2])
tet = colour.algebra.table_interpolation_tetrahedral
film = colour.read_LUT(str(lut_dir / "Vision3 500T/500t_film.cube"))
sdr = colour.read_LUT(str(lut_dir / "Vision3 500T/500t_2383_print_rec709.cube"))
pq = colour.read_LUT(str(lut_dir / "Vision3 500T/500t_2383_print_pq1000.cube"))
hlg = colour.read_LUT(str(lut_dir / "Vision3 500T/500t_2383_print_hlg1000.cube"))

dwg = colour.RGB_COLOURSPACES["DaVinci Wide Gamut"]
cc = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
XYZ = colour.chromatic_adaptation(colour.xyY_to_XYZ(np.array(list(cc.data.values()))),
                                  colour.xy_to_XYZ(cc.illuminant), colour.xy_to_XYZ(dwg.whitepoint))
patches = colour.XYZ_to_RGB(XYZ, dwg)
ramp = np.repeat(0.18 * 2.0 ** np.linspace(-6, 6, 24)[:, None], 3, axis=1)
scene = np.stack([patches * 2.0**s for s in (-3, -1.5, 0, 1.5, 3)] + [ramp]).astype(np.float32)
code = film.apply(colour.models.oetf_DaVinciIntermediate(scene), interpolator=tet).astype(np.float32)

looks = {"default": [], "push +2": ["push=2"], "pull -2": ["push=-2"], "interlayer 0": ["inter=0"],
         "interlayer 1.5": ["inter=1.5"], "bleach bypass 1.0": ["bleach=1"]}
rows, dev = [], {}
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    np.save(td / "in.npy", code)
    for name, args in looks.items():
        subprocess.run([sys.executable, "-I", str(root / "scripts/dctl_harness.py"), str(root / "dctl/FilmLab.dctl"),
                        td / "in.npy", td / "out.npy", "stock=V500T", *args], check=True)
        dev[name] = np.load(td / "out.npy")
        rows.append(np.clip(sdr.apply(dev[name], interpolator=tet), 0, 1))

lin = lambda v: colour.models.eotf_BT1886(np.clip(v, 0, 1))
for name in looks:
    r = lin(np.clip(sdr.apply(dev[name], interpolator=tet), 0, 1))[-1]
    hi, mid, lo = r[19], r[12], r[5]  # +3.5, +0.3, -3.4 stops on the ramp
    sat = colour.XYZ_to_Lab(colour.RGB_to_XYZ(lin(rows[list(looks).index(name)][2]), "ITU-R BT.709"))
    chroma = np.hypot(sat[:, 1], sat[:, 2]).mean()
    print(f"{name:18s} ramp R/B  highlight {hi[0] / hi[2]:.3f}  mid {mid[0] / mid[2]:.3f}  shadow {lo[0] / lo[2]:.3f}"
          f"   mean chroma (0 EV patches) {chroma:.1f}   contrast hi/lo {hi[1] / lo[1]:.1f}")

nits_pq = colour.models.eotf_BT2100_PQ(pq.apply(dev["default"], interpolator=tet))
nits_hlg = colour.models.eotf_BT2100_HLG(hlg.apply(dev["default"], interpolator=tet), L_B=0, L_W=1000)
y = lambda n: colour.RGB_luminance(n, colour.RGB_COLOURSPACES["ITU-R BT.2020"].primaries,
                                   colour.RGB_COLOURSPACES["ITU-R BT.2020"].whitepoint)
g_pq = y(colour.models.eotf_BT2100_PQ(pq.apply(film.apply(colour.models.oetf_DaVinciIntermediate(np.full((1, 3), 0.18)), interpolator=tet), interpolator=tet)))
print(f"HDR PQ:  18% grey {g_pq[0]:.1f} nit, brightest ramp step {y(nits_pq[-1]).max():.0f} nit")
print(f"HDR HLG: brightest ramp step {y(nits_hlg[-1]).max():.0f} nit (1000 nit display)")

cell, gap = 22, np.ones((6, 24 * 22, 3))
canvas = []
for img in rows:
    canvas += [np.kron(img, np.ones((cell, cell, 1))), gap]
Image.fromarray((np.concatenate(canvas[:-1]) * 255).astype(np.uint8)).save(out_png)
print(f"wrote {out_png} (rows: {', '.join(looks)})")
