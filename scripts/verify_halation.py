"""Night frame with practical lights -> Halation.dctl -> 500T film LUT -> 2383 print LUT. Writes a comparison PNG."""
import subprocess
import sys
from pathlib import Path

import colour
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parent.parent
lut_dir = root / "luts/Vision3 500T"
out_png = Path(sys.argv[1])
tmp = out_png.parent

W, H = 960, 540
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
lin = np.full((H, W, 3), 0.02, np.float32) * (0.5 + xx[..., None] / W)  # dark background
lin[300:460, 80:300] = [0.18, 0.18, 0.18]  # grey card: below threshold, must not glow
lin[80:240, 380:560] = [6.0, 5.5, 4.5]  # bright window
for cx, cy, rgb in [(700, 150, [60, 50, 30]), (820, 380, [40, 40, 40]), (620, 420, [2, 4, 60])]:  # warm bulb, white bulb, blue LED
    lin[(xx - cx) ** 2 + (yy - cy) ** 2 < 36] = rgb
np.save(tmp / "hal_in.npy", colour.models.oetf_DaVinciIntermediate(lin).astype(np.float32))

harness = [sys.executable, "-I", str(root / "scripts/dctl_harness.py"), str(root / "dctl/Halation.dctl")]
subprocess.run(harness + [tmp / "hal_in.npy", tmp / "hal_on.npy"], check=True)
subprocess.run(harness + [tmp / "hal_in.npy", tmp / "hal_only.npy", "show=1", "amount=1.0"], check=True)

t = colour.algebra.table_interpolation_tetrahedral
film = colour.read_LUT(str(lut_dir / "500t_film.cube"))
prnt = colour.read_LUT(str(lut_dir / "500t_2383_print_rec709.cube"))


def render(di):
    return np.clip(prnt.apply(film.apply(di, interpolator=t), interpolator=t), 0, 1)


off = np.load(tmp / "hal_in.npy")
on = np.load(tmp / "hal_on.npy")
only = np.clip(colour.models.oetf_inverse_DaVinciIntermediate(np.load(tmp / "hal_only.npy")), 0, 1)

d = colour.models.oetf_inverse_DaVinciIntermediate(on) - lin
print(f"grey card added light (should be ~0): {np.abs(d[320:440, 100:280]).max():.5f}")
print(f"blue LED surroundings red gain (no red in source, should be ~0): {d[400:440, 640:680, 0].max():.5f}")
ring = d[150 - 40:150 + 40, 700 + 15:700 + 40]
print(f"warm bulb ring added R / G / B: {ring[..., 0].mean():.4f} / {ring[..., 1].mean():.4f} / {ring[..., 2].mean():.4f}")

rows = [render(off), render(on), only ** (1 / 2.4)]
gap = np.ones((6, W, 3))
img = np.concatenate([rows[0], gap, rows[1], gap, rows[2]])
Image.fromarray((img * 255).astype(np.uint8)).save(out_png)
print(f"wrote {out_png} (top: no halation, middle: halation default, bottom: halation only)")
