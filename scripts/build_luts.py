"""Bake film/print LUTs on ONE fixed density wire (so FilmLab.dctl can push, bleach and time the negative
between them), SDR Rec.709 print LUTs, and HDR print LUTs (Rec.2100 PQ 1000 nit, Rec.2100 HLG).

usage: build_luts.py OUT_DIR
HDR mapping (our choice; a print has no HDR range of its own): the SDR print is placed at the BT.2408 anchor
(print linear 1.0 = 203 nit), then luminance above mid-grey is expanded smoothly in log2 so the print's
brightest white reaches 1000 nit. Hue and saturation ratios are kept.
"""
import json
import shutil
import sys
from pathlib import Path

import colour
import numpy as np

from spektrafilm_lut_creator.bundles import BundleSpec
from spektrafilm_lut_creator.builders import BundleBuilder
from spektrafilm_lut_creator.color_spaces import decode_cctf
from spektrafilm_lut_creator.wires import DensityWire

D_MIN, D_MAX = -0.2, 4.0
STOCKS = {"kodak_vision3_50d": "Vision3 50D", "kodak_vision3_200t": "Vision3 200T",
          "kodak_vision3_250d": "Vision3 250D", "kodak_vision3_500t": "Vision3 500T"}
PRINTS = ["kodak_2383", "kodak_2393"]
REF_WHITE, PEAK = 203.0, 1000.0


class FixedWireBuilder(BundleBuilder):
    def _compute_density_wire(self, pipeline, spec):
        return DensityWire(d_max=(D_MAX,) * 3, d_min=(D_MIN,) * 3)


def bake(film, output, out):
    spec = BundleSpec(film_profile=film, print_profiles=tuple(PRINTS), input_color_space="DaVinci Intermediate",
                      output_color_space=output, topology="2lut", resolution=65)
    builder = FixedWireBuilder(spec)
    return builder.write(builder.build(), out / spec.name)


def hdr_from_rec2020(cube_path, out_pq, out_hlg):
    lut = colour.read_LUT(str(cube_path))
    lin = decode_cctf(lut.table, "Rec.2020")  # display-relative print light, 1.0 = SDR white
    y = colour.RGB_luminance(lin, colour.RGB_COLOURSPACES["ITU-R BT.2020"].primaries,
                             colour.RGB_COLOURSPACES["ITU-R BT.2020"].whitepoint)
    y = np.maximum(y, 1e-6)
    y_mid = 0.18
    x = np.log2(y / y_mid)
    x_max = np.log2(y.max() / y_mid)
    y_target = np.log2(PEAK / (REF_WHITE * y_mid))
    a = (y_target - x_max) / x_max**2
    x_out = np.where(x > 0, x + a * x**2, x)  # slope 1 at mid-grey, print white -> PEAK
    nits = (REF_WHITE * y_mid * 2.0**x_out / y)[..., None] * lin
    nits = np.clip(nits, 0, PEAK)
    for path, enc in ((out_pq, colour.models.eotf_inverse_BT2100_PQ(nits)),
                      (out_hlg, colour.models.eotf_inverse_BT2100_HLG(nits, L_B=0, L_W=PEAK))):
        colour.write_LUT(colour.LUT3D(np.clip(enc, 0, 1), name=path.stem), str(path))
    return float(y.max()), float(nits.max())


def main(out):
    out = Path(out)
    tmp = out / "_bundles"
    tmp.mkdir(parents=True, exist_ok=True)
    for film, label in STOCKS.items():
        dest = out / label
        dest.mkdir(parents=True, exist_ok=True)
        sdr = bake(film, "rec709", tmp)
        wide = bake(film, "rec2020", tmp)
        tag = film.replace("kodak_vision3_", "")
        shutil.copy(next(sdr.glob("lut_*_film.cube")), dest / f"{tag}_film.cube")
        for pr in PRINTS:
            p = pr.replace("kodak_", "")
            shutil.copy(next(sdr.glob(f"lut_*_{p}_print.cube")), dest / f"{tag}_{p}_print_rec709.cube")
            ymax, nmax = hdr_from_rec2020(next(wide.glob(f"lut_*_{p}_print.cube")),
                                          dest / f"{tag}_{p}_print_pq1000.cube", dest / f"{tag}_{p}_print_hlg1000.cube")
            print(f"{label} {p}: print white {ymax:.3f} -> {nmax:.0f} nit")
        wire = json.loads((sdr / "bundle.json").read_text())["wires"]["cmy_film"]
        assert wire["d_min"] == [D_MIN] * 3 and wire["d_max"] == [D_MAX] * 3, wire
    shutil.copy(next(tmp.glob("*/SPEKTRAFILM_LICENSE.txt")), out / "SPEKTRAFILM_LICENSE.txt")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
