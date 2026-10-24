"""Fit push/pull and interlayer as density-space transforms, validated against the full spektrafilm simulation.

Push/pull: per-layer development gamma (Kodak: contrast gain is largest in the cyan and magenta layers, least in
yellow -> yellow highlights / blue shadows when mid-scale neutrals are balanced; pull is the reverse). The
magnitudes below are our choice; Kodak publishes the direction, not numbers.
Interlayer: dir_couplers.inhibition_interlayer (1.0 = stock).

Model per output channel c (D = default cmy_film density):
    D'_c = sum_k w_ck hat_k(D_c) + sum_{j != c} b_cj (D_j - mid_j)
Reference = the real variant re-timed so mid-grey density matches the default (what printer lights do), printed
through the default print stage. Writes fits.json (knots, weights, cross terms per stock/setting).
"""
import json
import sys
from pathlib import Path

import colour
import numpy as np

from spektrafilm.runtime.params_builder import digest_params, init_params
from spektrafilm.runtime.pipeline import SimulationPipeline

STOCKS = ["kodak_vision3_50d", "kodak_vision3_200t", "kodak_vision3_250d", "kodak_vision3_500t"]
PUSH_LEVELS = {  # stops -> per-layer gamma (cyan, magenta, yellow)
    -2: (0.80, 0.82, 0.92),
    -1: (0.90, 0.91, 0.96),
    0: (1.0, 1.0, 1.0),
    1: (1.15, 1.13, 1.05),
    2: (1.30, 1.26, 1.10),
}
INTERLAYER = [0.0, 0.5, 1.0, 1.5]
KNOTS = np.round(np.arange(-0.2, 2.61, 0.2), 4)


def pipeline(film, gamma=1.0, interlayer=1.0, print_stock="kodak_2383"):
    p = init_params(film_profile=film, print_profile=print_stock)
    p.debug.lut_mode = True
    p.io.input_color_space = "DaVinci Wide Gamut"
    p.io.output_color_space = "ITU-R BT.709"
    p.io.input_cctf_decoding = False
    p.io.output_cctf_encoding = False
    p.film_render.density_curve_gamma = gamma
    p = digest_params(p)
    p.film_render.dir_couplers.inhibition_interlayer = interlayer
    return SimulationPipeline(p)


def samples(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    dwg = colour.RGB_COLOURSPACES["DaVinci Wide Gamut"]
    rec709 = colour.RGB_COLOURSPACES["ITU-R BT.709"]
    refl = rng.uniform(0.03, 1.0, (n, 3)) ** 1.5
    refl /= colour.RGB_luminance(refl, rec709.primaries, rec709.whitepoint)[:, None]
    exposure = 0.18 * 2.0 ** rng.uniform(-6, 5, n)
    lin = colour.RGB_to_RGB(refl * exposure[:, None], rec709, dwg)
    return np.clip(lin, 0, None).reshape(-1, 1, 3).astype(np.float32)


def run(pipe, img, collect, inject=None):
    return np.asarray(pipe.process(img, inject=inject, collect=collect), dtype=float).reshape(-1, 3)


def de(a, b):
    to_lab = lambda rgb: colour.XYZ_to_Lab(colour.RGB_to_XYZ(np.clip(rgb, 0, 1), "ITU-R BT.709"))
    return colour.delta_E(to_lab(a), to_lab(b), method="CIE 2000")


def hats(x):
    x = np.clip(x, KNOTS[0], KNOTS[-1])
    step = KNOTS[1] - KNOTS[0]
    return np.clip(1.0 - np.abs(x[:, None] - KNOTS[None, :]) / step, 0.0, None)


def features(d, mid, c):
    others = [j for j in range(3) if j != c]
    return np.hstack([hats(d[:, c]), d[:, others] - mid[others]])


def fit(d0, target, mid0):
    model = {"w": [], "b": []}
    for c in range(3):
        coef, *_ = np.linalg.lstsq(features(d0, mid0, c), target[:, c], rcond=None)
        model["w"].append(coef[: len(KNOTS)])
        model["b"].append(coef[len(KNOTS):])
    # re-time exactly: mid-grey maps to itself
    fix = apply(model, mid0[None, :], mid0)[0] - mid0
    for c in range(3):
        model["w"][c] = model["w"][c] - fix[c]
    return model


def apply(model, d, mid):
    return np.stack([features(d, mid, c) @ np.concatenate([model["w"][c], model["b"][c]]) for c in range(3)], axis=1)


def report(label, base, real_density, model_density):
    real = run(base, real_density.reshape(-1, 1, 3).astype(np.float32), "rgb_out", inject="cmy_film")
    mod = run(base, model_density.reshape(-1, 1, 3).astype(np.float32), "rgb_out", inject="cmy_film")
    err = de(real, mod)
    print(f"{label}: dE2000 median {np.median(err):.2f}  p95 {np.percentile(err, 95):.2f}  max {err.max():.2f}")


def grid_params(push, inter):
    """Continuous push (stops) -> per-layer gamma by linear interpolation between PUSH levels."""
    levels = sorted(PUSH_LEVELS)
    lo = max(l for l in levels if l <= push) if push < levels[-1] else levels[-2]
    hi = levels[levels.index(lo) + 1]
    t = (push - lo) / (hi - lo)
    g = tuple((1 - t) * a + t * b for a, b in zip(PUSH_LEVELS[lo], PUSH_LEVELS[hi]))
    return dict(gamma=g, interlayer=inter)


def blend(models, d0, mid0, push, inter):
    """Bilinear blend of the four surrounding grid models (what the DCTL does)."""
    pl = sorted(PUSH_LEVELS)
    i0 = min(int(np.searchsorted(pl, push, side="right")) - 1, len(pl) - 2)
    j0 = min(int(np.searchsorted(INTERLAYER, inter, side="right")) - 1, len(INTERLAYER) - 2)
    tp = (push - pl[i0]) / (pl[i0 + 1] - pl[i0])
    ti = (inter - INTERLAYER[j0]) / (INTERLAYER[j0 + 1] - INTERLAYER[j0])
    out = 0
    for di, wp in ((0, 1 - tp), (1, tp)):
        for dj, wi in ((0, 1 - ti), (1, ti)):
            out = out + wp * wi * apply(models[(pl[i0 + di], INTERLAYER[j0 + dj])], d0, mid0)
    return out


def main(out_path):
    img = samples()
    gray = np.full((1, 1, 3), 0.18, np.float32)
    out = {"knots": KNOTS.tolist(), "push_levels": sorted(PUSH_LEVELS), "interlayer_levels": INTERLAYER,
           "push_gamma": {str(k): v for k, v in PUSH_LEVELS.items()}, "stocks": {}}

    def timed(film, kw, mid0):
        var = pipeline(film, **kw)
        return run(var, img, "cmy_film") - run(var, gray, "cmy_film")[0] + mid0

    for film in STOCKS:
        base = pipeline(film)
        d0 = run(base, img, "cmy_film")
        mid0 = run(base, gray, "cmy_film")[0]
        models, grid = {}, []
        for push in sorted(PUSH_LEVELS):
            row = []
            for inter in INTERLAYER:
                target = timed(film, dict(gamma=PUSH_LEVELS[push], interlayer=inter), mid0)
                m = fit(d0, target, mid0)
                models[(push, inter)] = m
                report(f"{film:22s} grid push {push:+d} inter {inter:.1f}", base, target, apply(m, d0, mid0))
                row.append({"w": [w.round(5).tolist() for w in m["w"]], "b": [b.round(5).tolist() for b in m["b"]]})
            grid.append(row)
        for push, inter in [(0.5, 0.75), (1.5, 1.25), (-1.5, 0.25), (-0.5, 1.0), (2.0, 1.5)]:
            target = timed(film, grid_params(push, inter), mid0)
            report(f"{film:22s} OFF-GRID push {push:+.1f} inter {inter:.2f}", base, target, blend(models, d0, mid0, push, inter))
        out["stocks"][film] = {"mid": mid0.round(5).tolist(), "grid": grid}
    Path(out_path).write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
