# filmlab-resolve

Open-source film emulation for DaVinci Resolve, built on [spektrafilm](https://github.com/andreavolpato/spektrafilm)
(spectral negative → print simulation by Andrea Volpato): Kodak Vision3 negatives, 2383/2393 print, printer points,
push/pull, interlayer, bleach bypass, halation, grain, SDR and HDR outputs.

## Install

Copy `luts/` and `dctl/` into one folder inside Resolve's LUT folder, then refresh LUTs on the Color page:

- Windows: `C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\LUT\`
- macOS: `/Library/Application Support/Blackmagic Design/DaVinci Resolve/LUT/`
- Linux: `/opt/resolve/LUT/`

## Node tree (working space: DaVinci Wide Gamut / DaVinci Intermediate)

1. Your grade (exposure, balance)
2. DCTL `Halation` — red/orange glow around highlights (scene light, before the negative)
3. LUT `Vision3 <stock>/<stock>_film.cube` — scene → negative density
4. DCTL `FilmLab` — same stock as the LUTs:
   - Push / Pull (−2 … +2 stops) · Interlayer (0 … 1.5, 1 = stock) · Bleach Bypass (0 … 1)
   - Printer Density / Red / Green / Blue (1 pt = 0.025 logE)
5. LUT `Vision3 <stock>/<stock>_<2383|2393>_print_<output>.cube`
   - `rec709` SDR · `pq1000` Rec.2100 PQ, 1000 nit · `hlg1000` Rec.2100 HLG, 1000 nit
6. DCTL `RCS Film Grain` (MIT, pg0/rcs-film-grain)

All LUTs share one density wire (D = code × 4.2 − 0.2), but always use film + print LUTs of the same stock:
the print LUT depends on the negative's dyes.

## Accuracy (`scripts/verify_filmlab.py luts`)

FilmLab vs the full spektrafilm development, re-timed at mid-grey, out-of-sample colours, all four stocks:
defaults dE2000 median 0.07 / p95 0.2; push, interlayer and combinations median < 0.5 / p95 < 1.8.
Weakest corner: Push +2 with Interlayer 1.5 (fit median ~1.8).

Status: the DCTLs compile and run in the C++ harness; they have not yet been tested inside Resolve itself.

## What is modelled vs chosen

- Stocks, print, interlayer (DIR couplers): spektrafilm's physical model.
- Push/pull: spektrafilm per-layer development gamma. Direction from Kodak (contrast builds most in cyan and
  magenta layers → warm highlights / blue shadows; pull the reverse); the magnitudes are our choice
  (`PUSH_LEVELS` in `scripts/fit_processes.py`).
- Bleach bypass: our model — retained silver adds neutral density ∝ developed density (`SILVER` in FilmLab).
- HDR: our mapping — print at the BT.2408 anchor (18% grey ≈ 35 nit), highlights expanded so print white ≈ 1000 nit.

## Rebuild

Needs a spektrafilm checkout beside this repo, installed in `../spektrafilm/.venv` (Python 3.13).

```bash
PY=../spektrafilm/.venv/bin/python
$PY -I scripts/fit_processes.py fits.json        # fit push/pull x interlayer models (prints validation)
$PY -I scripts/build_luts.py luts                 # bake film + SDR/HDR print LUTs on the fixed wire
python3 -I scripts/build_dctl.py fits.json dctl/FilmLab.dctl
$PY -I scripts/verify_filmlab.py luts             # end-to-end check vs the simulation
$PY -I scripts/verify_looks.py luts OUT.png       # look strip + push bias + HDR nits
$PY -I scripts/verify_halation.py OUT.png
```

DCTLs are tested outside Resolve: `scripts/dctl_harness.py` compiles a .dctl as C++ and runs it on an image.

## Licences

- Scripts and `FilmLab.dctl` / `Halation.dctl`: GPL-3.0 (`LICENSE`), as they build on spektrafilm (GPL-3.0).
- LUTs and `fits.json` (derived from spektrafilm's profiles): CC BY-SA 4.0 + `luts/SPEKTRAFILM_LICENSE.txt`
  (commercial use OK, free to share, no resale). Credit: spektrafilm by Andrea Volpato.
- `RCS Film Grain.dctl`: MIT, © Patrick Gawron / Red Coral Studios (`dctl/RCS Film Grain.LICENSE`).

Not affiliated with Kodak or with any commercial film emulation product.
