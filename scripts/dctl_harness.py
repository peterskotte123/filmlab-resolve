"""Compile a .dctl as C++ (shim for Resolve's DCTL API) and run it over a float32 planar RGB image.

usage: dctl_harness.py FILE.dctl IN.npy OUT.npy [name=value ...]
IN/OUT: float32 arrays (H, W, 3). UI params default to the DCTL's defaults; override with name=value.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SHIM = r"""
#include <cmath>
#include <cstdio>
#include <cstdlib>
struct float3 { float x, y, z; };
static inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }
typedef const float* __TEXTURE__;
static int g_W;
#define __DEVICE__ static inline
#define __CONSTANT__ static const
#define __CONSTANTREF__ const
#define _tex2D(t, x, y) ((t)[(y) * g_W + (x)])
#define _fabs fabsf
#define _powf powf
#define _logf logf
#define _log2f log2f
#define _log10f log10f
#define _expf expf
#define _exp2f exp2f
#define _fmaxf fmaxf
#define _fminf fminf
#define _sqrtf sqrtf
#define _floorf floorf
#define _ceilf ceilf
#define _round roundf
#define _cosf cosf
#define _sinf sinf
#define _clampf(x, a, b) fminf(fmaxf(x, a), b)
#define _saturatef(x) fminf(fmaxf(x, 0.0f), 1.0f)
"""

MAIN_TEX = r"""
int main(int argc, char** argv) {
  int W = atoi(argv[1]), H = atoi(argv[2]); g_W = W;
  size_t n = (size_t)W * H;
  float* buf = (float*)malloc(n * 3 * sizeof(float));
  FILE* f = fopen(argv[3], "rb"); fread(buf, sizeof(float), n * 3, f); fclose(f);
  float* out = (float*)malloc(n * 3 * sizeof(float));
  #pragma omp parallel for
  for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
    float3 o = %CALL%;
    size_t i = (size_t)y * W + x;
    out[i] = o.x; out[n + i] = o.y; out[2 * n + i] = o.z;
  }
  f = fopen(argv[4], "wb"); fwrite(out, sizeof(float), n * 3, f); fclose(f);
  return 0;
}
"""

CALL_TEX = "transform(W, H, x, y, buf, buf + n, buf + 2 * n)"
CALL_RGB = "transform(W, H, x, y, buf[(size_t)y*W+x], buf[n+(size_t)y*W+x], buf[2*n+(size_t)y*W+x])"


def translate(src: str, overrides: dict) -> str:
    out = []
    for line in src.splitlines():
        m = re.match(r"\s*DEFINE_UI_PARAMS\((.*)\)\s*$", line)
        if not m:
            out.append(line)
            continue
        body = m.group(1)
        name, _label, kind, rest = [s.strip() for s in body.split(",", 3)]
        default = rest.split(",", 1)[0].strip()
        value = overrides.get(name, default)
        if kind == "DCTLUI_COMBO_BOX":
            enums = re.search(r"\{([^}]*)\}", rest).group(1)
            out.append(f"enum {{ {enums} }};")
            out.append(f"static const int {name} = {value};")
        elif kind == "DCTLUI_CHECK_BOX":
            out.append(f"static const int {name} = {value};")
        else:
            out.append(f"static const float {name} = {value}f;" if "." in value else f"static const float {name} = {value}.0f;")
    return "\n".join(out)


def main():
    dctl, inp, outp = sys.argv[1:4]
    overrides = dict(a.split("=", 1) for a in sys.argv[4:])
    src = Path(dctl).read_text()
    uses_tex = "__TEXTURE__ p_TexR" in src
    code = SHIM + translate(src, overrides) + MAIN_TEX.replace("%CALL%", CALL_TEX if uses_tex else CALL_RGB)
    img = np.load(inp).astype(np.float32)
    h, w, _ = img.shape
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "k.cpp").write_text(code)
        subprocess.run(["g++", "-O2", "-fopenmp", "-Wall", "-Wno-unused-variable", "-Wno-unused-result",
                        "-o", td / "k", td / "k.cpp"], check=True)
        np.ascontiguousarray(img.transpose(2, 0, 1)).tofile(td / "in.raw")
        subprocess.run([td / "k", str(w), str(h), td / "in.raw", td / "out.raw"], check=True)
        res = np.fromfile(td / "out.raw", dtype=np.float32).reshape(3, h, w).transpose(1, 2, 0)
    np.save(outp, res)


if __name__ == "__main__":
    main()
