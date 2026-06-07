# Standalone verification of CascadeSpecification.hpp + the octahedral mapping.
# Mirrors the C++ math exactly so we can confirm the per-cascade table, the
# A4/B2 validators, VRAM, and the B1 octahedral round-trip without an MSVC build.
#
# Run:  python Source/GI/spec_selftest.py

import math, random

# ---- constants (mirror CascadeSpecification.hpp) ----
INTERVAL_LENGTH_FACTOR = 4.0

# ---- HierarchySpecification defaults ----
RC_QUERY_RADIUS   = 8.0     # m
BASE_INTERVAL_LEN = 0.15    # m  (t0)
BASE_PROBE_AXIS   = 64
BASE_ANGULAR_AXIS = 4       # R0

def resolve_interval_start(t0, c):
    return t0 * (INTERVAL_LENGTH_FACTOR ** c - 1.0) / 3.0

def resolve_interval_end(t0, c):
    return resolve_interval_start(t0, c + 1)

def resolve_base_probe_pitch():
    return BASE_INTERVAL_LEN

def resolve_cascade_count(override=0):
    if override != 0:
        return override
    diameter = 2.0 * RC_QUERY_RADIUS
    ratio    = 3.0 * diameter / BASE_INTERVAL_LEN + 1.0
    levels   = math.ceil(math.log(ratio) / math.log(INTERVAL_LENGTH_FACTOR))
    return max(1, int(levels))

def resolve_cascade(c):
    probe_axis = BASE_PROBE_AXIS >> c
    if probe_axis < 2:
        probe_axis = 2
    oct_side = BASE_ANGULAR_AXIS << c
    return {
        "c": c,
        "probe_axis": probe_axis,
        "oct_side": oct_side,
        "dirs": oct_side * oct_side,
        "int_start": resolve_interval_start(BASE_INTERVAL_LEN, c),
        "int_end":   resolve_interval_end(BASE_INTERVAL_LEN, c),
        "atlas_w": probe_axis * oct_side,
        "atlas_h": probe_axis * oct_side,
        "atlas_d": probe_axis,
    }

def resolve_hierarchy():
    return [resolve_cascade(c) for c in range(resolve_cascade_count())]

def confirm_angular_axis_pow2():
    a = BASE_ANGULAR_AXIS
    return a != 0 and (a & (a - 1)) == 0

def confirm_top_covers_region():
    count = resolve_cascade_count()
    top = resolve_cascade(count - 1)
    pitch = resolve_base_probe_pitch() * (2.0 ** (count - 1))
    return top["probe_axis"] * pitch >= 2.0 * RC_QUERY_RADIUS

def ray_budget(d):
    return d["probe_axis"] ** 3 * d["dirs"]

# ---- octahedral mapping, exact port of octahedral.glsl ----
def normalize3(v):
    l = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    return (v[0]/l, v[1]/l, v[2]/l)

def oct_encode(d):
    ax, ay, az = abs(d[0]), abs(d[1]), abs(d[2])
    denom = ax + ay + az
    ex, ey = d[0]/denom, d[2]/denom    # .xz
    if d[1] < 0.0:
        fx = (1.0 - abs(ey)) * (1.0 if ex >= 0.0 else -1.0)
        fy = (1.0 - abs(ex)) * (1.0 if ey >= 0.0 else -1.0)
        ex, ey = fx, fy
    return (ex * 0.5 + 0.5, ey * 0.5 + 0.5)

def oct_decode(e):
    cx, cy = e[0]*2.0 - 1.0, e[1]*2.0 - 1.0
    dx = cx
    dy = 1.0 - abs(cx) - abs(cy)
    dz = cy
    fold = max(-dy, 0.0)
    dx += -fold if dx >= 0.0 else fold
    dz += -fold if dz >= 0.0 else fold
    return normalize3((dx, dy, dz))

def main():
    print("=== HierarchySpecification ===")
    print(f"RcQueryRadius={RC_QUERY_RADIUS:.3f} m  t0={BASE_INTERVAL_LEN:.3f} m  "
          f"s0={resolve_base_probe_pitch():.3f} m  BaseProbeAxis={BASE_PROBE_AXIS}  R0={BASE_ANGULAR_AXIS}\n")

    r0pow2  = confirm_angular_axis_pow2()
    covered = confirm_top_covers_region()
    print(f"ConfirmAngularAxisPowerOfTwo  : {'PASS' if r0pow2 else 'FAIL'}")
    print(f"ConfirmTopCascadeCoversRegion : {'PASS' if covered else 'FAIL'}\n")

    count = resolve_cascade_count()
    print(f"ResolveCascadeCount = {count}\n")

    print(" c | ProbeAxis | OctSide |  IntStart |  IntEnd  |  Pitch  |    AtlasW x H x D    |    RayBudget   | VRAM(MB f16vec4)")
    print("---+-----------+---------+-----------+----------+---------+----------------------+----------------+-----------------")
    total_bytes = 0.0
    hier = resolve_hierarchy()
    for d in hier:
        pitch = resolve_base_probe_pitch() * (2.0 ** d["c"])
        b = d["atlas_w"] * d["atlas_h"] * d["atlas_d"] * 8.0   # vec4 f16 = 8 B/texel
        total_bytes += b
        print(f" {d['c']} |   {d['probe_axis']:5d}   |  {d['oct_side']:5d}  | "
              f" {d['int_start']:7.4f}  | {d['int_end']:7.4f}  | {pitch:6.4f}  | "
              f"{d['atlas_w']:5d} x {d['atlas_h']:5d} x {d['atlas_d']:4d} | "
              f"{ray_budget(d):14d} | {b/(1024*1024):8.2f}")

    print(f"\nTotal atlas VRAM (f16 vec4, single buffer) = {total_bytes/(1024*1024):.2f} MB")
    print(f"With a transient merge ping copy (worst)   = {2*total_bytes/(1024*1024):.2f} MB\n")

    top = hier[-1]
    tp = resolve_base_probe_pitch() * (2.0 ** top["c"])
    print(f"A4 top-cascade coverage: ProbeAxis({top['probe_axis']}) * pitch({tp:.3f}) = "
          f"{top['probe_axis']*tp:.3f} m  vs  diameter {2*RC_QUERY_RADIUS:.3f} m\n")

    # B1 round-trip
    rng = random.Random(1234567)
    max_err = 0.0; sum_err = 0.0; n = 0; outliers = 0
    for _ in range(10000):
        v = (rng.uniform(-1,1), rng.uniform(-1,1), rng.uniform(-1,1))
        if v[0]*v[0]+v[1]*v[1]+v[2]*v[2] < 1e-6:
            continue
        d = normalize3(v)
        r = oct_decode(oct_encode(d))
        e = math.sqrt((r[0]-d[0])**2 + (r[1]-d[1])**2 + (r[2]-d[2])**2)
        max_err = max(max_err, e); sum_err += e; n += 1
        if e > 1e-3 and outliers < 5:
            print(f"  outlier d={d} -> r={r} err={e:.5f}")
            outliers += 1
    print(f"B1 octahedral round-trip over {n} dirs: maxErr={max_err:.3e} "
          f"meanErr={sum_err/n:.3e}  {'PASS' if max_err < 1e-5 else 'FAIL'} (threshold 1e-5)")

    # texel-centre dirs unit length on c0 4x4
    max_off = 0.0
    for y in range(4):
        for x in range(4):
            dirv = oct_decode(((x+0.5)/4.0, (y+0.5)/4.0))
            ln = math.sqrt(dirv[0]**2 + dirv[1]**2 + dirv[2]**2)
            max_off = max(max_off, abs(ln - 1.0))
    print(f"Texel-centre dirs unit-length (c0 4x4): max |len-1| = {max_off:.3e} "
          f"{'PASS' if max_off < 1e-5 else 'FAIL'}")

if __name__ == "__main__":
    main()
