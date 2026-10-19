"""Camera rigs (pure numpy) shared by renderer and post-processing."""
import math
import numpy as np
from config import *
import physics as P

def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)

def _l(pair, u): return pair[0] + (pair[1] - pair[0]) * u

def look_basis(pos, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - pos; f /= np.linalg.norm(f)
    up = np.asarray(up, float)
    if abs(np.dot(f, up)) > 0.9995: up = np.array([0, 1.0, 0])
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return r, u, -f      # right, up, back (camera looks along -back)

def rig(t, st, shot=None):
    shot = shot or P.shot_at(t)
    c = shot["cam"]
    u = ease((t - shot["t0"]) / (shot["t1"] - shot["t0"]))
    r, el, az, lens = _l(c["r"], u), _l(c["el"], u), _l(c["az"], u), _l(c["lens"], u)
    if c["frame"] == "rot":
        az = az + math.degrees(st["phi"])
    pivot = {"com": np.zeros(3), "bh1": st["p1"], "bh2": st["p2"]}[c["pivot"]]
    if c["pivot"] != "com" and st["tau"] <= 0:
        pivot = np.zeros(3)
    ele, azr = math.radians(el), math.radians(az)
    off = r * np.array([math.cos(ele) * math.cos(azr), math.cos(ele) * math.sin(azr), math.sin(ele)])
    pos = pivot + off
    target = np.zeros(3) if c.get("target", c["pivot"]) == "com" else pivot
    # gentle handheld drift
    tt = t * 0.35
    pos = pos + 0.004 * r * np.array([math.sin(tt * 1.7), math.cos(tt * 1.3), math.sin(tt * 0.9)])
    R, U, B = look_basis(pos, target)
    return dict(pos=pos, R=R, U=U, B=B, lens=lens, target=target)

def project(cam, p, W):
    """Pixel coords (x right, y down) of world point p. Returns (px,py,dist)."""
    f_px = cam["lens"] / 36.0 * W
    d = np.asarray(p) - cam["pos"]
    xc, yc, zc = d @ cam["R"], d @ cam["U"], d @ cam["B"]   # zc<0 in front
    dist = float(np.linalg.norm(d))
    if zc >= -1e-6: return None
    return W / 2 + f_px * xc / -zc, (W * 9 / 16) / 2 - f_px * yc / -zc, dist
