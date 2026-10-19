"""Frame post-processing: star-field lensing, composite, bloom, tonemap, HUD, captions."""
import math, os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from config import *
import physics as P
from camrig import project

_FONTS = {}
def font(px, bold=False):
    key = (px, bold)
    if key not in _FONTS:
        for p in (["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else []) + [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]:
            if os.path.exists(p):
                _FONTS[key] = ImageFont.truetype(p, px); break
        else:
            _FONTS[key] = ImageFont.load_default()
    return _FONTS[key]

class Sky:
    def __init__(self, assets):
        self.img = np.load(os.path.join(assets, "sky.npy")).astype(np.float32)
        # rotate the sky so the galactic band crosses the shots nicely
        a = math.radians(35); b = math.radians(-20)
        ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
        Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, cb, -sb], [0, sb, cb]])
        self.Rsky = (Rx @ Rz).T
        s = np.load(os.path.join(assets, "strain.npz"), allow_pickle=True)
        self.st_t, self.st_h = s["t"], s["h"]

    def sample(self, dirs):
        d = dirs @ self.Rsky.T
        lon = np.arctan2(d[..., 1], d[..., 0])
        lat = np.arcsin(np.clip(d[..., 2], -1, 1))
        H, W = self.img.shape[:2]
        mx = ((lon / (2 * math.pi) + 0.5) * W).astype(np.float32)
        my = ((0.5 - lat / math.pi) * H).astype(np.float32)
        return cv2.remap(self.img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

def lensed_sky(sky, cam, st, W, H, strength=0.6):
    ms = 2 if W > 1920 else 1                     # compute deflection map at reduced res
    w, h = W // ms, H // ms
    f_px = cam["lens"] / 36.0 * w
    xs = (np.arange(w, dtype=np.float32) + 0.5 - w / 2)[None, :].repeat(h, 0)
    ys = (np.arange(h, dtype=np.float32) + 0.5 - h / 2)[:, None].repeat(w, 1)
    bx, by = xs.copy(), ys.copy()                 # source-plane offset (px, y down)
    holes = []
    if st["tau"] > 0:
        holes = [(st["p1"], st["r1"]), (st["p2"], st["r2"])]
        if st["k"] > 0.95: holes = [(np.zeros(3), st["r1"])]
    else:
        holes = [(np.zeros(3), RF)]
    for p, rad in holes:
        pr = project(dict(cam, lens=cam["lens"]), p, w)
        if pr is None: continue
        px, py, dist = pr
        rs = rad / SHADOW
        th_e = math.sqrt(2.0 * rs / dist) * strength
        E = th_e * f_px
        if st["tau"] > 0 and st["k"] <= 0.95 and len(holes) == 2:
            pass
        else:
            E *= 1.35  # merged object has twice the mass-ish
        dx, dy = xs - (px - w / 2), ys - (py - h / 2)
        r2 = dx * dx + dy * dy
        eps = (rad / dist * f_px) * 0.6
        fac = E * E / (r2 + eps * eps)
        bx -= dx * fac; by -= dy * fac
    # gravitational-wave ripples (source-plane wobble)
    if st["amp"] > 0.05:
        cpr = project(cam, np.zeros(3), w)
        if cpr is not None:
            cx, cy, dist = cpr
            dx, dy = xs - (cx - w / 2), ys - (cy - h / 2)
            rr = np.sqrt(dx * dx + dy * dy) + 1e-3
            lam = 0.07 * w
            phase = 2 * math.pi * (rr / lam) - 2 * math.pi * 0.55 * st["t_story"] * 0.18
            env = 1.0 / (1.0 + rr / (0.25 * w))
            amp = min(st["amp"], 1.4) * 0.006 * w * env
            bx += dx / rr * np.sin(phase) * amp; by += dy / rr * np.sin(phase) * amp
    # camera looks along -B (forward)
    d = bx[..., None] * cam["R"] + (-by)[..., None] * cam["U"] + f_px * (-cam["B"])
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    dirs = d.astype(np.float32)
    if ms > 1:
        # upsample directions smoothly then sample
        dirs = cv2.resize(dirs, (W, H), interpolation=cv2.INTER_LINEAR)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    return sky.sample(dirs)

def aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)

def bloom(img, W):
    s = W / 1920.0
    out = img.copy()
    acc = np.zeros_like(img)
    for sig, wgt in ((3, 0.30), (9, 0.30), (28, 0.28), (90, 0.20)):
        small = cv2.resize(img, (W // 4, int(W * 9 / 16) // 4), interpolation=cv2.INTER_AREA)
        bl = cv2.GaussianBlur(small, (0, 0), sig * s / 4)
        acc += cv2.resize(bl, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR) * wgt
    return out + acc * 0.35

def composite(rgba, sky_img, exposure=1.0):
    rgb, a = rgba[..., :3], rgba[..., 3:4]
    return rgb + (1.0 - a) * sky_img

def finish(lin, W, H, fade=1.0, seed=0):
    lin = bloom(lin, W) * 1.0
    img = aces(lin * 0.95)
    # subtle grade: cool shadows, warm highlights
    lum = img.mean(axis=2, keepdims=True)
    img = img + (np.array([-0.012, 0.0, 0.02]) * (1 - lum) + np.array([0.02, 0.006, -0.01]) * lum)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = 1.0 - 0.35 * (((xx / W - 0.5) ** 2 + ((yy / H - 0.5) * 0.56) ** 2) * 2.2)
    img = img * v[..., None]
    rng = np.random.default_rng(seed)
    img = img + rng.normal(0, 0.006, (H // 2, W // 2, 1)).astype(np.float32).repeat(2, 0).repeat(2, 1)[:H, :W]
    img = np.clip(img, 0, 1) * fade
    return (np.clip(img, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)

def srgb_encode_fix(img8):  # (kept for API symmetry)
    return img8

def hud_overlay(img8, t, st, shot, sky, fps):
    H, W = img8.shape[:2]
    S = H / 1080.0
    im = Image.fromarray(img8).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    mx = int(48 * S)
    # telemetry panel (top-left)
    lines = [("GW150914 - BINARY BLACK HOLE MERGER", True),
             (f"M1 {M1_SUN:.1f} Msun   M2 {M2_SUN:.1f} Msun", False)]
    if st["tau"] > 0:
        lines += [(f"SEPARATION  {st['sep_km']:,.0f} km", False),
                  (f"GW FREQUENCY  {2*st['f_orb']:,.0f} Hz", False),
                  (f"ORBITAL SPEED  {st['v_c']*100:.0f}% of c", False),
                  (f"TIME TO MERGER  {st['tau']*M_SEC*1000:,.1f} ms", False)]
    else:
        lines += [("REMNANT  62 Msun BLACK HOLE", False), (f"RINGDOWN  +{st['trd']*M_SEC*1000:,.1f} ms", False),
                  ("SPIN  0.67", False)]
    # slow-motion factor from local dτ/dt
    dt = 1.0 / fps
    tau2 = P.tau_at(t + dt, shot)
    slow = (abs(st["tau"] - tau2) * M_SEC) / dt
    if slow > 0: lines.append((f"SLOW MOTION  x{1.0/slow:,.0f}" if slow < 1 else "TIME LAPSE", False))
    fs, fb = font(int(21 * S)), font(int(22 * S), True)
    wbox = int(470 * S); hbox = int(len(lines) * 32 * S + 24 * S)
    d.rounded_rectangle([mx, mx, mx + wbox, mx + hbox], int(8 * S), fill=(5, 8, 14, 150), outline=(120, 170, 255, 120))
    y = mx + int(12 * S)
    for txt, b in lines:
        d.text((mx + int(16 * S), y), txt, font=fb if b else fs, fill=(150, 200, 255, 255) if b else (225, 235, 245, 255))
        y += int(32 * S)
    # strain plot (top-right)
    big = shot.get("hud_big")
    pw, ph = int((720 if big else 430) * S), int((190 if big else 110) * S)
    x0, y0 = W - mx - pw, mx
    d.rounded_rectangle([x0, y0, x0 + pw, y0 + ph + int(34 * S)], int(8 * S), fill=(5, 8, 14, 150), outline=(120, 170, 255, 120))
    d.text((x0 + int(12 * S), y0 + int(6 * S)), "LIGO HANFORD - H1 STRAIN (whitened 20-500 Hz)", font=font(int(15 * S)), fill=(150, 200, 255, 255))
    tt, hh = sky.st_t, sky.st_h
    sel = slice(None, None, max(1, len(tt) // 700))
    xs = x0 + (tt[sel] + 1.2) / 1.6 * pw
    ys = y0 + int(34 * S) + ph / 2 - hh[sel] * (ph / 2 - 6 * S)
    d.line(list(zip(xs.tolist(), ys.tolist())), fill=(120, 220, 255, 255), width=max(1, int(2 * S)))
    cur = -(st["tau"] if st["tau"] > 0 else -st["trd"]) * M_SEC
    cx = x0 + (min(max(cur, -1.2), 0.4) + 1.2) / 1.6 * pw
    d.line([cx, y0 + int(30 * S), cx, y0 + ph + int(34 * S)], fill=(255, 190, 90, 255), width=max(1, int(2 * S)))
    d.text((x0 + pw - int(70 * S), y0 + ph + int(36 * S) - int(22 * S)), "merger", font=font(int(13 * S)), fill=(255, 190, 90, 220))
    # captions
    for (a, b, txt) in CAPTIONS:
        if a <= t <= b:
            f = min(1.0, (t - a) / 0.5, (b - t) / 0.5)
            fo = font(int(34 * S), True)
            tw = d.textlength(txt, font=fo)
            lim = W - 2 * mx
            words, lns, cur_l = txt.split(), [], ""
            for w_ in words:
                tr = (cur_l + " " + w_).strip()
                if d.textlength(tr, font=fo) > lim * 0.9 and cur_l:
                    lns.append(cur_l); cur_l = w_
                else: cur_l = tr
            lns.append(cur_l)
            bh = int(len(lns) * 46 * S + 24 * S)
            by = H - mx - bh - int(30 * S)
            tmax = max(d.textlength(l_, font=fo) for l_ in lns)
            bx0 = (W - tmax) / 2 - 24 * S
            d.rounded_rectangle([bx0, by, bx0 + tmax + 48 * S, by + bh], int(10 * S), fill=(0, 0, 0, int(150 * f)))
            yy = by + int(12 * S)
            for l_ in lns:
                d.text(((W - d.textlength(l_, font=fo)) / 2, yy), l_, font=fo, fill=(255, 255, 255, int(255 * f)))
                yy += int(46 * S)
    # watermark + footer
    d.text((W - mx - d.textlength(CHANNEL_NAME, font=font(int(22 * S), True)), H - mx - int(14 * S)), CHANNEL_NAME,
           font=font(int(22 * S), True), fill=(255, 255, 255, 170))
    if t > DURATION_S - 14:
        fo = font(int(18 * S))
        for i, ln in enumerate([SOURCES_NOTE[:105], SOURCES_NOTE[105:]]):
            d.text((mx, mx + int(250 * S) + i * int(24 * S)), ln.strip(), font=fo, fill=(200, 210, 225, 200))
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))
