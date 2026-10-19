"""Synthesise the soundtrack: drone pad, physics-locked inspiral chirp, merger boom, ringdown,
whooshes at cuts, and the real LIGO strain (audified) during the detection shots."""
import argparse, math, os, wave
import numpy as np
from scipy import signal
from config import *
import physics as P

SR = 48000

def tone_env(t, a, b, fi=0.5, fo=0.5):
    return np.clip(np.minimum((t - a) / fi, (b - t) / fo), 0, 1)

def main(out, assets):
    n = int(DURATION_S * SR); t = np.arange(n) / SR
    rng = np.random.default_rng(3)
    L = np.zeros(n); R = np.zeros(n)
    # --- pad: slow chord changes (Am - F - C - G) ---
    chords = [(110.0, 130.81, 164.81), (87.31, 110.0, 130.81), (130.81, 164.81, 196.0), (98.0, 123.47, 146.83)]
    for i, base in enumerate(chords * 6):
        a, b = i * 20.0, i * 20.0 + 24.0
        if a >= DURATION_S: break
        env = tone_env(t, a, b, 6, 6)
        for k, f in enumerate(base):
            for det in (-0.4, 0.4):
                ph = 2 * np.pi * (f * (1 + det * 0.003)) * t + rng.uniform(0, 6.28)
                s = np.sin(ph) + 0.25 * np.sin(2 * ph)
                pan = 0.5 + 0.35 * np.sin(2 * np.pi * (0.03 + 0.01 * k) * t + k)
                L += 0.035 * env * s * (1 - pan); R += 0.035 * env * s * pan
    # sub drone
    sub = np.sin(2 * np.pi * 36.7 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * t))
    L += 0.07 * sub; R += 0.07 * sub
    # --- inspiral chirp locked to the visual orbit (per-frame omega) ---
    nf = int(DURATION_S * FPS) + 1
    om = np.array([P.omega_of_tau(P.tau_at(i / FPS)) for i in range(nf)])
    taus = np.array([P.tau_at(i / FPS) for i in range(nf)])
    om_i = np.interp(t, np.arange(nf) / FPS, om)
    tau_i = np.interp(t, np.arange(nf) / FPS, taus)
    om0 = A0 ** -1.5
    f_au = 55.0 * (np.clip(om_i, om0, None) / om0) ** 0.55
    phase = 2 * np.pi * np.cumsum(f_au) / SR
    amp = np.clip(0.04 + 0.16 * (om_i / om0) ** 0.35 / 4.0, 0, 0.35)
    insp = (tau_i > 0) & (t > 12)
    gate = np.convolve(insp.astype(float), np.hanning(2400) / np.hanning(2400).sum(), "same")
    ch = amp * gate * (np.sin(phase) + 0.4 * np.sin(2 * phase))
    L += ch * 0.9; R += ch * 0.9
    # --- merger events: whenever tau crosses 0 (film shots 07c) or the replay ends ---
    ev = np.where((taus[:-1] > 0.2) & (taus[1:] <= 0.2))[0]
    ev = list(ev) + [int(s["t1"] * FPS) - 1 for s in SHOTS if s["id"].startswith(("07c", "10_", "11_"))]
    for e in sorted(set(ev)):
        te = e / FPS
        m = (t >= te) & (t < te + 8)
        tt = t[m] - te
        boom = np.sin(2 * np.pi * (55 * np.exp(-tt * 0.5) + 25) * tt) * np.exp(-tt * 0.9) * 0.55
        ring = np.sin(2 * np.pi * 330 * tt) * np.exp(-tt * 1.2) * 0.12 + np.sin(2 * np.pi * 495 * tt) * np.exp(-tt * 2.0) * 0.05
        nz = rng.normal(0, 1, m.sum()); sos = signal.butter(2, 400, "low", fs=SR, output="sos")
        nz = signal.sosfilt(sos, nz) * np.exp(-tt * 3) * 0.5
        L[m] += boom + ring + nz; R[m] += boom + ring + nz
    # --- whooshes at cuts ---
    for s in SHOTS[1:]:
        c = s["t0"]; m = (t >= c - 0.6) & (t < c + 0.8)
        nz = rng.normal(0, 1, m.sum())
        sos = signal.butter(2, [200, 2200], "band", fs=SR, output="sos"); nz = signal.sosfilt(sos, nz)
        env = np.sin(np.pi * (t[m] - (c - 0.6)) / 1.4) ** 2
        L[m] += 0.05 * nz * env; R[m] += 0.05 * nz[::-1] * env
    # --- real LIGO strain, audified, in the two detection shots ---
    try:
        d = np.load(os.path.join(assets, "strain.npz"), allow_pickle=True)
        h = d["h"].astype(float); tt = d["t"]; fs_s = 4096
        seg = h[(tt > -0.45) & (tt < 0.25)]
        seg = signal.resample_poly(seg, SR // 1024, fs_s // 1024)   # real-time speed
        seg = seg / (np.max(np.abs(seg)) + 1e-9)
        for start in (141.0, 148.0, 372.0, 384.0):
            i0 = int(start * SR); i1 = min(n, i0 + len(seg))
            L[i0:i1] += 0.35 * seg[: i1 - i0]; R[i0:i1] += 0.35 * seg[: i1 - i0]
    except Exception as e:
        print("no strain audio:", e)
    # master: soft clip, fade, normalise
    st = np.vstack([L, R]).T
    st = np.tanh(st * 1.4) / 1.4
    fade = np.clip(np.minimum(t / 2.0, (DURATION_S - t) / 4.0), 0, 1)[:, None]
    st = st * fade; st /= max(1e-9, np.abs(st).max()) / 0.89
    pcm = (st * 32767).astype("<i2")
    with wave.open(os.path.join(out, "soundtrack.wav"), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
    print("wrote soundtrack.wav", len(pcm) / SR, "s")

def srt(out):
    def ts(x):
        h, r = divmod(x, 3600); m, s = divmod(r, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"
    with open(os.path.join(out, "captions.srt"), "w", encoding="utf-8") as f:
        for i, (a, b, txt) in enumerate(CAPTIONS, 1):
            f.write(f"{i}\n{ts(a)} --> {ts(b)}\n{txt}\n\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="../assets"); ap.add_argument("--assets", default="../assets")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    main(a.out, a.assets); srt(a.out)
