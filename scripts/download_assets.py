"""Fetch real data (NASA star map, LIGO strain) with procedural fallbacks.
Outputs into --out: sky.npy (float16 RGB equirect, display-referred), strain.npz, sky_info.json"""
import argparse, gzip, io, json, os, sys, urllib.request
import numpy as np

NASA = "https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/starmap_2020_{res}.exr"
GWOSC = ("https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3/"
         "H-H1_GWOSC_4KHZ_R1-1126259447-32.txt.gz")
GPS_EVENT, GPS_START = 1126259462.4, 1126259447.0

def fetch(url, dest, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "bh-collision-render/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    f.write(b)
            return True
        except Exception as e:
            print(f"  download failed ({i+1}/{tries}): {e}", file=sys.stderr)
    return False

def fallback_sky(w=4096, h=2048, seed=7):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    # Milky-Way-like band: tilted great circle with noise
    ys, xs = np.mgrid[0:h, 0:w]
    lon = (xs / w - 0.5) * 2 * np.pi; lat = (0.5 - ys / h) * np.pi
    x, y, z = np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)
    n = np.array([0.35, -0.2, 0.91]); n /= np.linalg.norm(n)
    d = x * n[0] + y * n[1] + z * n[2]
    band = np.exp(-(d / 0.16) ** 2)
    import cv2
    noise = cv2.resize(rng.random((64, 128)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    noise2 = cv2.resize(rng.random((256, 512)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    glow = band * (0.25 + 0.6 * noise + 0.3 * noise2) * 0.12
    img[..., 0] = glow * 1.0; img[..., 1] = glow * 0.9; img[..., 2] = glow * 0.8
    n_st = 60000
    sx = rng.integers(0, w, n_st); 
    sy = np.clip((np.arccos(rng.uniform(-1, 1, n_st)) / np.pi * h).astype(int), 0, h - 1)
    # concentrate along band
    keep = rng.random(n_st) < (0.25 + band[sy, sx] * 3)
    sx, sy = sx[keep], sy[keep]
    mag = rng.pareto(2.2, len(sx)) * 0.05 + 0.03
    col = rng.uniform(0.7, 1.0, (len(sx), 3)); col[:, 2] = np.clip(col[:, 2] + rng.uniform(-.1, .25, len(sx)), 0, 1)
    for k in range(len(sx)):
        img[sy[k], sx[k]] += mag[k] * col[k]
    img = cv2.GaussianBlur(img, (0, 0), 0.8) * 2.2
    return np.clip(img, 0, 8)

def prep_sky(out, res):
    exr = os.path.join(out, f"starmap_{res}.exr")
    info = {"source": "fallback"}
    sky = None
    if not os.path.exists(exr):
        print(f"Downloading NASA SVS Deep Star Map {res} ..."); fetch(NASA.format(res=res), exr)
    if os.path.exists(exr) and os.path.getsize(exr) > 1e6:
        try:
            os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
            import cv2
            im = cv2.imread(exr, cv2.IMREAD_UNCHANGED)
            if im is None: raise RuntimeError("cv2 could not read EXR")
            im = im[..., :3][..., ::-1].astype(np.float32)          # BGR->RGB
            lum = im.mean(axis=2)
            p = float(np.percentile(lum, 99.97)) or 1.0
            gamma = float(os.environ.get("SKY_GAMMA", "0.6"))
            gain = float(os.environ.get("SKY_GAIN", "1.0"))
            sky = np.clip((im / p) ** gamma, 0, 6) * gain
            info = {"source": f"NASA SVS 4851 starmap_2020_{res}.exr", "p9997": p, "gamma": gamma, "gain": gain}
        except Exception as e:
            print("EXR processing failed, using fallback:", e, file=sys.stderr)
    if sky is None:
        sky = fallback_sky()
    np.save(os.path.join(out, "sky.npy"), sky.astype(np.float16))
    json.dump(info, open(os.path.join(out, "sky_info.json"), "w"))
    print("sky:", info, sky.shape)

def prep_strain(out):
    from scipy import signal
    raw = os.path.join(out, "H1_strain.txt.gz")
    fs = 4096
    data = None
    if not os.path.exists(raw):
        print("Downloading LIGO H1 strain ..."); fetch(GWOSC, raw)
    try:
        data = np.loadtxt(raw)
        assert len(data) == 32 * fs
        src = "GWOSC GW150914 H1 4kHz"
    except Exception as e:
        print("strain unavailable, synthetic chirp:", e, file=sys.stderr)
        t = np.arange(32 * fs) / fs - 15.4
        tc = np.clip(-t, 1e-3, None); f = 35 * (0.2 / np.maximum(tc, 0.004)) ** 0.375
        ph = 2 * np.pi * np.cumsum(f) / fs
        data = np.random.default_rng(1).normal(0, 1, t.size) * 1e-21 + 1e-19 * np.where(t < 0.02, np.exp(-np.abs(t)*8) * np.sin(ph), 0) * (t < 0.01)
        src = "synthetic"
    # whiten (Welch PSD) + bandpass 20-500 Hz
    win = signal.windows.tukey(len(data), 0.1)
    f, pxx = signal.welch(data, fs, nperseg=4 * fs)
    dd = np.fft.rfft(data * win); fr = np.fft.rfftfreq(len(data), 1 / fs)
    psd = np.interp(fr, f, pxx)
    w = np.fft.irfft(dd / np.sqrt(psd), n=len(data))
    sos = signal.butter(4, [20, 500], btype="band", fs=fs, output="sos")
    w = signal.sosfiltfilt(sos, w)
    t = np.arange(len(w)) / fs + GPS_START - GPS_EVENT   # 0 = merger
    m = (t > -1.2) & (t < 0.4)
    w = w[m] / np.max(np.abs(w[(t > -0.3) & (t < 0.1)]))
    np.savez(os.path.join(out, "strain.npz"), t=t[m].astype(np.float32), h=w.astype(np.float32), src=src)
    print("strain:", src, m.sum(), "samples")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../assets"); ap.add_argument("--sky-res", default="8k")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    prep_sky(a.out, a.sky_res); prep_strain(a.out)
