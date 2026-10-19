# What Happens When Two Black Holes Collide? - Blender Cycles + GitHub Actions

8-minute film (480 s), 16 shots from 14 camera set-ups, rendered with **Blender Cycles** (`bpy` 5.2.2 as a Python module), sharded across GitHub Actions runners, at **1080p and/or 4K**.

## What is real
- Masses, spin, distance: LIGO/Virgo **GW150914** (36 + 29 Msun -> 62 Msun, ~3 Msun radiated).
- Orbit: quadrupole-formula inspiral (da/dt ~ 1/a^3), so the chirp speed-up is physical; the merger/ringdown shapes are stylised.
- Sky: **NASA SVS Deep Star Maps 2020** (Gaia DR2), downloaded on the runner. Falls back to a procedural sky if the download fails.
- HUD strain plot + audio: **real LIGO Hanford strain** from GWOSC (whitened, 20-500 Hz). Falls back to a synthetic chirp.
- Gravitational lensing of the stars (point-lens deflection per hole) and GW ripples are done in numpy post-processing; Cycles renders the holes, disks and photon rings. Accretion disks are artistic (stated on screen).

## Run it
1. Create a GitHub repo (public is best: free runner minutes), push this folder.
2. Actions -> **Render** -> Run workflow.
   - First run **mode = preview**: 1 still per shot + contact sheet (a few minutes). Check the look, star-map exposure (`SKY_GAMMA`/`SKY_GAIN` env vars in `download_assets.py`).
   - Then **mode = full**, resolution `1080`, `2160` or `1080,2160`.
3. Result: artifact `final-1080p` / `final-2160p` (MP4 + `captions.srt`), and optionally a GitHub Release asset (best for the large 4K file).

## Cost / time (estimates - not measured on GitHub runners)
Locally (2 cores) a 960x540 frame took ~6-15 s. Expect roughly 20-60 s per 1080p frame and 4x that for 4K on a 4-vCPU runner at 64 samples.
11,520 frames (24 fps) = many runner-hours (~100+ for 1080p, ~400+ for 4K). With 20 parallel runners that is several hours (1080p) to more than a day (4K). Levers: `samples` (32 is usually fine with the denoiser), `fps` 24, `shards`, `max_parallel`. Private repos have only 2,000 free minutes/month - use a public repo or reduce scope.
Each shard job must finish in <6 h; at 4K use more shards (128 max).

## Files
- `scripts/config.py` - shot list, captions, channel watermark text, constants
- `scripts/physics.py`, `camrig.py` - orbit/merger/ringdown + camera moves
- `scripts/render_shard.py` - Cycles scene (procedural disks, photon rings), per-frame render + post
- `scripts/postfx.py` - lensing, bloom, tonemap, HUD, captions
- `scripts/download_assets.py`, `make_audio.py`, `plan.py`
- `.github/workflows/render.yml`

## Local test
```
pip install -r requirements.txt
cd scripts && python download_assets.py --out ../assets && python make_audio.py
python render_shard.py --frames 100,2400,5000 --res 540 --samples 16 --assets ../assets --out ../frames
```
(`make_audio.py` takes a few minutes.)

## Known limits
Not tested on a real GitHub runner (my sandbox could not reach NASA/GWOSC; the fallback paths were tested instead). No voice-over: captions are burned in and exported as SRT.
