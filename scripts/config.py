"""Shared configuration: constants, physics units, shot list, captions."""
import math

CHANNEL_NAME = "Technical Talk India"   # watermark text (edit freely)
TITLE = "What Happens When Two Black Holes Collide?"
DURATION_S = 480.0                      # 8 minutes
FPS = 24

# ---- GW150914 (GWOSC GWTC-1 medians) -------------------------------------
M1_SUN, M2_SUN, MF_SUN = 35.6, 30.6, 63.1
MT_SUN = M1_SUN + M2_SUN
# geometric units: total mass M = 1
m1, m2, mf = M1_SUN / MT_SUN, M2_SUN / MT_SUN, MF_SUN / MT_SUN
ETA = m1 * m2
GMSUN_S = 4.925491e-6                   # G*Msun/c^3 in seconds
M_SEC = MT_SUN * GMSUN_S                # one "M" in seconds (3.26e-4 s)
M_KM = MT_SUN * 1.476625                # one "M" in km (97.8 km)

SHADOW = 1.3                            # visible shadow radius = 1.3 * r_s
R1, R2 = SHADOW * 2 * m1, SHADOW * 2 * m2
RF = SHADOW * 2 * mf
A0 = 14.0                               # starting separation (M)
A_CONTACT = R1 + R2

# ---- Shots ----------------------------------------------------------------
# tau = time to merger (M units). kind: lin/log inspiral (tau>0), post (ringdown, t since merger)
def _tau_a():  # tau at A0
    return (5.0 / 256.0) * A0 ** 4 / ETA
TAU0 = _tau_a()

def _logsplit(a, b, n, i):
    la, lb = math.log(a), math.log(b)
    f0, f1 = i / n, (i + 1) / n
    return math.exp(la + (lb - la) * f0), math.exp(la + (lb - la) * f1)

_m = [_logsplit(22.0, 0.05, 3, i) for i in range(3)]

# camera dict: pivot(com|bh1|bh2), target(com|pivot), frame(fixed|rot), r,el,az,lens as (start,end)
SHOTS = [
 dict(id="01_title",   t0=0,   t1=18,  kind="lin", s=(TAU0, TAU0-300),   cam=dict(pivot="com", frame="fixed", r=(90,72), el=(12,18),  az=(20,50),   lens=(50,50))),
 dict(id="02_meet",    t0=18,  t1=50,  kind="lin", s=(TAU0-300, TAU0-800), cam=dict(pivot="bh1", frame="fixed", r=(13,9.5), el=(6,20), az=(200,330), lens=(40,40))),
 dict(id="03_dance",   t0=50,  t1=95,  kind="lin", s=(TAU0-800, TAU0-1600), cam=dict(pivot="com", frame="fixed", r=(40,31), el=(78,88), az=(0,-30), lens=(35,35))),
 dict(id="04_skim",    t0=95,  t1=135, kind="lin", s=(TAU0-1600, TAU0-2450), cam=dict(pivot="com", frame="fixed", r=(34,26), el=(3,7), az=(120,205), lens=(45,45))),
 dict(id="05_ligo",    t0=135, t1=160, kind="lin", s=(TAU0-2450, TAU0-2800), hud_big=True, cam=dict(pivot="com", frame="fixed", r=(56,50), el=(20,26), az=(300,335), lens=(35,35))),
 dict(id="06_chase",   t0=160, t1=185, kind="lin", s=(TAU0-2800, 22.0), cam=dict(pivot="com", frame="rot", r=(22,16), el=(18,10), az=(90,90), lens=(40,40))),
 dict(id="07a_side",   t0=185, t1=195, kind="log", s=_m[0], cam=dict(pivot="com", frame="rot", r=(20,14), el=(5,8), az=(60,60), lens=(45,45))),
 dict(id="07b_top",    t0=195, t1=205, kind="log", s=_m[1], cam=dict(pivot="com", frame="rot", r=(18,12), el=(85,85), az=(0,0), lens=(40,40))),
 dict(id="07c_close",  t0=205, t1=215, kind="log", s=_m[2], cam=dict(pivot="com", frame="rot", r=(17,12), el=(15,25), az=(150,200), lens=(50,50))),
 dict(id="08_ringdown",t0=215, t1=250, kind="post", s=(0,120),  cam=dict(pivot="com", frame="fixed", r=(14,30), el=(10,30), az=(0,60), lens=(45,45))),
 dict(id="09_energy",  t0=250, t1=285, kind="post", s=(120,400), cam=dict(pivot="com", frame="fixed", r=(22,18), el=(25,25), az=(60,250), lens=(50,50))),
 dict(id="10_replay_top", t0=285, t1=325, kind="log", s=(8.0,0.05), cam=dict(pivot="com", frame="rot", r=(10,6), el=(88,88), az=(0,0), lens=(40,40))),
 dict(id="11_replay_pov", t0=325, t1=365, kind="log", s=(12.0,0.05), cam=dict(pivot="bh2", target="com", frame="rot", r=(5.5,4.2), el=(3,9), az=(0,0), lens=(40,40))),
 dict(id="12_ligo2",   t0=365, t1=405, kind="lin", s=(300,0.5), hud_big=True, cam=dict(pivot="com", frame="fixed", r=(60,45), el=(25,15), az=(330,300), lens=(35,35))),
 dict(id="13_aftermath", t0=405, t1=445, kind="post", s=(400,1400), cam=dict(pivot="com", frame="fixed", r=(24,140), el=(20,35), az=(60,140), lens=(35,28))),
 dict(id="14_outro",   t0=445, t1=480, kind="post", s=(1400,2200), cam=dict(pivot="com", frame="fixed", r=(140,180), el=(35,35), az=(140,170), lens=(28,28))),
]

# (t_in, t_out, text)  -- lower-third captions, also exported as SRT
CAPTIONS = [
 (1.5, 8.5,  "WHAT HAPPENS WHEN TWO BLACK HOLES COLLIDE?"),
 (9.0, 16.5, "1.3 billion years ago, two black holes began their final dance."),
 (19, 28,   "Each one is a region where gravity is so strong that nothing - not even light - can escape."),
 (29, 38,   "These two are monsters: 36 and 29 times the mass of our Sun."),
 (39, 49,   "The glowing rings are illustrative gas. Real black holes are invisible - we see the stars they bend."),
 (51, 62,   "They orbit their common centre of mass, each one hauling the other round."),
 (63, 76,   "Einstein predicted in 1915 that moving masses shake spacetime itself."),
 (77, 93,   "Those ripples - gravitational waves - carry energy away, so the orbit slowly shrinks."),
 (96, 108,  "As they spiral in, they speed up. The closer they get, the faster energy leaks away."),
 (109, 121, "Light from distant stars is bent into Einstein rings around each black hole."),
 (122, 134, "Orbital speed is now a large fraction of the speed of light."),
 (136, 147, "September 14, 2015 - 09:50:45 UTC. Two LIGO detectors in the USA feel a tiny shudder."),
 (148, 159, "The whole event in our detectors lasts about a fifth of a second."),
 (161, 172, "Final orbits: the pair now circles each other dozens of times every second."),
 (173, 184, "The universe is about to produce more power than all the stars combined."),
 (186, 197, "SLOW MOTION - thousands of times slower than reality."),
 (198, 206, "The two event horizons touch..."),
 (207, 214, "...and merge into one."),
 (216, 227, "The new black hole is lopsided, and it rings like a struck bell."),
 (228, 240, "This 'ringdown' fades in milliseconds as spacetime settles down."),
 (241, 249, "Final result: one black hole of about 62 solar masses."),
 (251, 263, "But 36 + 29 does not equal 62. Three solar masses vanished."),
 (264, 275, "That mass became pure energy: E = mc^2, about 5 x 10^47 joules."),
 (276, 284, "For a split second, the peak power outshone every star in the observable universe."),
 (286, 296, "REPLAY - the merger seen from directly above."),
 (297, 324, "Watch the horizons stretch toward each other, then fuse."),
 (326, 336, "REPLAY - a view from beside the smaller black hole."),
 (337, 364, "From here, its partner sweeps through the sky at nearly the speed of light."),
 (366, 380, "How do we know? LIGO's 4-km arms measured a stretch of about 10^-18 metres."),
 (381, 394, "That is the real signal from our detector: a rising 'chirp' that ends in a whisper."),
 (395, 404, "In 2017 this discovery won the Nobel Prize in Physics."),
 (406, 420, "The ripples crossed 1.3 billion light-years before reaching Earth."),
 (421, 433, "Since 2015, detectors have caught hundreds of mergers like this."),
 (434, 444, "Each one tests Einstein's gravity in its most extreme form."),
 (446, 458, "The remnant spins at about two-thirds of the maximum allowed rate."),
 (459, 470, "Black holes are the simplest objects in the universe - and still full of secrets."),
 (471, 479, "Thanks for watching - subscribe for more space science."),
]

SOURCES_NOTE = ("Masses, spin, distance: LIGO/Virgo GWTC-1 (GW150914). Star map: NASA SVS Deep Star Maps 2020 "
                "(NASA/GSFC SVS, Gaia DR2 ESA/DPAC). Strain data: GWOSC. Disks & colours are artistic.")
