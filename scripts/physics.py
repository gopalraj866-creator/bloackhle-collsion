"""Quadrupole-formula inspiral + stylised merger/ringdown. Units: G=c=1, M_total=1."""
import math
import numpy as np
from config import *

_K = 1.6 * (51.2 * ETA) ** -0.375
PHI_END = _K * TAU0 ** 0.625
TAU_C = (5.0 / 256.0) * A_CONTACT ** 4 / ETA     # contact (shadows touch)
OMEGA_RD = 0.58 / mf                              # l=2 QNM (a~0.69)
DAMP = 30.0                                       # visual damping time (M)

def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)

def shot_at(t):
    for s in SHOTS:
        if s["t0"] <= t < s["t1"]:
            return s
    return SHOTS[-1]

def tau_at(t, s=None):
    """Story tau (M) at film time t. Negative values = time since merger."""
    s = s or shot_at(t)
    u = min(max((t - s["t0"]) / (s["t1"] - s["t0"]), 0), 1)
    a, b = s["s"]
    if s["kind"] == "lin":
        return a + (b - a) * u
    if s["kind"] == "log":
        return math.exp(math.log(a) + (math.log(b) - math.log(a)) * u)
    return -(a + (b - a) * u)   # post: negative tau

def state(tau):
    """Full physical/visual state at story time tau."""
    st = dict(tau=tau)
    if tau > 0:
        a = (256.0 / 5.0 * ETA * tau) ** 0.25
        phi = _K * (TAU0 ** 0.625 - tau ** 0.625)
        merge = 0.0
        k = smooth(1 - tau / TAU_C) if tau < TAU_C else 0.0
        r1 = R1 + (RF - R1) * k
        r2 = R2 + (RF - R2) * k
        trd = 0.0
        amp = 1.0 / max(a, 0.3)
    else:
        a = 0.0
        trd = -tau
        phi = PHI_END + 0.12 * trd
        r1 = r2 = RF
        k = 1.0
        amp = 1.3 * math.exp(-trd / 12.0) + 0.03
    st.update(a=a, phi=phi, k=k, trd=trd, amp=amp, r1=r1, r2=r2)
    c, s_ = math.cos(phi), math.sin(phi)
    p1 = np.array([a * m2 * c, a * m2 * s_, 0.0])
    p2 = np.array([-a * m1 * c, -a * m1 * s_, 0.0])
    st["p1"], st["p2"] = p1, p2
    # remnant wobble (quadrupole)
    if tau <= 0:
        e = 0.18 * math.exp(-trd / DAMP) * math.cos(OMEGA_RD * trd)
        st["wob"] = e
    else:
        st["wob"] = 0.0
    # disks
    if tau > 0:
        base = (1 - k)
        st["rout1"] = max(min(3.6 * 2 * m1, 0.42 * a), 0) * base
        st["rout2"] = max(min(3.6 * 2 * m2, 0.42 * a), 0) * base
        st["rout_f"] = 0.0
    else:
        st["rout1"] = st["rout2"] = 0.0
        st["rout_f"] = 3.8 * RF * smooth(trd / 80.0) + 0.0
    st["t_story"] = (TAU0 - tau) if tau > 0 else (TAU0 + trd)
    # HUD numbers
    if tau > 0:
        omega = max(a, 0.3) ** -1.5
    else:
        omega = OMEGA_RD
    st["f_orb"] = omega / (2 * math.pi * M_SEC)
    st["v_c"] = min(0.9, 1.0 / math.sqrt(max(a, 1.0))) if tau > 0 else 0.0
    st["sep_km"] = a * M_KM
    return st

def omega_of_tau(tau):
    if tau > 0:
        return max((256.0 / 5.0 * ETA * tau) ** 0.25, 0.3) ** -1.5
    return OMEGA_RD
