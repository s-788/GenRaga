"""Instrument rendering (violin / veena tone). Pitch curve = written skeleton + REAL gamaka residual curves cut from the varnam
recordings (retrieved by raga, note, duration, same singer when possible). No invented gamaka, no learned model (see gamaka_eval.json).
   render(raga, tokens, out_mp3, instrument='violin'|'veena', akshara_s=0.75)"""
import subprocess, os
import numpy as np, pandas as pd
from scipy.io import wavfile
from scipy.signal import fftconvolve
from common import *
from phase1 import semi
from transcribe import DATA, HOP
from v2.objectives import events

SR, CR = 32000, 100.0
_bank = {}


def bank(raga):
    if raga in _bank: return _bank[raga]
    A = pd.read_csv(DATA / "aligned_notes_dtw.csv"); A = A[A.raga == raga].sort_values(["uid", "entry", "ncycle", "t0"]); B = {}
    for uid, g in A.groupby("uid"):
        R = np.load(DATA / f"contours/{uid}.npy"); t0 = (g.t0 / HOP).astype(int).to_numpy().copy(); t1 = (g.t1 / HOP).astype(int).to_numpy().copy()
        for i in range(len(g) - 1): t1[i] = max(t0[i] + 1, t0[i + 1])
        sw, oc, sg = g.swara.to_numpy(), g.octave.to_numpy(), g.singer.iloc[0]
        for i in range(len(g)):
            a, b = t0[i], t1[i]
            if b - a < 8 or b > len(R): continue
            seg = R[a:b]; tgt = semi(raga, sw[i]) * 100 + 1200 * oc[i]; res = seg - tgt
            ok = ~np.isnan(res) & (np.abs(np.nan_to_num(res)) < 400)
            if ok.mean() < .75: continue
            x = np.arange(len(res)); res = np.interp(x, x[ok], res[ok])
            B.setdefault((sw[i], int(oc[i])), []).append((b - a, res, sg, sw[i - 1] if i else None))
    _bank[raga] = B; return B


def contour(raga, tokens, sub_s, rng, singer=None):
    B = bank(raga); ev = events(tokens); n = int(len(tokens) * sub_s * CR) + 1; sk = np.zeros(n); rs = np.zeros(n); onsets = []; used = 0
    for i, (tok, d, st) in enumerate(ev):
        base = tok.rstrip("^_"); oc = 1 if tok.endswith("^") else -1 if tok.endswith("_") else 0
        a, b = int(st * sub_s * CR), min(n, int((st + d) * sub_s * CR)); L = b - a
        sk[a:b] = semi(raga, base) * 100 + 1200 * oc; onsets.append(a)
        cand = B.get((base, oc)) or []
        if cand and L >= 4:
            pv = ev[i - 1][0].rstrip("^_") if i else None
            sc = [abs(np.log(len(r) / L)) + 0.4 * (p != pv) + 0.4 * (s != singer) + rng.uniform(0, .2) for (_, r, s, p) in cand]
            r = cand[int(np.argmin(sc))][1]; rs[a:b] = np.interp(np.linspace(0, len(r) - 1, L), np.arange(len(r)), r); used += 1
    # smooth only the residual around note boundaries (~60 ms) so curves join without clicks
    k = np.hanning(13)[1:-1]; k /= k.sum(); sm = np.convolve(np.pad(rs, 6, mode="edge"), k, "valid")
    w = np.zeros(n)
    for a in onsets: w[max(0, a - 5):a + 6] = 1
    rs = np.where(w > 0, sm[:n], rs)
    return sk + rs, ev, used / max(len(ev), 1)


def synth(cents, ev, sub_s, tonic, inst, rng):
    n = int(len(cents) / CR * SR) + SR
    f = tonic * 2 ** (np.interp(np.arange(n) / SR, np.arange(len(cents)) / CR, cents) / 1200)
    tt = np.arange(n) / SR
    
    if inst == "violin":
        # Violin: rich harmonic spectrum (14 harmonics) + subtle 5.3 Hz expressive vibrato
        vib = 1.0 + 0.003 * np.sin(2 * np.pi * 5.3 * tt)
        ph = 2 * np.pi * np.cumsum(f * vib) / SR
        harm_weights = [1.0, 0.75, 0.55, 0.40, 0.30, 0.22, 0.16, 0.12, 0.09, 0.07, 0.05, 0.04, 0.03, 0.02]
        w = sum(a * np.sin((k + 1) * ph) for k, a in enumerate(harm_weights))
    elif inst == "veena":
        # Veena: rich plucked string spectrum (16 harmonics)
        ph = 2 * np.pi * np.cumsum(f) / SR
        harm_weights = [1.0, 0.85, 0.65, 0.50, 0.38, 0.28, 0.20, 0.15, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01]
        w = sum(a * np.sin((k + 1) * ph) for k, a in enumerate(harm_weights))
    else:
        # Pure musical tone (additive 3 harmonics with clean envelope)
        ph = 2 * np.pi * np.cumsum(f) / SR
        w = sum(a * np.sin((k + 1) * ph) for k, a in enumerate((1.0, 0.45, 0.20)))

    on = np.array([s for _, _, s in ev]) * sub_s
    off = on + np.array([d for _, d, _ in ev]) * sub_s
    env = np.zeros(n)
    for a, b in zip(on, off):
        i, j = int(a * SR), int(b * SR)
        L = j - i
        if L > 0:
            x = np.arange(L)
            t_note = x / SR
            if inst == "veena":
                # Plucked envelope: sharp 5ms attack followed by organic exponential decay
                att = np.minimum(1.0, x / (0.005 * SR))
                dec = np.exp(-t_note / 0.85) * np.minimum(1.0, (L - x) / (0.01 * SR)).clip(0)
                note_env = att * dec
            elif inst == "violin":
                # Bowed envelope: 15ms smooth bow onset and 20ms release
                note_env = np.minimum(1.0, x / (0.015 * SR)) * np.minimum(1.0, (L - x) / (0.02 * SR)).clip(0)
            else:
                note_env = np.maximum(0, np.minimum(1.0, x / (0.01 * SR)) * np.minimum(1.0, (L - x) / (0.01 * SR)).clip(0))
            env[i:j] = np.maximum(env[i:j], note_env)
            
    return w * env


def tanpura(n, tonic, raga="mohanam"):
    y = np.zeros(n); t0 = np.arange(int(2.2 * SR)) / SR
    # If raga forbids Pa (varjya Pa, like Abhogi), tune tanpura to Madhyama sruti (Sa + Ma, 4/3 ratio)
    pa_ratio = (4.0 / 3.0) if raga == "abhogi" else 0.75
    for j, (ratio, g) in enumerate(((pa_ratio, 0.6), (1.0, 1.0), (1.0, 1.0), (0.5, 0.9))):
        s = sum((1.0 / k) * np.sin(2 * np.pi * tonic * ratio * k * t0 * (1 + 0.0002 * k)) * np.exp(-t0 / (1.1 / (1 + 0.12 * k))) for k in range(1, 20))
        for m in range(int(n / SR / (4 * 1.45)) + 2):
            i = int((m * 4 + j) * 1.45 * SR)
            if i < n: y[i:i + len(s)] += g * s[:n - i]
    return y


def render(raga, tokens, out_mp3, instrument="tone", akshara_s=0.75, tonic=None, seed=0, singer=None):
    rng = np.random.default_rng(seed); sub_s = akshara_s / 4; tonic = tonic or 146.83
    c, ev, cov = contour(raga, tokens, sub_s, rng, singer); y = synth(c, ev, sub_s, tonic, instrument, rng)
    y = y / (np.abs(y).max() + 1e-9) * 0.75 + 0.05 * tanpura(len(y), tonic, raga=raga) / 4
    y = y / np.abs(y).max() * 0.85
    wav = str(out_mp3)[:-4] + ".wav"; wavfile.write(wav, SR, (y * 32767).astype(np.int16))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-q:a", "2", str(out_mp3)], check=True); os.remove(wav)
    return dict(cents=c, events=ev, sub_s=sub_s, tonic=tonic, curve_coverage=cov)
