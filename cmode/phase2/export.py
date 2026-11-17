"""Swara notation + MIDI + a flat-pitch preview (.mp3) of an evolved skeleton. NO gamaka: Phase 3 adds it."""
import subprocess, wave
import numpy as np, mido
from scipy.io import wavfile
from common import *
from phase1 import semi
from phase2.objectives import events

AKSHARA_S = 0.75        # seconds per beat; real varnam recordings: median 0.75 s (about 80 beats/min)
SUB_S = AKSHARA_S / 4   # one subdivision (1/4 akshara)


def pitch_semi(raga, tok):
    base = tok.rstrip("^_")
    return semi(raga, base) + (12 if tok.endswith("^") else -12 if tok.endswith("_") else 0)


def swara_text(tokens):
    lines = []
    for a in range(0, len(tokens), 32):
        cyc = tokens[a:a + 32]
        lines.append(" | ".join(" ".join(f"{t:<3}" for t in cyc[i:i + 4]) for i in range(0, 32, 4)))
    return "\n".join(lines)


def to_midi(raga, tokens, path, base=60):
    mid = mido.MidiFile(ticks_per_beat=480); tr = mido.MidiTrack(); mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=int(AKSHARA_S * 1e6)))     # 1 beat = 1 akshara
    tick = 120                                                           # one subdivision
    ev = events(tokens)
    for t, d, _ in ev:
        n = base + pitch_semi(raga, t)
        tr.append(mido.Message("note_on", note=n, velocity=80, time=0))
        tr.append(mido.Message("note_off", note=n, velocity=0, time=d * tick))
    mid.save(path)


def render_preview(raga, tokens, path_mp3, tonic_hz=146.83, sr=22050):
    tot = len(tokens) * SUB_S; y = np.zeros(int((tot + 1) * sr))
    ev = events(tokens)
    for t, d, st in ev:
        f = tonic_hz * 2 ** (pitch_semi(raga, t) / 12)
        n = int(d * SUB_S * sr); tt = np.arange(n) / sr
        w = sum(a * np.sin(2 * np.pi * f * (k + 1) * tt) for k, a in enumerate((1, .45, .2)))
        env = np.minimum(1, np.minimum(tt / .02, (n / sr - tt) / .05).clip(0))
        i = int(st * SUB_S * sr); y[i:i + n] += 0.35 * w[:len(y) - i] * env[:len(y) - i]
    tt = np.arange(len(y)) / sr                                           # tanpura-like drone: Sa + Pa
    y += 0.06 * (np.sin(2 * np.pi * tonic_hz / 2 * tt) + 0.7 * np.sin(2 * np.pi * tonic_hz * 1.5 / 2 * tt))
    y = y / max(1e-9, np.abs(y).max()) * 0.8
    wav = str(path_mp3).replace(".mp3", ".wav"); wavfile.write(wav, sr, (y * 32767).astype(np.int16))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-q:a", "3", str(path_mp3)], check=True)
    import os; os.remove(wav)
