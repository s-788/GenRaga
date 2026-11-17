"""Phase E1: align the written notation to the recorded pitch contour (banded DTW around the tala ticks) and measure the gain.
   python curves2.py   -> corpus_data/aligned_notes_dtw.csv , corpus_data/alignment_report.json"""
import json
import numpy as np, pandas as pd
from common import *
from phase1 import semi
from transcribe import DATA, HOP, aligned_events, track

TARGET = ("mohanam", "kalyani", "abhogi")
B, PEN = 35, 6.0


def circ(a, b): return np.abs(((a - b + 600) % 1200) - 600)


def skeleton(ev, n, pitch):
    s = np.full(n, np.nan)
    for (_, _, _, _, sw, oc, t0, t1) in ev:
        s[max(0, int(t0 / HOP)):min(n, int(t1 / HOP))] = pitch(sw, oc)
    return s


def dtw_map(S, R, lo, hi):
    """banded DTW over skeleton frames lo..hi against real frames (offset in [-B,B]); returns real index for each skeleton frame."""
    m = hi - lo; W = 2 * B + 1; offs = np.arange(-B, B + 1); INF = 1e9
    D = np.full((m, W), INF); bp = np.zeros((m, W), np.int8)
    def cost(i):
        r = lo + i + offs; ok = (r >= 0) & (r < len(R)); c = np.full(W, 300.0)
        rv = np.where(ok, R[np.clip(r, 0, len(R) - 1)], np.nan)
        d = np.minimum(circ(rv, S[lo + i]), 300.0); d = np.where(np.isnan(rv), 120.0, d); return np.where(ok, d, 300.0)
    D[0] = cost(0) + 0.02 * np.abs(offs)
    for i in range(1, m):
        prev = D[i - 1]; same = prev; up = np.full(W, INF); dn = np.full(W, INF)
        up[:-1] = prev[1:] + PEN; dn[1:] = prev[:-1] + PEN               # offset +1 / -1
        st = np.stack([same, up, dn]); k = st.argmin(0); D[i] = st[k, np.arange(W)] + cost(i) + 0.02 * np.abs(offs); bp[i] = k
    j = int(D[-1].argmin()); path = np.zeros(m, int)
    for i in range(m - 1, -1, -1):
        path[i] = j; k = bp[i, j]
        if k == 1: j += 1
        elif k == 2: j -= 1
        j = int(np.clip(j, 0, W - 1))
    return lo + np.arange(m) + offs[path]


def note_quality(R, rows, key0, key1, tol=60):
    ok = tot = 0
    for r in rows:
        a, b = int(r[key0] / HOP), int(r[key1] / HOP); n = b - a
        if n < 6: continue
        seg = R[a + int(.25 * n):a + int(.85 * n) + 1]; seg = seg[~np.isnan(seg)]
        if len(seg) < 3: continue
        tot += 1; ok += circ(np.median(seg), r["tgt"]) <= tol
    return ok / max(tot, 1), tot


def main():
    rec = pd.read_csv(UNIFIED / "recordings.csv", keep_default_na=False)
    tl = pd.read_csv(UNIFIED / "symbolic/tala_annotations.csv", dtype={"label": str}); st = pd.read_csv(UNIFIED / "symbolic/varnam_structure.csv")
    nt = pd.read_csv(UNIFIED / "symbolic/varnam_notation.csv", keep_default_na=False); notation = {}
    for (r, sec, av), g in nt.groupby(["raga", "section", "avartanam"], sort=False):
        g = g.sort_values(["akshara", "sub"]); notation.setdefault((r, sec), []).append(list(zip(g.swara, g.octave.astype(int))))
    out, rep = [], []
    for r in rec[(rec.source == "varnam")].itertuples():
        if r.raga not in TARGET: continue
        cf = DATA / f"contours/{r.uid}.npy"
        R = np.load(cf) if cf.exists() else track(ROOT / r.audio_path, float(r.tonic_hz))
        T = tl[(tl.uid == r.uid) & tl.kind.isin(["manual_cycle_start", "auto_akshara_div"])].copy()
        T["m"] = T.label.str.split(".").str[0].astype(int); T["n"] = T.label.str.split(".").str[1].astype(int); T = T.sort_values(["m", "n"])
        tk = T.t_s.values; tk = np.append(tk, tk[-1] + np.median(np.diff(tk)))
        ev = aligned_events(r.uid, r.raga, notation, st[st.uid == r.uid].sort_values("order"), tk)
        pitch = lambda sw, oc, rg=r.raga: semi(rg, sw) * 100 + 1200 * oc
        S = skeleton(ev, len(R), pitch); lo, hi = int(ev[0][6] / HOP), min(len(R) - 1, int(ev[-1][7] / HOP))
        mp = dtw_map(S, R, lo, hi - 1)
        rows = []
        for (e, sec, ci, spt, sw, oc, t0, t1) in ev:
            a, b = int(t0 / HOP) - lo, int(t1 / HOP) - lo
            a, b = int(np.clip(a, 0, len(mp) - 1)), int(np.clip(b, 0, len(mp) - 1))
            rows.append(dict(uid=r.uid, raga=r.raga, singer=r.artist, entry=e, section=sec, ncycle=ci, spt=spt, swara=sw, octave=oc, t0_tick=t0, t1_tick=t1,
                             t0=mp[a] * HOP, t1=max(mp[b], mp[a] + 1) * HOP, tgt=pitch(sw, oc)))
        q0, n0 = note_quality(R, rows, "t0_tick", "t1_tick"); q1, n1 = note_quality(R, rows, "t0", "t1")
        shift = np.median([abs(x["t0"] - x["t0_tick"]) for x in rows])
        rep.append(dict(uid=r.uid, notes=len(rows), within60_ticks=q0, within60_dtw=q1, median_shift_s=shift)); print(rep[-1], flush=True)
        out += rows
    A = pd.DataFrame(out); A.to_csv(DATA / "aligned_notes_dtw.csv", index=False); P = pd.DataFrame(rep)
    summ = dict(recordings=len(P), notes=int(P.notes.sum()), within60_ticks=float(P.within60_ticks.mean()), within60_dtw=float(P.within60_dtw.mean()), median_shift_s=float(P.median_shift_s.median()))
    json.dump(summ, open(DATA / "alignment_report.json", "w"), indent=1); print(summ)


if __name__ == "__main__":
    main()
