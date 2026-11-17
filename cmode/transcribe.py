"""Phase A: pitch contour -> swara sequence.
 track()      Praat pitch (cents re tonic), glitch frames removed
 transcribe() quantise to the raga's own pitch map, find steady segments, merge -> [(note, t0, t1)]
 validate     python transcribe.py --validate   (compares against the WRITTEN varnam notation)"""
import sys, json
import numpy as np, pandas as pd
from common import *
from phase1 import semi

DATA = ROOT / "corpus_data"; HOP = 0.01


def track(audio, tonic):
    import parselmouth
    p = parselmouth.Sound(str(audio)).to_pitch_ac(time_step=HOP, pitch_floor=tonic * 0.6, pitch_ceiling=tonic * 2.6, very_accurate=True,
                                                   silence_threshold=0.03, voicing_threshold=0.5, octave_cost=0.05, octave_jump_cost=0.6, voiced_unvoiced_cost=0.2)
    f = p.selected_array["frequency"]
    c = np.where(f > 0, 1200 * np.log2(np.where(f > 0, f, 1) / tonic), np.nan)
    c[np.flatnonzero(np.abs(np.diff(c)) > 900) + 1] = np.nan
    return c.astype(np.float32)


def note_grid(raga, swaras, extra=None):
    """list of (name, cents) over octaves -1..2. name like 'R2', 'R2^', 'R2_'."""
    out = []
    for s in swaras:
        base = (extra or {}).get(s, semi(raga, s) * 100)
        for o, suf in ((-1, "_"), (0, ""), (1, "^"), (2, "^^")):
            out.append((s + suf, base + 1200 * o))
    return out


def fill_gaps(c, maxgap=8):
    c = c.copy(); n = len(c); i = 0
    while i < n:
        if np.isnan(c[i]):
            j = i
            while j < n and np.isnan(c[j]): j += 1
            if 0 < i and j < n and j - i <= maxgap and abs(c[j] - c[i - 1]) < 150:
                c[i:j] = np.interp(np.arange(i, j), [i - 1, j], [c[i - 1], c[j]])
            i = j
        else: i += 1
    return c


def transcribe_frames(cents, grid, hop=HOP, sm_s=0.05, min_len=0.08, merge_gap=0.10, maxgap_s=0.08):
    """frame-level: fill short gaps, median-smooth, quantise each frame to nearest grid note, merge runs. -> [(name,t0,t1)]"""
    from scipy.ndimage import median_filter
    c = fill_gaps(cents, int(maxgap_s / hop)); names = [g[0] for g in grid]; pit = np.array([g[1] for g in grid])
    k = max(3, int(sm_s / hop) | 1)
    m = median_filter(np.where(np.isnan(c), 0, c), size=k); m[np.isnan(c)] = np.nan
    q = np.full(len(c), -1); v = ~np.isnan(m); q[v] = np.abs(m[v, None] - pit[None, :]).argmin(1)
    out, i, n = [], 0, len(q)
    while i < n:
        if q[i] < 0: i += 1; continue
        j = i
        while j < n and q[j] == q[i]: j += 1
        if out and out[-1][0] == q[i] and i * hop - out[-1][2] <= merge_gap: out[-1][2] = j * hop
        else: out.append([q[i], i * hop, j * hop])
        i = j
    return [(names[a], t0, t1) for a, t0, t1 in out if t1 - t0 >= min_len]


def transcribe(cents, grid, steady_sd=28, win=7, min_len=0.06, merge_gap=0.12):
    """steady segments quantised to nearest grid note; returns list of (name, t0, t1)."""
    c = fill_gaps(cents); n = len(c)
    names = [g[0] for g in grid]; pit = np.array([g[1] for g in grid])
    q = np.full(n, -1)
    sd = np.full(n, np.inf)
    for i in range(win, n - win):
        w = c[i - win:i + win + 1]
        if not np.isnan(w).any(): sd[i] = w.std()
    ok = sd < steady_sd
    med = np.where(ok, np.array([np.nanmedian(c[max(0, i - win):i + win + 1]) if ok[i] else np.nan for i in range(n)]), np.nan)
    v = ~np.isnan(med)
    q[v] = np.abs(med[v, None] - pit[None, :]).argmin(1)
    segs, i = [], 0
    while i < n:
        if q[i] < 0: i += 1; continue
        j = i
        while j < n and q[j] == q[i]: j += 1
        segs.append([q[i], i * HOP, j * HOP]); i = j
    out = []
    for s in segs:                                   # merge same-note segments split by short dropouts
        if out and out[-1][0] == s[0] and s[1] - out[-1][2] <= merge_gap: out[-1][2] = s[2]
        else: out.append(s)
    return [(names[k], a, b) for k, a, b in out if b - a >= min_len]


# ------------------------------------------------------------------ validation against written notation
def aligned_events(uid, raga, notation, structure, ticks):
    rows, cur = [], 0
    for e, (_, x) in enumerate(structure.iterrows()):
        spt = int(x.n_avartanams); span = 4 // spt
        for ci, cyc in enumerate(notation[(raga, x.section)]):
            def tk(k):
                if k >= 32: return ticks[(cur + span) * 8]
                pc, kk = divmod(k, 8 * spt); t = (cur + pc) * 8 + kk // spt
                return ticks[t] + (kk % spt) / spt * (ticks[t + 1] - ticks[t])
            k = 0
            while k < 32:
                if cyc[k][0] == ",": k += 1; continue
                d = 1
                while k + d < 32 and cyc[k + d][0] == ",": d += 1
                rows.append((e, x.section, ci, spt, cyc[k][0], cyc[k][1], tk(k), tk(k + d))); k += d
            cur += span
    return rows


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


def validate(ragas=("mohanam", "kalyani", "abhogi", "begada", "sahana", "saveri", "sri")):
    rec = pd.read_csv(UNIFIED / "recordings.csv", keep_default_na=False)
    tl = pd.read_csv(UNIFIED / "symbolic/tala_annotations.csv", dtype={"label": str}); st = pd.read_csv(UNIFIED / "symbolic/varnam_structure.csv")
    nt = pd.read_csv(UNIFIED / "symbolic/varnam_notation.csv", keep_default_na=False)
    notation = {}
    for (r, sec, av), g in nt.groupby(["raga", "section", "avartanam"], sort=False):
        g = g.sort_values(["akshara", "sub"]); notation.setdefault((r, sec), []).append(list(zip(g.swara, g.octave.astype(int))))
    res = []
    for r in rec[(rec.source == "varnam") & rec.raga.isin(ragas)].itertuples():
        cf = DATA / f"contours/{r.uid}.npy"
        if not cf.exists(): np.save(cf, track(ROOT / r.audio_path, float(r.tonic_hz)))
        c = np.load(cf); sc = sorted({x[0] for x in notation[(r.raga, next(iter(k[1] for k in notation if k[0] == r.raga)))]  [0]} - {","})
        scale = sorted({s for (rg, sec), cs in notation.items() if rg == r.raga for cyc in cs for s, o in cyc if s != ","}, key=lambda s: semi(r.raga, s))
        grid = note_grid(r.raga, scale)
        T = tl[(tl.uid == r.uid) & tl.kind.isin(["manual_cycle_start", "auto_akshara_div"])].copy()
        T["m"] = T.label.str.split(".").str[0].astype(int); T["n"] = T.label.str.split(".").str[1].astype(int)
        T = T.sort_values(["m", "n"]); ticks = T.t_s.values; ticks = np.append(ticks, ticks[-1] + np.median(np.diff(ticks)))
        ev = aligned_events(r.uid, r.raga, notation, st[st.uid == r.uid].sort_values("order"), ticks)
        tr = transcribe(c, grid)
        # (A) aligned accuracy: transcribed note covering most of the notated note's window
        hit_exact = hit_pc = n = 0
        for (e, sec, ci, spt, sw, oc, t0, t1) in ev:
            seg = [(nm, min(b, t1) - max(a, t0)) for nm, a, b in tr if min(b, t1) - max(a, t0) > 0]
            n += 1
            if not seg: continue
            nm = max(seg, key=lambda x: x[1])[0]; base = nm.rstrip("^_")
            want = sw + ("^" * oc if oc > 0 else "_" * (-oc) if oc < 0 else "")
            hit_exact += (nm == want or (oc >= 0 and nm == want.replace("^", "^") ))
            hit_pc += base == sw
        # (B) alignment-free: per performed section, edit-distance similarity between note-name sequences
        sims = []
        for e, g in pd.DataFrame(ev, columns="e sec ci spt sw oc t0 t1".split()).groupby("e"):
            t0, t1 = g.t0.min(), g.t1.max(); a = [x.sw for x in g.itertuples()]
            b = [nm.rstrip("^_") for nm, s, f in tr if s >= t0 - .05 and f <= t1 + .05]
            sims.append(1 - lev(a, b) / max(len(a), len(b), 1))
        res.append(dict(uid=r.uid, raga=r.raga, notated=n, acc_exact=hit_exact / n, acc_swara=hit_pc / n, seq_sim=np.mean(sims), voiced=float(np.mean(~np.isnan(c)))))
        print(res[-1], flush=True)
    D = pd.DataFrame(res); D.to_csv(DATA / "transcription_validation.csv", index=False)
    print(D.groupby("raga")[["acc_exact", "acc_swara", "seq_sim"]].mean().round(3).to_string()); print("ALL", D[["acc_exact", "acc_swara", "seq_sim"]].mean().round(3).to_dict())


def validate_phrases(ragas=("mohanam", "kalyani", "begada", "sri", "sahana", "saveri")):
    """alignment-free test: expert-labelled melodic phrases (carnatic/*.mphrases-manual) vs our transcription of the same time window."""
    rec = pd.read_csv(UNIFIED / "recordings.csv", keep_default_na=False).set_index("uid")
    ph = pd.read_csv(UNIFIED / "symbolic/cmd_mphrases.csv"); ph = ph[ph.raga.isin(ragas)]
    nt = pd.read_csv(UNIFIED / "symbolic/varnam_notation.csv", keep_default_na=False)
    scales = {r: sorted({s for s in g.swara if s != ","}, key=lambda s: semi(r, s)) for r, g in nt.groupby("raga")}
    rows = []
    for uid, g in ph.groupby("uid"):
        z = np.load(UNIFIED / f"contours/{uid}.npz"); c, hop = z["cents"], float(z["hop_s"]); raga = g.raga.iloc[0]
        tr = transcribe_frames(c, note_grid(raga, scales[raga]), hop)
        for r in g.itertuples():
            a = [x for x in r.phrase]
            b = [nm.rstrip("^_")[0].lower() for nm, t0, t1 in tr if t0 >= r.start_s - 0.1 and t1 <= r.start_s + r.dur_s + 0.1]
            bc = [x for i, x in enumerate(b) if i == 0 or x != b[i - 1]]; ac = [x for i, x in enumerate(a) if i == 0 or x != a[i - 1]]
            rows.append(dict(uid=uid, raga=raga, sim=1 - lev(ac, bc) / max(len(ac), len(bc), 1), exp_len=len(ac), got_len=len(bc), src=rec.loc[uid, "pitch_source"]))
    D = pd.DataFrame(rows); D.to_csv(DATA / "transcription_vs_expert_phrases.csv", index=False)
    print(D.groupby("raga").sim.agg(["count", "mean", "median"]).round(3).to_string()); print(D.groupby("src").sim.agg(["count", "mean"]).round(3).to_string())
    print("ALL mean", round(D.sim.mean(), 3), "median", round(D.sim.median(), 3), " share >=0.6:", round((D.sim >= .6).mean(), 3))
    return D


if __name__ == "__main__":
    if "--validate" in sys.argv: validate()
    if "--phrases" in sys.argv: validate_phrases()
