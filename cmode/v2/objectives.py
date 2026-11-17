"""Objectives v2 (all in [0,1], maximised).
 f1 raga fidelity   : pitch profile (recorded) + interval shape + raga classifier prob + expert-phrase coverage
 f2 gamaka-ready    : symbolic proxy + share of (prev,note,next) contexts present in the written varnam   [still a proxy]
 f3 tala alignment  : parametric (sama / anga onsets, akshara onsets, cadence)
 f4 diversity       : distinct cells + NEW arrangements of real cells (3-cell windows not in varnam)
 f5 structure       : section start/end cells are real section start/end cells; register change between cycles like the varnam"""
import numpy as np, pandas as pd
from common import *
from phase1 import semi, ordinal
from v2 import clf
from v2.tala import TALAS


def events(tokens):
    ev = []
    for i, t in enumerate(tokens):
        if t == "," and ev: ev[-1][1] += 1
        elif t != ",": ev.append([t, 1, i])
    return ev


def js_sim(p, q):
    p, q = np.asarray(p, float) + 1e-9, np.asarray(q, float) + 1e-9; p, q = p / p.sum(), q / q.sum(); m = (p + q) / 2
    return 1 - 0.5 * (float((p * np.log2(p / m)).sum()) + float((q * np.log2(q / m)).sum()))


sim = lambda x, ref, tol: float(np.exp(-abs(x - ref) / tol))


class Scorer:
    def __init__(self, raga, lib, tala="adi", section=None):
        self.raga, self.lib, self.scale, self.tala, self.section = raga, lib, lib.scale, TALAS[tala], section
        g = lib.g; self.sem = {s: semi(raga, s) % 12 for s in self.scale}
        prof = np.array(g["recorded_scale_profile_12bins"], float); idx = sorted(set(self.sem.values()))
        self.prof_idx = idx; self.ref_prof = prof[idx] / prof[idx].sum()
        self.ref_iv = self._iv(lib.avs)
        self.pray4 = {p[i:i + 4] for p in g["prayoga_letter_phrases"] for i in range(len(p) - 3)}
        self.clf = clf.load(); self.cls = list(self.clf.classes_)
        self.ref = self._refs(); st = [t for s in lib.streams.values() for t in s]
        ev = [t.rstrip("^_") for t, _, _ in events(st)]
        self.ctx = {(ev[i - 1], ev[i], ev[i + 1]) for i in range(1, len(ev) - 1)}
        self.pitch = lambda t: semi(raga, t.rstrip("^_")) + (12 if t.endswith("^") else -12 if t.endswith("_") else 0)
        cyc_h = [np.mean([self.pitch(t) for t in a if t != ","]) for a in lib.avs]
        self.ref_dh = float(np.mean(np.abs(np.diff(cyc_h)))) if len(cyc_h) > 1 else 0.0

    def _iv(self, avs):
        h = np.zeros(9)
        for a in avs:
            ev = events(a)
            for x, y in zip(ev, ev[1:]): h[int(np.clip(ordinal(y[0], self.scale) - ordinal(x[0], self.scale), -4, 4)) + 4] += 1
        return h / max(h.sum(), 1)

    def _stats(self, tok, tala=None):
        tala = tala or self.tala
        ev = events(tok); n = len(tok) // tala.subs; d = lambda t: ordinal(t, self.scale)
        leap = [abs(d(b[0]) - d(a[0])) >= 3 for a, b in zip(ev, ev[1:]) if a[1] == 1 and b[1] == 1]
        sust = [abs(d(b[0]) - d(a[0])) <= 2 for a, b in zip(ev, ev[1:]) if a[1] >= 2]
        S = tala.subs; on = tala.onset_subs
        return dict(leap=np.mean(leap) if leap else 0., hold=sum(t == "," for t in tok) / len(tok), glide=np.mean(sust) if sust else 0.,
                    sama=np.mean([tok[k * S] != "," for k in range(n)]), ang=np.mean([tok[k * S + o] != "," for k in range(n) for o in on[1:]] or [1]),
                    akr=np.mean([tok[i] != "," for i in range(0, len(tok), 4)]),
                    ends=[next((t for t in reversed(tok[k * S:(k + 1) * S]) if t != ","), None) for k in range(n)])

    def _refs(self):
        S = [self._stats(a, TALAS["adi"]) for a in self.lib.avs]      # references always come from the written (Adi) varnam
        r = {k: float(np.mean([s[k] for s in S])) for k in ("leap", "hold", "glide", "sama", "ang", "akr")}
        r["end_set"] = {e for s in S for e in s["ends"]}; return r

    def f1(self, tok):
        ev = events(tok)
        if any(t.rstrip("^_") not in self.sem for t, _, _ in ev): return 0.0
        w = np.zeros(12)
        for t, d, _ in ev: w[self.sem[t.rstrip("^_")]] += d
        a, b = js_sim(w[self.prof_idx], self.ref_prof), js_sim(self._iv([tok]), self.ref_iv)
        h = clf.sym_hist(self.raga, [(t, d) for t, d, _ in ev]); p = self.clf.predict_proba(np.sqrt(h)[None])[0]
        c = float(p[self.cls.index(self.raga)]) if self.raga in self.cls else 0.0
        if self.pray4:
            s = "".join(t[0].lower() for t, _, _ in ev); gr = [s[i:i + 4] for i in range(len(s) - 3)]
            pr = float(np.mean([x in self.pray4 for x in gr])) if gr else 0.0
            return 0.3 * a + 0.25 * b + 0.3 * c + 0.15 * pr
        return 0.35 * a + 0.3 * b + 0.35 * c

    def f2(self, tok):
        s, r = self._stats(tok), self.ref
        proxy = np.mean([np.exp(-max(0, s["leap"] - r["leap"]) / .1), sim(s["hold"], r["hold"], .15), sim(s["glide"], r["glide"], .15)])
        ev = [t.rstrip("^_") for t, _, _ in events(tok)]; tri = [(ev[i - 1], ev[i], ev[i + 1]) for i in range(1, len(ev) - 1)]
        return float(0.5 * proxy + 0.5 * (np.mean([t in self.ctx for t in tri]) if tri else 0.))

    def f3(self, tok):
        s, r = self._stats(tok), self.ref; cad = np.mean([e in r["end_set"] for e in s["ends"]])
        return float(.3 * sim(s["sama"], r["sama"], .2) + .2 * sim(s["ang"], r["ang"], .2) + .2 * sim(s["akr"], r["akr"], .15) + .3 * cad)

    def f4(self, tok):
        cells = [tuple(tok[i:i + 4]) for i in range(0, len(tok), 4)]
        tri = [tuple(tok[i:i + 12]) for i in range(0, len(tok) - 11, 4)]
        return float(.5 * len(set(cells)) / len(cells) + .5 * (np.mean([t not in self.lib.tri for t in tri]) if tri else 0.))

    def f5(self, tok):
        S = self.tala.subs; n = len(tok) // S; cells = [tuple(tok[i:i + 4]) for i in range(0, len(tok), 4)]
        if self.section and self.section in self.lib.starts:
            sc = float(cells[0] in self.lib.starts[self.section]) * .5 + float(cells[-1] in self.lib.ends[self.section]) * .5
        else:
            allst = set().union(*self.lib.starts.values()); alle = set().union(*self.lib.ends.values())
            sc = float(cells[0] in allst) * .5 + float(cells[-1] in alle) * .5
        if n > 1:
            h = [np.mean([self.pitch(t) for t in tok[k * S:(k + 1) * S] if t != ","] or [0]) for k in range(n)]
            reg = sim(float(np.mean(np.abs(np.diff(h)))), self.ref_dh, 2.0)
        else: reg = 1.0
        return float(.6 * sc + .4 * reg)

    def __call__(self, tok): return np.array([self.f1(tok), self.f2(tok), self.f3(tok), self.f4(tok), self.f5(tok)])
