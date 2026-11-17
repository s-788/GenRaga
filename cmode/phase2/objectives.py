"""Four objectives (all in [0,1], maximised). f2 is a SYMBOLIC PROXY (autoencoder only exists in Phase 3).
Every reference value is measured from the corpus/recordings; no note-transition probability model is used."""
import zlib
import numpy as np
from common import *
from phase1 import semi, ordinal, AVART, CELL, PAIR


def events(tokens):
    """[(token, duration_in_subdivisions, start_sub)] ; ',' extends the previous note."""
    ev = []
    for i, t in enumerate(tokens):
        if t == "," and ev:
            ev[-1][1] += 1
        elif t != ",":
            ev.append([t, 1, i])
    return ev


def js_sim(p, q):
    p, q = np.asarray(p, float) + 1e-9, np.asarray(q, float) + 1e-9
    p, q = p / p.sum(), q / q.sum(); m = (p + q) / 2
    kl = lambda a, b: float((a * np.log2(a / b)).sum())
    return 1 - 0.5 * (kl(p, m) + kl(q, m))


sim = lambda x, ref, tol: float(np.exp(-abs(x - ref) / tol))


class Scorer:
    def __init__(self, raga, lib):
        self.raga, self.lib, self.scale = raga, lib, lib.scale
        g = lib.g
        self.sem = {s: semi(raga, s) % 12 for s in self.scale}
        prof = np.array(g["recorded_scale_profile_12bins"], float)
        idx = sorted(set(self.sem.values()))
        self.ref_prof = prof[idx] / prof[idx].sum(); self.prof_idx = idx
        self.ref_iv = self._intervals([a for a in lib.avs])
        ph = g["prayoga_letter_phrases"]
        self.pray4 = {p[i:i + 4] for p in ph for i in range(len(p) - 3)}
        self.ref = self._corpus_refs()

    # ---- helpers
    def _deg(self, t): return ordinal(t, self.scale)

    def _intervals(self, avs):
        h = np.zeros(9)
        for a in avs:
            ev = events(a)
            for x, y in zip(ev, ev[1:]):
                h[int(np.clip(self._deg(y[0]) - self._deg(x[0]), -4, 4)) + 4] += 1
        return h / max(h.sum(), 1)

    def _stats(self, tokens):
        ev = events(tokens)
        n_av = len(tokens) // AVART
        leap = [(abs(self._deg(b[0]) - self._deg(a[0])) >= 3) for a, b in zip(ev, ev[1:]) if a[1] == 1 and b[1] == 1]
        sust = [abs(self._deg(b[0]) - self._deg(a[0])) <= 2 for a, b in zip(ev, ev[1:]) if a[1] >= 2]
        sama = np.mean([tokens[k * AVART] != "," for k in range(n_av)])
        ang = np.mean([tokens[k * AVART + o] != "," for k in range(n_av) for o in (16, 24)])
        akr = np.mean([tokens[i] != "," for i in range(0, len(tokens), CELL)])
        ends = [next((t for t in reversed(tokens[k * AVART:(k + 1) * AVART]) if t != ","), None) for k in range(n_av)]
        return dict(leap=np.mean(leap) if leap else 0.0, hold=sum(t == "," for t in tokens) / len(tokens),
                    glide=np.mean(sust) if sust else 0.0, sama=sama, ang=ang, akr=akr, ends=ends)

    def _corpus_refs(self):
        S = [self._stats(a) for a in self.lib.avs]
        r = {k: float(np.mean([s[k] for s in S])) for k in ("leap", "hold", "glide", "sama", "ang", "akr")}
        r["end_set"] = {e for s in S for e in s["ends"]}
        return r

    # ---- objectives
    def f1(self, tokens):
        ev = events(tokens)
        if any(t.rstrip("^_") not in self.sem for t, _, _ in ev):      # out-of-scale swara: hard raga violation
            return 0.0
        w = np.zeros(12)
        for t, d, _ in ev:
            w[self.sem[t.rstrip("^_")]] += d
        a = js_sim(w[self.prof_idx], self.ref_prof)
        b = js_sim(self._intervals([tokens]), self.ref_iv)
        if self.pray4:
            s = "".join(t[0].lower() for t, _, _ in ev)
            grams = [s[i:i + 4] for i in range(len(s) - 3)]
            c = float(np.mean([x in self.pray4 for x in grams])) if grams else 0.0
            return 0.4 * a + 0.4 * b + 0.2 * c
        return 0.5 * a + 0.5 * b

    def f2(self, tokens):
        s, r = self._stats(tokens), self.ref
        return float(np.mean([np.exp(-max(0, s["leap"] - r["leap"]) / 0.10), sim(s["hold"], r["hold"], 0.15),
                              sim(s["glide"], r["glide"], 0.15)]))

    def f3(self, tokens):
        s, r = self._stats(tokens), self.ref
        cad = np.mean([e in r["end_set"] for e in s["ends"]])
        return float(0.3 * sim(s["sama"], r["sama"], 0.2) + 0.2 * sim(s["ang"], r["ang"], 0.2)
                     + 0.2 * sim(s["akr"], r["akr"], 0.15) + 0.3 * cad)

    def f4(self, tokens):
        cells = [tuple(tokens[i:i + CELL]) for i in range(0, len(tokens), CELL)]
        uniq = len(set(cells)) / len(cells)
        pairs = [tuple(tokens[i:i + PAIR]) for i in range(0, len(tokens) - PAIR + 1, CELL)]
        nov = float(np.mean([p not in self.lib.corpus_pairs for p in pairs]))
        return 0.5 * uniq + 0.5 * nov

    def __call__(self, tokens):
        return np.array([self.f1(tokens), self.f2(tokens), self.f3(tokens), self.f4(tokens)])
