"""Phase 2 library: 'fillers' = what one gene encodes = 8 subdivisions (2 aksharas = one pair slot).
   A filler is either an observed PAIR phrase, or two observed/shifted CELLS joined with a legal junction.
   Contour-shifted cells (+-1,+-2 scale degrees) enlarge the tiny single-composition vocabulary (Phase 1 finding)."""
import json
import numpy as np
from common import *
from phase1 import semi, ordinal, unit_feats, CELL, PAIR, AVART


def jkey(last, first, hold):
    return (str(last), str(first), str(bool(hold)))


class Filler:
    __slots__ = ("tokens", "origin", "first", "last", "starts_hold")

    def __init__(self, tokens, origin):
        self.tokens, self.origin = tuple(tokens), origin
        f, l, _ = unit_feats(list(tokens))
        self.first, self.last, self.starts_hold = f, l, tokens[0] == ","


class Library:
    ORIGIN_P = {"corpus_pair": 0.4, "cell_combo": 0.4, "shifted": 0.2}   # initial-population sampling weights

    def __init__(self, raga, shifts=(-2, -1, 1, 2)):
        g = json.load(open(PHASE1 / f"grammars/{raga}.json"))
        self.raga, self.scale, self.g = raga, g["scale"], g
        self.n = len(self.scale)
        corp = json.load(open(PHASE1 / "corpus_hierarchy.json"))[raga]
        self.sections = corp
        self.avs = [a for s in corp.values() for a in s]
        self.tokens = {t for a in self.avs for t in a if t != ","}
        self.J = {tuple(j) for j in g["legal_junctions"]}
        cells = [tuple(u["tokens"]) for u in g["units"]["cell"]]
        pairs = [tuple(u["tokens"]) for u in g["units"]["pair"]]
        shifted = {c2 for c in cells for k in shifts for c2 in [self.shift(c, k)] if c2 and c2 not in set(cells)}
        seen, F = set(), []

        def add(t, o):
            if t not in seen:
                seen.add(t); F.append(Filler(t, o))
        for p in pairs:
            add(p, "corpus_pair")
        legal = lambda x, y: jkey(unit_feats(list(x))[1], unit_feats(list(y))[0], y[0] == ",") in self.J
        for x in cells:
            for y in cells:
                if legal(x, y):
                    add(x + y, "cell_combo")
        for x in cells:
            for y in shifted:
                if legal(x, y): add(x + y, "shifted")
                if legal(y, x): add(y + x, "shifted")
        for p in pairs:            # shifted whole pairs
            for k in shifts:
                s = self.shift(p, k)
                if s: add(s, "shifted")
        self.F = F
        self.N = len(F)
        self.origin = np.array([f.origin for f in F])
        self._mask = {}
        self.E = self._embed()
        self.corpus_pairs = {tuple(a[i:i + PAIR]) for a in self.avs for i in range(0, AVART, CELL)}
        self.corpus_avs = {tuple(a) for a in self.avs}

    # --- contour shift: move every note k scale degrees; reject if any token is unseen in the corpus
    def shift(self, u, k):
        out = []
        for t in u:
            if t == ",":
                out.append(t); continue
            o = ordinal(t, self.scale) + k
            oc, i = divmod(o, self.n)
            nt = self.scale[i] + ("^" if oc > 0 else "_" if oc < 0 else "")
            if abs(oc) > 1 or nt not in self.tokens:
                return None
            out.append(nt)
        return tuple(out)

    def _embed(self):
        rows = []
        for f in self.F:
            notes = [ordinal(t, self.scale) for t in f.tokens if t != ","]
            if not notes:
                notes = [0]
            iv = np.abs(np.diff(notes)) if len(notes) > 1 else np.array([0])
            rows.append([notes[0], notes[-1], np.mean(notes), min(notes), max(notes), notes[-1] - notes[0],
                         len(notes) / 8, sum(t == "," for t in f.tokens) / 8, iv.mean()])
        E = np.array(rows, float)
        return (E - E.mean(0)) / (E.std(0) + 1e-9)

    def legal_mask(self, prev_idx):
        """boolean mask over fillers that may follow filler prev_idx (cached by its last swara)."""
        last = self.F[prev_idx].last
        if last not in self._mask:
            self._mask[last] = np.array([jkey(last, f.first, f.starts_hold) in self.J for f in self.F])
        return self._mask[last]

    def legal(self, prev_idx, idx):
        return bool(self.legal_mask(prev_idx)[idx])

    def tokens_of(self, genes):
        return [t for i in genes for t in self.F[i].tokens]

    def sample_gene(self, rng, prev=None):
        mask = np.ones(self.N, bool) if prev is None else self.legal_mask(prev)
        for _ in range(8):
            o = rng.choice(list(self.ORIGIN_P), p=list(self.ORIGIN_P.values()))
            c = np.flatnonzero(mask & (self.origin == o))
            if len(c):
                return int(rng.choice(c))
        c = np.flatnonzero(mask)
        return int(rng.choice(c)) if len(c) else int(rng.integers(self.N))

    def sample(self, rng, n_genes):
        g = []
        for _ in range(n_genes):
            g.append(self.sample_gene(rng, g[-1] if g else None))
        return g

    def is_valid(self, genes, n_av):
        t = self.tokens_of(genes)
        return (len(genes) == 4 * n_av and len(t) == AVART * n_av and all(x == "," or x in self.tokens for x in t)
                and all(self.legal(genes[i - 1], genes[i]) for i in range(1, len(genes))))
