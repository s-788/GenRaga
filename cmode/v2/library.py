"""Cell library: every gene is a REAL one-akshara cell from the written varnam. No shifted or mined phrases.
 mode 'strict'  : cell->cell moves only if that adjacency occurs in the varnam (within a section)
 mode 'balanced': a move is allowed if the note-level junction (last swara, first swara, leading hold) occurs in the varnam"""
import json
import numpy as np
from common import *
from phase1 import ordinal, unit_feats


def jk(last, first, hold): return (str(last), str(first), str(bool(hold)))


class CellLibrary:
    def __init__(self, raga, mode="strict"):
        g = json.load(open(PHASE1 / f"grammars/{raga}.json"))
        self.raga, self.mode, self.scale, self.g = raga, mode, g["scale"], g
        self.sections = json.load(open(PHASE1 / "corpus_hierarchy.json"))[raga]          # section -> [cycle(32 tokens)]
        self.avs = [a for s in self.sections.values() for a in s]
        self.tokens = {t for a in self.avs for t in a if t != ","}
        self.J = {tuple(j) for j in g["legal_junctions"]}
        streams = {sec: [t for a in cyc for t in a] for sec, cyc in self.sections.items()}
        self.streams = streams
        self.adj, self.starts, self.ends = set(), {}, {}
        for sec, st in streams.items():
            cs = [tuple(st[i:i + 4]) for i in range(0, len(st), 4)]
            self.adj |= set(zip(cs, cs[1:])); self.starts.setdefault(sec, set()).add(cs[0]); self.ends.setdefault(sec, set()).add(cs[-1])
        cells = sorted({tuple(st[i:i + 4]) for st in streams.values() for i in range(0, len(st), 4)})
        if mode == "strict":                                    # prune cells that cannot continue (until stable)
            while True:
                keep = [c for c in cells if any((c, d) in self.adj for d in cells)]
                if len(keep) == len(cells): break
                cells = keep
        self.cells = cells; self.N = len(cells); self.idx = {c: i for i, c in enumerate(cells)}
        self._m = {}; self.E = self._embed()
        self.cellset = set(cells)
        self.tri = {tuple(st[i:i + 12]) for st in streams.values() for i in range(0, len(st) - 11, 4)}

    def _embed(self):
        rows = []
        for c in self.cells:
            n = [ordinal(t, self.scale) for t in c if t != ","] or [0]
            iv = np.abs(np.diff(n)) if len(n) > 1 else np.array([0])
            rows.append([n[0], n[-1], np.mean(n), min(n), max(n), n[-1] - n[0], len(n) / 4, sum(t == "," for t in c) / 4, iv.mean()])
        E = np.array(rows, float); return (E - E.mean(0)) / (E.std(0) + 1e-9)

    def legal_mask(self, prev):
        if self.mode == "strict":
            if prev not in self._m:
                p = self.cells[prev]; self._m[prev] = np.array([(p, c) in self.adj for c in self.cells])
            return self._m[prev]
        last = unit_feats(list(self.cells[prev]))[1]
        if ("L", last) not in self._m:
            self._m[("L", last)] = np.array([jk(last, unit_feats(list(c))[0], c[0] == ",") in self.J for c in self.cells])
        return self._m[("L", last)]

    def legal(self, a, b): return bool(self.legal_mask(a)[b])

    def sample_gene(self, rng, prev=None):
        m = np.ones(self.N, bool) if prev is None else self.legal_mask(prev)
        c = np.flatnonzero(m); return int(rng.choice(c)) if len(c) else int(rng.integers(self.N))

    def sample(self, rng, G, start=None):
        g = [self.sample_gene(rng, None) if start is None else int(rng.choice(start))]
        while len(g) < G: g.append(self.sample_gene(rng, g[-1]))
        return g

    def tokens_of(self, genes): return [t for i in genes for t in self.cells[i]]

    def is_valid(self, genes):
        return all(self.legal(genes[i - 1], genes[i]) for i in range(1, len(genes))) and all(t == "," or t in self.tokens for t in self.tokens_of(genes))
