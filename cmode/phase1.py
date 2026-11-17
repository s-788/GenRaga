"""Phase 1 of C-MODE: corpus tokenisation, raga CFGs, sangati phrase library, verification.
Run:  python phase1.py     (needs carnatic_unified/ built by build_dataset.py)"""
import json, math, random, itertools, collections
import numpy as np, pandas as pd
from common import *

SEMI = {"S": 0, "R1": 1, "R2": 2, "R3": 3, "G1": 2, "G2": 3, "G3": 4, "M1": 5, "M2": 6, "P": 7,
        "D1": 8, "D2": 9, "D3": 10, "D": 9, "N1": 9, "N2": 10, "N3": 11, "N": 10}
# Dataset swara labels are raga-specific (CompMusic). Verified against recorded pitch histograms:
# Abhogi 'G1' sits at ~300 cents (histogram bin 3), not 200 as in the generic table.
RAGA_SEMI = {"abhogi": {"G1": 3},          # Abhogi Ga = sadharana (300 c)
             "mohanam": {"G2": 4},         # Mohanam Ga = antara (400 c); dataset calls it G2
             "kalyani": {"G2": 4, "N2": 11}}  # Kalyani Ga = antara (400 c), Ni = kakali (1100 c)
# Verified by cmode/verify_pitchmap.py against recorded pitch histograms (cmd+rrd).


def semi(raga, sw):
    return RAGA_SEMI.get(raga, {}).get(sw, SEMI[sw])


U = UNIFIED
SUBS_PER_AKSHARA, AKSHARAS = 4, 8            # Adi tala, 4 subdivisions per akshara
CELL, PAIR, HALF, AVART = 4, 8, 16, 32       # phrase-length classes in subdivisions (= 1,2,4,8 aksharas)
LEVELS = {"cell": CELL, "pair": PAIR, "half": HALF}


def tok(sw, octv):  # 'R2',1 -> 'R2^' ; hold -> ','
    return sw if sw == "," else sw + ("^" if octv > 0 else "_" if octv < 0 else "")


def load_corpus():
    n = pd.read_csv(U / "symbolic/varnam_notation.csv", keep_default_na=False)
    corp = {}
    for (raga, sec, av), g in n.groupby(["raga", "section", "avartanam"], sort=False):
        g = g.sort_values(["akshara", "sub"])
        corp.setdefault(raga, {}).setdefault(sec, []).append([tok(s, int(o)) for s, o in zip(g.swara, g.octave)])
    return corp


def ordinal(t, scale):
    """Scale-degree ordinal of a token in the raga (transposition-free contour coordinate)."""
    base, o = t.rstrip("^_"), (1 if t.endswith("^") else -1 if t.endswith("_") else 0)
    return o * len(scale) + scale.index(base)


def unit_feats(u):
    p = [t for t in u if t != ","]
    return (p[0] if p else None), (p[-1] if p else None), u[0] == ","


class RagaGrammar:
    """Non-probabilistic CFG for one raga, terminals = observed phrase units, plus junction constraints.

        Varnam      -> Section+                        (order from singer structure files)
        Section     -> Avartanam^n
        Avartanam   -> Half Half | Half Pair Pair | Pair Pair Half | Pair Pair Pair Pair     (32 subdivisions, Adi tala)
        Half        -> <observed 16-subdivision phrase>      Pair -> <observed 8-sub phrase>
        Pair        -> Cell Cell ;  Cell -> <observed 4-sub phrase>   (fall-back level)
    Junction constraint J: (last swara of X, first swara of Y, Y-starts-with-hold) must have been seen in corpus."""

    def __init__(self, raga, corpus, extra_prayoga=None):
        self.raga = raga
        self.sections = corpus
        avs = [a for s in corpus.values() for a in s]
        self.scale = sorted({t.rstrip("^_") for a in avs for t in a if t != ","}, key=lambda x: (semi(raga, x), x))
        self.lib = {lv: collections.Counter() for lv in LEVELS}
        self.J = set()
        for a in avs:
            assert len(a) == AVART, (raga, len(a))
            for lv, L in LEVELS.items():
                for i in range(0, AVART, L):
                    self.lib[lv][tuple(a[i:i + L])] += 1
            for i in range(CELL, AVART, CELL):
                self.J.add(self._junction(a[i - CELL:i], a[i:i + CELL]))
        self.prayoga = extra_prayoga or set()

    @staticmethod
    def _junction(x, y):
        return (unit_feats(x)[1], unit_feats(y)[0], y[0] == ",")

    def legal(self, x, y):
        return self._junction(x, y) in self.J

    AV_RULES = [("half", "half"), ("half", "pair", "pair"), ("pair", "pair", "half"), ("pair",) * 4]

    def rules_text(self):
        L = [f"# CFG for raga {self.raga}   scale = {' '.join(self.scale)}",
             "Varnam    -> " + " ".join(f"Sec_{s}" for s in self.sections),
             *[f"Sec_{s} -> Avartanam{{{len(v)}}}" for s, v in self.sections.items()],
             *["Avartanam -> " + " ".join(r.capitalize() for r in rule) for rule in self.AV_RULES],
             "Pair      -> Cell Cell   |  <Pair terminals>", "Half      -> Pair Pair   |  <Half terminals>",
             f"# terminals: {', '.join(f'{k}={len(v)}' for k, v in self.lib.items())} distinct units; "
             f"{len(self.J)} legal junctions"]
        return "\n".join(L)

    # ---- generation (uniform over derivations, respecting junction constraint J)
    def generate_avartanam(self, rng, max_try=200):
        for _ in range(max_try):
            rule = rng.choice(self.AV_RULES)
            units = [list(rng.choice(list(self.lib[lv]))) for lv in rule]
            if all(self.legal(units[i], units[i + 1]) for i in range(len(units) - 1)):
                return [t for u in units for t in u]
        return None

    # ---- membership: is the avartanam derivable from the grammar? (DP over unit segmentation)
    def parses(self, av, allow=("cell", "pair", "half"), lib=None):
        lib = lib or self.lib
        n = len(av)
        ok = {0: [None]}
        best = {0: True}
        stack = [(0, None)]
        seen = set()
        while stack:
            i, prev = stack.pop()
            if i == n:
                return True
            for lv in allow:
                L = LEVELS[lv]
                if i % L or i + L > n:
                    continue
                u = tuple(av[i:i + L])
                if u in lib[lv] and (prev is None or self._junction(prev, u) in self.J) and (i + L, u[-CELL:]) not in seen:
                    seen.add((i + L, u[-CELL:]))
                    stack.append((i + L, list(u)))
        return False

    def search_space(self):
        """#distinct avartanam-derivations at phrase level vs. raw note level."""
        h, p = len(self.lib["half"]), len(self.lib["pair"])
        phrase = h * h + 2 * h * p * p + p ** 4
        note = (len(self.scale) * 3 + 1) ** AVART   # per-subdivision: any scale swara in 3 octaves, or hold
        return phrase, note


def build_prayoga(raga):
    """Expert-annotated melodic phrases (CMD mphrases) as letter strings, for f1 (prayoga match)."""
    p = pd.read_csv(U / "symbolic/cmd_mphrases.csv")
    p = p[p.raga == raga]
    return sorted({s for s in p.phrase})


def scale_profile(raga):
    """Data-driven swara usage profile (12 semitone bins) from recorded pitch histograms: used by f1, no Markov prior."""
    r = pd.read_csv(U / "recordings.csv")
    H = pd.read_csv(U / "pitch_class_hist_120.csv", index_col=0)
    u = r[(r.raga == raga) & r.uid.isin(H.index)].uid
    if not len(u):
        return None
    return np.round(H.loc[u].mean().values.reshape(12, 10).sum(1), 4).tolist(), len(u)


def heldout_coverage(raga, corp):
    """Leave-one-section-out: fraction of unseen avartanams derivable, by unit level (generalisation of the library)."""
    res = {lv: [0, 0] for lv in ("cell", "pair", "half", "cell+pair+half")}
    for held in corp:
        train = {k: v for k, v in corp.items() if k != held}
        g = RagaGrammar(raga, train)
        for a in corp[held]:
            for name, allow in (("cell", ("cell",)), ("pair", ("pair",)), ("half", ("half",)), ("cell+pair+half", ("cell", "pair", "half"))):
                res[name][1] += 1
                res[name][0] += g.parses(a, allow)
    return {k: v[0] / v[1] for k, v in res.items()}


def unit_coverage(raga, corp, L):
    """Leave-one-section-out: fraction of held-out L-sub UNITS (not whole avartanams) already in the library (exact)."""
    hit = tot = 0
    for held in corp:
        seen = {tuple(a[i:i + L]) for k, v in corp.items() if k != held for a in v for i in range(0, AVART, L)}
        for a in corp[held]:
            for i in range(0, AVART, L):
                tot += 1; hit += tuple(a[i:i + L]) in seen
    return hit / tot


def shape(u, scale):
    p = [t for t in u if t != ","]
    if not p:
        return tuple(u)
    o0 = ordinal(p[0], scale)
    return tuple("," if t == "," else ordinal(t, scale) - o0 for t in u)


def unit_coverage_shape(raga, corp, L, scale):
    """Same as unit_coverage but units are compared by relative scale-degree contour (transposition-free)."""
    hit = tot = 0
    for held in corp:
        seen = {shape(a[i:i + L], scale) for k, v in corp.items() if k != held for a in v for i in range(0, AVART, L)}
        for a in corp[held]:
            for i in range(0, AVART, L):
                tot += 1; hit += shape(a[i:i + L], scale) in seen
    return hit / tot


def junction_legal_rate(g, rng, n=4000):
    cells = list(g.lib["cell"]); return sum(g.legal(list(rng.choice(cells)), list(rng.choice(cells))) for _ in range(n)) / n


def main():
    PHASE1.mkdir(exist_ok=True); (PHASE1 / "grammars").mkdir(exist_ok=True)
    corp = load_corpus()
    json.dump(corp, open(PHASE1 / "corpus_hierarchy.json", "w"), indent=1, ensure_ascii=False)
    scales = pd.read_csv(U / "ragas.csv").set_index("raga")
    rows, lib_rows, rep = [], [], []
    rng = random.Random(7)
    for raga, c in corp.items():
        g = RagaGrammar(raga, c, extra_prayoga=build_prayoga(raga))
        # data-quality: corpus swaras must be inside the raga's declared arohana/avarohana
        decl = {t.rstrip("^_") for t in (scales.loc[raga, "arohana"] + " " + scales.loc[raga, "avarohana"]).split()}
        stray = sorted(set(g.scale) - {d for d in decl} - {"D" if "D2" in decl or "D1" in decl else ""})
        ph, nt = g.search_space()
        # generation test: every generated avartanam must be 32 subs, in-scale, parse again
        gen = [g.generate_avartanam(rng) for _ in range(500)]
        gen_ok = [x for x in gen if x]
        valid = sum(len(x) == AVART and g.parses(x) for x in gen_ok) / max(len(gen_ok), 1)
        novel = len({tuple(x) for x in gen_ok} - {tuple(a) for s in c.values() for a in s}) / max(len({tuple(x) for x in gen_ok}), 1)
        prof = scale_profile(raga)
        cov = heldout_coverage(raga, c)
        scov = {lv: unit_coverage_shape(raga, c, L, g.scale) for lv, L in LEVELS.items()}
        jr = junction_legal_rate(g, rng)
        ucov = {lv: unit_coverage(raga, c, L) for lv, L in LEVELS.items()}
        gj = {"raga": raga, "scale": g.scale, "arohana": scales.loc[raga, "arohana"], "avarohana": scales.loc[raga, "avarohana"],
              "tala": {"name": "Adi", "aksharas": 8, "subdivisions_per_akshara": 4, "avartanam_subdivisions": 32, "angas": [4, 2, 2]},
              "structure_avartanams": {s: len(v) for s, v in c.items()},
              "av_rules": g.AV_RULES, "legal_junctions": sorted(map(list, [[str(x) for x in j] for j in g.J])),
              "units": {lv: [{"tokens": list(u), "count": n} for u, n in g.lib[lv].most_common()] for lv in g.lib},
              "prayoga_letter_phrases": sorted(g.prayoga), "recorded_scale_profile_12bins": prof[0] if prof else None,
              "n_recordings_profile": prof[1] if prof else 0}
        json.dump(gj, open(PHASE1 / f"grammars/{raga}.json", "w"), indent=1, ensure_ascii=False)
        open(PHASE1 / f"grammars/{raga}.cfg", "w").write(g.rules_text() + "\n")
        for lv in g.lib:
            for i, (u, n) in enumerate(g.lib[lv].most_common()):
                f = unit_feats(list(u))
                lib_rows.append((raga, lv, f"{raga[:3]}_{lv[0]}{i:03d}", " ".join(u), n, f[0], f[1],
                                 " ".join(str(ordinal(t, g.scale) - ordinal(f[0], g.scale)) if t != "," else "," for t in u) if f[0] else ""))
        rows.append(dict(raga=raga, n_avartanams=sum(len(v) for v in c.values()), scale_size=len(g.scale),
                         stray_swaras=",".join(stray) or "-", cells=len(g.lib["cell"]), pairs=len(g.lib["pair"]), halves=len(g.lib["half"]),
                         junctions=len(g.J), phrase_space=f"{ph:.2e}", note_space=f"{nt:.2e}", reduction_log10=round(math.log10(nt / ph), 1),
                         gen_valid=round(valid, 3), gen_novel=round(novel, 3), n_prayoga=len(g.prayoga),
                         ucov_cell=round(ucov["cell"], 2), ucov_pair=round(ucov["pair"], 2), ucov_half=round(ucov["half"], 2),
                         shape_cell=round(scov["cell"], 2), shape_pair=round(scov["pair"], 2), shape_half=round(scov["half"], 2),
                         junction_legal_rate=round(jr, 2),
                         cov_cell_only=round(cov["cell"], 2), cov_pair_only=round(cov["pair"], 2), cov_half_only=round(cov["half"], 2),
                         cov_all=round(cov["cell+pair+half"], 2)))
    pd.DataFrame(lib_rows, columns="raga level id tokens count first_swara last_swara contour_ordinals".split()
                 ).to_csv(PHASE1 / "sangati_phrase_library.csv", index=False)
    df = pd.DataFrame(rows); df.to_csv(PHASE1 / "phase1_metrics.csv", index=False)
    print(df.T.to_string())
    return df


if __name__ == "__main__":
    main()
