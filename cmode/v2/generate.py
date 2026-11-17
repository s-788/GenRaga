"""Compose a full varnam-shaped piece: each section (pallavi, anupallavi, ...) is evolved with C-MODE using the section template
(start/end cells must be real section start/end cells), then all sections are rendered as violin or veena."""
import numpy as np
from common import *
from v2.library import CellLibrary
from v2.objectives import Scorer
from v2.engine import cmode2
from v2.render import render
from v2.tala import TALAS
from phase2.export import swara_text

DEFAULT_SECTIONS = ("pallavi", "anupallavi", "muktayiswaram", "charanam")


def pick(F, fronts):
    ok = [i for i in range(len(F)) if F[i][3] >= 0.5] or list(range(len(F)))
    return max(ok, key=lambda i: F[i].mean())


def compose(raga, sections=DEFAULT_SECTIONS, mode="strict", tala="adi", seed=0, pop=40, gens=40):
    lib = CellLibrary(raga, mode); ta = TALAS[tala]; toks, report = [], []
    for k, sec in enumerate(sections):
        ncyc = len(lib.sections.get(sec, [0, 0])) ; sc = Scorer(raga, lib, tala, section=sec)
        r = cmode2(lib, sc, ta.aksharas * ncyc, pop, gens, seed=seed + k); b = pick(r["Ff"], r["front"])
        t = lib.tokens_of(r["front"][b]); toks += t; report.append(dict(section=sec, cycles=ncyc, scores=[round(float(x), 3) for x in r["Ff"][b]]))
    return toks, report, lib


def generate(raga, out_mp3, instrument="tone", mode="strict", seed=0, akshara_s=0.75):
    toks, rep, lib = compose(raga, mode=mode, seed=seed)
    info = render(raga, toks, out_mp3, instrument, akshara_s=akshara_s, seed=seed)
    return dict(tokens=toks, swara=swara_text(toks), sections=rep, curve_coverage=info["curve_coverage"], mode=mode, instrument=instrument)
