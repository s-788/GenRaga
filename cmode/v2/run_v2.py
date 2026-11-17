"""Phase C experiment. python -m v2.run_v2   -> v2_outputs/"""
import json, time, numpy as np, pandas as pd
from common import *
from v2.library import CellLibrary
from v2.objectives import Scorer
from v2.engine import cmode2, random_walk, weighted_ga, hypervolume
OUT = ROOT / "v2_outputs"; OUT.mkdir(exist_ok=True)
NAMES = ["f1_raga", "f2_gamaka_ready", "f3_tala", "f4_diversity", "f5_structure"]
G, POP, GENS, SEEDS = 16, 60, 60, 5
rows = []
for raga in ("mohanam", "kalyani", "abhogi"):
    for mode in ("strict", "balanced"):
        lib = CellLibrary(raga, mode); sc = Scorer(raga, lib); t0 = time.time(); budget = POP * (GENS + 1)
        for seed in range(SEEDS):
            for name, r in (("C-MODE", cmode2(lib, sc, G, POP, GENS, seed=seed)), ("random_walk", random_walk(lib, sc, G, budget, seed)),
                            ("weighted_GA", weighted_ga(lib, sc, G, POP, GENS, seed=seed))):
                best = r["Ff"][np.argmax(r["Ff"].mean(1))]
                verb = np.mean([any(tuple(lib.tokens_of(g)) == tuple(st[i:i + len(lib.tokens_of(g))]) for st in map(tuple, lib.streams.values()) for i in range(0, len(st) - 63, 4)) for g in r["front"]])
                rows.append(dict(raga=raga, mode=mode, method=name, seed=seed, hv=hypervolume(r["Ff"]), front=len(r["front"]),
                                 valid=np.mean([lib.is_valid(g) for g in r["pop"]]), verbatim_2cycles=verb, **dict(zip(NAMES, r["Ff"].mean(0))),
                                 best_mean=float(best.mean())))
        print(raga, mode, "cells", lib.N, f"{time.time()-t0:.0f}s", flush=True)
D = pd.DataFrame(rows); D.to_csv(OUT / "v2_runs.csv", index=False)
A = D.groupby(["raga", "mode", "method"]).mean(numeric_only=True).drop(columns="seed").round(3); A.to_csv(OUT / "v2_summary.csv"); print(A.to_string())
for (raga, mode), g in D.groupby(["raga", "mode"]):
    p = g.pivot(index="seed", columns="method", values="hv"); print(raga, mode, "C-MODE beats random in", int((p["C-MODE"] > p["random_walk"]).sum()), "/5; beats GA in", int((p["C-MODE"] > p["weighted_GA"]).sum()), "/5")
