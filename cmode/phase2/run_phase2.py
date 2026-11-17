"""Phase 2 experiment + verification.  python -m phase2.run_phase2 [--quick]"""
import sys, json, time
import numpy as np, pandas as pd
from common import *
from phase2.library import Library
from phase2.objectives import Scorer, events
from phase2.engine import cmode
from phase2.baselines import random_cfg, weighted_ga, hypervolume
from phase2.export import swara_text, to_midi, render_preview

OUT = ROOT / "phase2_outputs"
import shutil; [p.unlink() for p in OUT.glob('*') if p.is_file()] if OUT.exists() else None; OUT.mkdir(exist_ok=True)
QUICK = "--quick" in sys.argv
RAGAS = ["mohanam", "kalyani", "abhogi"]
N_AV, POP, GENS, SEEDS = 2, 60, (20 if QUICK else 100), (2 if QUICK else 5)
NAMES = ["f1_raga", "f2_gamaka_proxy", "f3_tala", "f4_diversity"]


def discrimination(lib, sc, rng):
    """Can the objectives tell musical from non-musical material? (real vs CFG-random vs shuffled vs note-random)"""
    real = [lib.avs[i] + lib.avs[i + 1] for i in range(0, len(lib.avs) - 1)]
    sets = {"real_corpus": real,
            "cfg_random": [lib.tokens_of(lib.sample(rng, 4 * N_AV)) for _ in range(100)]}
    shuf = []
    for a in real:
        ev = [t for t in a if t != ","]; rng.shuffle(ev); it = iter(ev)
        shuf.append([next(it) if t != "," else "," for t in a])
    sets["shuffled_notes"] = shuf
    toks = sorted(lib.tokens)
    sets["uniform_notes"] = [[(",") if rng.random() < .19 else toks[rng.integers(len(toks))] for _ in range(32 * N_AV)] for _ in range(100)]
    return {k: np.mean([sc(t) for t in v], 0).round(3).tolist() for k, v in sets.items()}


def pick_best(lib, sc, front, Ff):
    ok = [i for i in range(len(front)) if Ff[i][3] >= 0.6] or list(range(len(front)))
    return max(ok, key=lambda i: Ff[i][:3].mean())


summary, disc = [], {}
for raga in RAGAS:
    t0 = time.time(); lib = Library(raga); sc = Scorer(raga, lib)
    disc[raga] = discrimination(lib, sc, np.random.default_rng(1))
    budget = POP * (GENS + 1); rows = []
    for seed in range(SEEDS):
        cm = cmode(lib, sc, N_AV, POP, GENS, seed=seed)
        rc = random_cfg(lib, sc, N_AV, budget, seed=seed)
        ga = weighted_ga(lib, sc, N_AV, POP, GENS, seed=seed)
        for name, r in (("C-MODE", cm), ("random_CFG", rc), ("weighted_GA", ga)):
            valid = np.mean([lib.is_valid(g, N_AV) for g in r["pop"]])
            copies = np.mean([tuple(lib.tokens_of(g)[:32]) in lib.corpus_avs for g in r["front"]])
            shifted = np.mean([np.mean([lib.origin[i] == "shifted" for i in g]) for g in r["front"]])
            rows.append(dict(raga=raga, method=name, seed=seed, hv=hypervolume(r["Ff"]), front=len(r["front"]), valid=valid,
                             copy_of_corpus=copies, shifted_gene_frac=shifted, **dict(zip(NAMES, r["Ff"].mean(0))),
                             **{n + "_max": v for n, v in zip(NAMES, r["Ff"].max(0))}))
        if seed == 0:
            front, Ff = cm["front"], cm["Ff"]
            pd.DataFrame([lib.tokens_of(g) and dict(zip(NAMES, Ff[i]), swara=" ".join(lib.tokens_of(g))) for i, g in enumerate(front)]
                         ).to_csv(OUT / f"{raga}_pareto_front.csv", index=False)
            b = pick_best(lib, sc, front, Ff); tok = lib.tokens_of(front[b])
            open(OUT / f"{raga}_best_swara.txt", "w").write(
                f"# {raga} C-MODE seed0  {dict(zip(NAMES, Ff[b].round(3)))}\n# one line per avartanam (Adi tala, | = akshara)\n" + swara_text(tok) + "\n")
            to_midi(raga, tok, OUT / f"{raga}_best.mid"); render_preview(raga, tok, OUT / f"{raga}_best_preview.mp3")
    df = pd.DataFrame(rows); summary.append(df)
    print(f"{raga} done {time.time()-t0:.0f}s", flush=True)
S = pd.concat(summary); S.to_csv(OUT / "phase2_runs.csv", index=False)
agg = S.groupby(["raga", "method"]).mean(numeric_only=True).drop(columns="seed").round(3)
agg.to_csv(OUT / "phase2_summary.csv"); json.dump(disc, open(OUT / "objective_discrimination.json", "w"), indent=1)
print(agg.to_string()); print(json.dumps(disc, indent=1))
