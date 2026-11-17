"""C-MODE v2 on cell genes. DE mutation in cell-feature space, snap to nearest LEGAL cell, binomial crossover, repair, NSGA-II."""
import numpy as np
from phase2.engine import nd_sort, nsga_select, snap


def cmode2(lib, scorer, G, pop=60, gens=60, CR=0.7, seed=0):
    rng = np.random.default_rng(seed); fit = lambda g: scorer(lib.tokens_of(g))
    P = [lib.sample(rng, G) for _ in range(pop)]; F = np.array([fit(g) for g in P]); ev = pop
    for _ in range(gens):
        T = []
        for i in range(pop):
            a, b, c = rng.choice([k for k in range(pop) if k != i], 3, replace=False); Fm = rng.uniform(.4, .9); jr = rng.integers(G); t = []
            for j in range(G):
                prev = t[-1] if t else None
                cand = snap(lib, lib.E[P[a][j]] + Fm * (lib.E[P[b][j]] - lib.E[P[c][j]]), prev) if (rng.random() < CR or j == jr) else P[i][j]
                if prev is not None and not lib.legal(prev, cand): cand = snap(lib, lib.E[cand], prev, exclude=cand)
                t.append(cand)
            T.append(t)
        FT = np.array([fit(g) for g in T]); ev += pop
        P, F = nsga_select(P + T, np.vstack([F, FT]), pop)
    fr = nd_sort(F)[0]
    return dict(pop=P, F=F, front=[P[i] for i in fr], Ff=F[fr], evals=ev)


def random_walk(lib, scorer, G, budget, seed=0):
    rng = np.random.default_rng(seed); P = [lib.sample(rng, G) for _ in range(budget)]
    F = np.array([scorer(lib.tokens_of(g)) for g in P]); fr = nd_sort(F)[0]
    return dict(pop=P, F=F, front=[P[i] for i in fr], Ff=F[fr], evals=budget)


def weighted_ga(lib, scorer, G, pop=60, gens=60, pm=.15, seed=0):
    rng = np.random.default_rng(seed); fit = lambda g: scorer(lib.tokens_of(g))
    P = [lib.sample(rng, G) for _ in range(pop)]; F = np.array([fit(g) for g in P]); ev = pop
    for _ in range(gens):
        s = F.mean(1); kids = []
        for _ in range(pop):
            pick = lambda: max(rng.choice(pop, 3, replace=False), key=lambda i: s[i])
            a, b = P[pick()], P[pick()]; cut = int(rng.integers(1, G)); c = a[:cut] + b[cut:]
            for j in range(G):
                if rng.random() < pm: c[j] = lib.sample_gene(rng, c[j - 1] if j else None)
            for j in range(1, G):
                if not lib.legal(c[j - 1], c[j]): c[j] = snap(lib, lib.E[c[j]], c[j - 1], exclude=c[j])
            kids.append(c)
        FK = np.array([fit(g) for g in kids]); ev += pop; A, AF = P + kids, np.vstack([F, FK]); o = np.argsort(-AF.mean(1))[:pop]
        P, F = [A[i] for i in o], AF[o]
    fr = nd_sort(F)[0]
    return dict(pop=P, F=F, front=[P[i] for i in fr], Ff=F[fr], evals=ev)


def hypervolume(F, n=200000, seed=0):
    rng = np.random.default_rng(seed); X = rng.random((n, F.shape[1])); cov = np.zeros(n, bool)
    for f in F: cov |= (X <= f).all(1)
    return float(cov.mean())
