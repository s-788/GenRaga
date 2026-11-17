import numpy as np
from phase2.engine import nd_sort, nsga_select, snap


def random_cfg(lib, scorer, n_av, budget, seed=0):
    rng = np.random.default_rng(seed)
    P = [lib.sample(rng, 4 * n_av) for _ in range(budget)]
    F = np.array([scorer(lib.tokens_of(g)) for g in P])
    fr = nd_sort(F)[0]
    return dict(front=[P[i] for i in fr], Ff=F[fr], F=F, pop=P, evals=budget)


def weighted_ga(lib, scorer, n_av, pop=60, gens=100, pm=0.15, seed=0):
    """Plain GA, equal-weight scalarised fitness, one-point crossover, legal-random mutation."""
    rng = np.random.default_rng(seed); G = 4 * n_av
    fit = lambda g: scorer(lib.tokens_of(g))
    P = [lib.sample(rng, G) for _ in range(pop)]; F = np.array([fit(g) for g in P]); evals = pop
    for _ in range(gens):
        s = F.mean(1); kids = []
        for _ in range(pop):
            pick = lambda: max(rng.choice(pop, 3, replace=False), key=lambda i: s[i])
            a, b = P[pick()], P[pick()]; cut = int(rng.integers(1, G))
            c = a[:cut] + b[cut:]
            for j in range(G):
                if rng.random() < pm:
                    c[j] = lib.sample_gene(rng, c[j - 1] if j else None)
            for j in range(1, G):
                if not lib.legal(c[j - 1], c[j]):
                    c[j] = snap(lib, lib.E[c[j]], c[j - 1], exclude=c[j])
            kids.append(c)
        FK = np.array([fit(g) for g in kids]); evals += pop
        allp, allf = P + kids, np.vstack([F, FK]); o = np.argsort(-allf.mean(1))[:pop]
        P, F = [allp[i] for i in o], allf[o]
    fr = nd_sort(F)[0]
    return dict(front=[P[i] for i in fr], Ff=F[fr], F=F, pop=P, evals=evals)


def hypervolume(F, n=200000, seed=0):
    """Monte-Carlo hypervolume in [0,1]^4 against reference point 0 (all objectives maximised)."""
    rng = np.random.default_rng(seed); X = rng.random((n, F.shape[1]))
    covered = np.zeros(n, bool)
    for f in F:
        covered |= (X <= f).all(1)
    return float(covered.mean())
