"""C-MODE: discrete Differential Evolution (feature-space mutation + nearest-legal snap) with NSGA-II selection."""
import numpy as np


def dominates(a, b):
    return np.all(a >= b) and np.any(a > b)


def nd_sort(F):
    n = len(F); dom = np.zeros((n, n), bool)
    ge = (F[:, None, :] >= F[None, :, :]).all(2); gt = (F[:, None, :] > F[None, :, :]).any(2)
    dom = ge & gt                                  # dom[i,j]: i dominates j
    cnt = dom.sum(0); fronts, cur = [], list(np.flatnonzero(cnt == 0)); cnt = cnt.astype(int)
    while cur:
        fronts.append(cur); nxt = []
        for i in cur:
            for j in np.flatnonzero(dom[i]):
                cnt[j] -= 1
                if cnt[j] == 0: nxt.append(j)
        cur = nxt
    return fronts


def crowding(F):
    n, m = F.shape; d = np.zeros(n)
    for k in range(m):
        o = np.argsort(F[:, k]); d[o[0]] = d[o[-1]] = np.inf
        rng = F[o[-1], k] - F[o[0], k]
        if rng > 0 and n > 2:
            d[o[1:-1]] += (F[o[2:], k] - F[o[:-2], k]) / rng
    return d


def nsga_select(pop, F, N):
    keep = []
    for fr in nd_sort(F):
        if len(keep) + len(fr) <= N:
            keep += fr
        else:
            cd = crowding(F[fr]); keep += [fr[i] for i in np.argsort(-cd)[:N - len(keep)]]; break
    return [pop[i] for i in keep], F[keep]


def snap(lib, vec, prev, exclude=None):
    mask = np.ones(lib.N, bool) if prev is None else lib.legal_mask(prev).copy()
    if exclude is not None: mask[exclude] = False
    if not mask.any(): mask[:] = True
    d = ((lib.E - vec) ** 2).sum(1); d[~mask] = np.inf
    return int(np.argmin(d))


def cmode(lib, scorer, n_av=2, pop=60, gens=100, CR=0.7, seed=0, history=False):
    rng = np.random.default_rng(seed); G = 4 * n_av
    P = [lib.sample(rng, G) for _ in range(pop)]
    fit = lambda g: scorer(lib.tokens_of(g))
    F = np.array([fit(g) for g in P]); evals = pop; hist = []
    for gen in range(gens):
        T = []
        for i in range(pop):
            a, b, c = rng.choice([k for k in range(pop) if k != i], 3, replace=False)
            Fm = rng.uniform(0.4, 0.9); jr = rng.integers(G); trial = []
            for j in range(G):
                prev = trial[-1] if trial else None
                if rng.random() < CR or j == jr:
                    cand = snap(lib, lib.E[P[a][j]] + Fm * (lib.E[P[b][j]] - lib.E[P[c][j]]), prev)
                else:
                    cand = P[i][j]
                if prev is not None and not lib.legal(prev, cand):          # repair: nearest legal
                    cand = snap(lib, lib.E[cand], prev, exclude=cand)
                trial.append(cand)
            T.append(trial)
        FT = np.array([fit(g) for g in T]); evals += pop
        P, F = nsga_select(P + T, np.vstack([F, FT]), pop)
        if history: hist.append(F.copy())
    front = nd_sort(F)[0]
    return dict(pop=P, F=F, front=[P[i] for i in front], Ff=F[front], evals=evals, hist=hist)
