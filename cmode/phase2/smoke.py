import time, numpy as np
from phase2.library import Library
from phase2.objectives import Scorer
from phase2.engine import cmode
for raga in ("mohanam", "kalyani", "abhogi"):
    t = time.time(); lib = Library(raga); print(raga, "fillers", lib.N, {o: int((lib.origin == o).sum()) for o in set(lib.origin)}, f"{time.time()-t:.1f}s")
    sc = Scorer(raga, lib); print(" refs", {k: (round(v, 2) if not isinstance(v, set) else sorted(v)) for k, v in sc.ref.items()}, "pray4", len(sc.pray4))
    real = np.array([sc(a) for a in lib.avs]); print(" real avartanams mean f", real.mean(0).round(3))
    rng = np.random.default_rng(0); rnd = np.array([sc(lib.tokens_of(lib.sample(rng, 4))) for _ in range(300)]); print(" random cfg mean f", rnd.mean(0).round(3))
    t = time.time(); r = cmode(lib, sc, n_av=1, pop=30, gens=10); print(" cmode 10 gens", f"{time.time()-t:.1f}s", "front", len(r["front"]), r["Ff"].mean(0).round(3))
