"""Phase A3: mine phrases that RECUR across different compositions of a raga.
 1 transcribe all cmd+rrd recordings of the raga (frame-level), plus 'null' recordings of OTHER ragas (same grid)
 2 candidates: varnam units, expert phrases, frequent n-grams
 3 support = #distinct compositions with an approximate match (rapidfuzz partial_ratio); lift = target rate / null rate
 4 keep: support>=3 compositions and lift>=3.   Writes corpus_data/mined_<raga>.csv + corpus_data/mining_report.json"""
import json, random, collections
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from common import *
from phase1 import semi
from transcribe import transcribe_frames, note_grid, DATA

RAGAS = ("mohanam", "kalyani", "abhogi")
N_SUR = 6
LET = {"S": "s", "R": "r", "G": "g", "M": "m", "P": "p", "D": "d", "N": "n"}
nt = pd.read_csv(UNIFIED / "symbolic/varnam_notation.csv", keep_default_na=False)
SC = {r: sorted({s for s in g.swara if s != ","}, key=lambda s: semi(r, s)) for r, g in nt.groupby("raga")}
REC = pd.read_csv(UNIFIED / "recordings.csv", keep_default_na=False)
EXP = pd.read_csv(UNIFIED / "symbolic/cmd_mphrases.csv")


def collapse(s):
    return "".join(c for i, c in enumerate(s) if i == 0 or c != s[i - 1])


def seq_of(uid, raga):
    z = np.load(UNIFIED / f"contours/{uid}.npz"); c, hop = z["cents"], float(z["hop_s"])
    tr = transcribe_frames(c, note_grid(raga, SC[raga]), hop)
    return collapse("".join(LET[n.rstrip("^_")[0]] for n, a, b in tr))


def tol(L):  # minimum partial_ratio for a pattern of length L to count as present
    return 100 if L <= 4 else 83 if L <= 7 else 75


def present(p, s):
    return fuzz.partial_ratio(p, s, score_cutoff=tol(len(p))) >= tol(len(p)) if len(s) >= len(p) - 1 else False


def varnam_units(raga):
    out = {}
    for sec, cyc in json.load(open(PHASE1 / "corpus_hierarchy.json"))[raga].items():
        for a in cyc:
            for L in (4, 8):
                for i in range(0, 32, L):
                    u = a[i:i + L]; p = collapse("".join(LET[t.rstrip("^_")[0]] for t in u if t != ","))
                    if len(p) >= 3: out.setdefault(p, set()).add("varnam")
    return out


def run(raga, n_null=80, seed=0):
    rng = random.Random(seed)
    own = REC[(REC.raga == raga) & REC.source.isin(["cmd", "rrd"])]
    oth = REC[(REC.source == "rrd") & (REC.raga != raga) & (REC.raga != "")]
    oth = oth.iloc[rng.sample(range(len(oth)), min(n_null, len(oth)))]
    S_own = {r.uid: (r.title, seq_of(r.uid, raga)) for r in own.itertuples()}
    S_null = {r.uid: (r.title, seq_of(r.uid, raga)) for r in oth.itertuples()}
    comps_own = {t for t, s in S_own.values()}; comps_null = {t for t, s in S_null.values()}
    cand = {}
    for p, src in varnam_units(raga).items(): cand.setdefault(p, set()).update(src)
    for ph in EXP[EXP.raga == raga].phrase:
        p = collapse(ph)
        if len(p) >= 3: cand.setdefault(p, set()).add("expert")
    cnt = collections.defaultdict(set)                                    # frequent exact n-grams across compositions
    for uid, (t, s) in S_own.items():
        for L in (4, 5, 6):
            for i in range(len(s) - L + 1): cnt[s[i:i + L]].add(t)
    for p, ts in cnt.items():
        if len(ts) >= 3: cand.setdefault(p, set()).add("ngram")
    # surrogate null: per recording, sample sequences from ITS OWN first-order note transitions (same note usage, no phrase memory)
    nprng = np.random.default_rng(seed); SUR = []
    for uid, (t, s) in S_own.items():
        a = sorted(set(s)); ix = {c: i for i, c in enumerate(a)}; M = np.ones((len(a), len(a))) * 0.01
        for x, y in zip(s, s[1:]): M[ix[x], ix[y]] += 1
        M /= M.sum(1, keepdims=True)
        for _ in range(N_SUR):
            cur = nprng.integers(len(a)); out = [a[cur]]
            for _ in range(len(s) - 1):
                cur = nprng.choice(len(a), p=M[cur]); out.append(a[cur])
            SUR.append((t, "".join(out)))
    rows = []
    for p, src in cand.items():
        so = {t for uid, (t, s) in S_own.items() if present(p, s)}
        sn = {t for uid, (t, s) in S_null.items() if present(p, s)}
        sur = np.mean([len({t for t, s in SUR[k::N_SUR] if present(p, s)}) for k in range(N_SUR)])
        ro, rn = len(so) / max(len(comps_own), 1), len(sn) / max(len(comps_null), 1)
        rows.append(dict(raga=raga, phrase=p, length=len(p), sources="+".join(sorted(src)), support=len(so), null_support=len(sn), surrogate_support=round(float(sur), 2), lift_surr=round(len(so) / (sur + 0.5), 2),
                         rate=round(ro, 3), null_rate=round(rn, 3), lift=round(ro / (rn + 1 / (2 * max(len(comps_null), 1))), 2)))
    D = pd.DataFrame(rows); D["keep"] = (D.support >= 3) & (D.lift >= 3) & (D.lift_surr >= 2)
    D.sort_values(["keep", "lift", "support"], ascending=False).to_csv(DATA / f"mined_{raga}.csv", index=False)
    info = dict(own_recordings=len(S_own), own_compositions=len(comps_own), null_recordings=len(S_null), candidates=len(D), kept=int(D.keep.sum()),
                expert_mean_support=float(D[D.sources.str.contains("expert")].support.mean()) if D.sources.str.contains("expert").any() else None,
                expert_mean_surrogate=float(D[D.sources.str.contains("expert")].surrogate_support.mean()) if D.sources.str.contains("expert").any() else None,
                kept_by_source={s: int(D[D.keep & D.sources.str.contains(s)].shape[0]) for s in ("varnam", "expert", "ngram")},
                cand_by_source={s: int(D[D.sources.str.contains(s)].shape[0]) for s in ("varnam", "expert", "ngram")},
                mean_len_kept=float(D[D.keep].length.mean()) if D.keep.any() else None,
                note_rate_own=float(np.mean([len(s) for _, s in S_own.values()])))
    print(raga, json.dumps(info))
    return info


if __name__ == "__main__":
    rep = {r: run(r) for r in RAGAS}
    json.dump(rep, open(DATA / "mining_report.json", "w"), indent=1)
