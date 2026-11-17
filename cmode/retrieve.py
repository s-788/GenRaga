"""Step 3: ranked retrieval. L1 = exact piece (title [+raga]); L2 = same raga, other piece (audio);
L3 = same raga, archive pitch-only. Voice preference only re-orders (tonic-range heuristic), never filters."""
import shutil
from catalog import *

HIGH, LOW = 190.0, 160.0   # Hz: tonic >= HIGH -> 'higher-pitched singer', <= LOW -> 'lower-pitched'


def _order(df, voice):
    if voice == "female":
        return df.sort_values("tonic_hz", ascending=False)
    if voice == "male":
        return df.sort_values("tonic_hz", ascending=True)
    return df.sort_values(["source", "uid"])


def _tag(t):
    return "higher-pitched" if t >= HIGH else "lower-pitched" if t <= LOW else "mid-range"


def retrieve(R, cat: Catalog):
    rec, voice = cat.rec, R["voice"]
    audio = rec[rec.audio_path != ""]
    hits = []
    if R["title_key"]:
        ks = cat.same_composition(R["title_key"])
        d = audio[audio.title_key.isin(ks)]
        if R["raga"]:
            d = d[d.raga == R["raga"]]
        hits += [("L1", d)]
    elif R["raga"]:
        d = audio[(audio.raga == R["raga"])]
        if R["form"]:
            f = d[d.form.str.contains(R["form"], case=False) | ((R["form"] == "Varnam") & (d.source == "varnam"))]
            hits += [("L1", f)]
            d = d.drop(f.index)
        else:
            hits += [("L1", d[d.source == "varnam"])] ; d = d[d.source != "varnam"]
        hits += [("L2", d)]
    if R["raga"]:
        used = set().union(*[set(h.uid) for _, h in hits]) if hits else set()
        hits.append(("L2", audio[(audio.raga == R["raga"]) & ~audio.uid.isin(used)]))
        hits.append(("L3", rec[(rec.raga == R["raga"]) & (rec.audio_path == "")]))
    out = []
    for lvl, d in hits:
        for _, x in _order(d, voice).iterrows():
            out.append(dict(level=lvl, uid=x.uid, title=x.title, artist=x.artist, raga=x.raga, tala=x.tala,
                            tonic_hz=float(x.tonic_hz), range=_tag(x.tonic_hz), source=x.source,
                            audio=(str(ROOT / x.audio_path) if x.audio_path else None)))
    seen, uniq = set(), []
    for h in out:
        if h["uid"] not in seen:
            seen.add(h["uid"]); uniq.append(h)
    return uniq


def export_mp3(hit, outdir, rank):
    """Copy the existing mp3 (no re-encoding, so quality is untouched)."""
    outdir.mkdir(parents=True, exist_ok=True)
    name = f"{rank:02d}_{hit['level']}_{raga_slug(hit['title'])}_{raga_slug(hit['artist'])}.mp3"
    dst = outdir / name
    shutil.copy2(hit["audio"], dst)
    return dst
