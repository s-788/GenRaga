"""Step 2: validate parsed fields against OUR catalog. Never trusts the parser."""
from rapidfuzz import fuzz, process
from catalog import *


def _best(q, vocab, thr):
    m = process.extractOne(norm(q), vocab, scorer=fuzz.ratio)
    return (m[0], m[1]) if m and m[1] >= thr else (None, 0)


def resolve(p, cat: Catalog):
    R = dict(status="unknown", notes=[], title_key=None, raga=None, tala=None, form=None, voice=p.get("voice"))
    # raga
    if p.get("raga"):
        q = " ".join(w for w in norm(p["raga"]).split() if w not in RAGA_NOISE)
        raga, sc = _best(q, cat.ragas, 84)
        if raga:
            R["raga"] = raga
            if norm(p["raga"]) != norm(cat.raga_display.get(raga, raga)) and raga != q:
                R["notes"].append(f"raga '{p['raga']}' read as {cat.raga_display.get(raga, raga)}")
        else:
            R["notes"].append(f"raga '{p['raga']}' not in dataset")
    # title (a bare form word like 'thillana' is a form, not a title)
    if p.get("title") and norm(p["title"]) in FORM_WORDS:
        p = {**p, "form": p.get("form") or p["title"], "title": None}
    if p.get("title"):
        key, sc = _best(p["title"], cat.titles, 84)
        if key:
            R["title_key"] = key
            if key != norm(p["title"]):
                R["notes"].append(f"title '{p['title']}' read as {cat.title_display[key]}")
        else:
            R["notes"].append(f"title '{p['title']}' not in dataset")
    # tala / form
    if p.get("tala"):
        R["tala"] = TALA_ALIASES.get(norm(p["tala"]))
        if not R["tala"]:
            R["notes"].append(f"tala '{p['tala']}' not recognised")
    if p.get("form"):
        R["form"] = FORM_WORDS.get(norm(p["form"]).split()[0])
    # conflicts: the title's known raga/tala vs what the user asked
    status = "confirmed" if (R["title_key"] or R["raga"]) else "unknown"
    if R["title_key"]:
        info = cat.title_info(R["title_key"])
        if not R["raga"] and len(info["ragas"]) == 1:
            R["raga"] = info["ragas"][0]; R["notes"].append(f"raga inferred from title: {R['raga']}")
        if R["raga"] and info["ragas"] and R["raga"] not in info["ragas"]:
            status = "conflict"
            R["notes"].append(f"'{cat.title_display[R['title_key']]}' is in {', '.join(info['ragas'])}, not {R['raga']}")
            R["suggest_raga"] = info["ragas"][0]
        if R["tala"] and info["talas"] and R["tala"] not in info["talas"]:
            status = "conflict"; R["notes"].append(f"that piece is in {', '.join(info['talas'])}, not {R['tala']}")
        R["known_tala"] = info["talas"]
    if status == "confirmed" and R["notes"]:
        status = "corrected"
    R["status"] = status
    return R
