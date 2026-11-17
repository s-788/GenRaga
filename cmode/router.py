"""Phase 1.5: request -> parse -> resolve -> retrieve-or-report.
   python router.py "Ninnukori varnam in Mohanam, female voice" [--no-llm] [--all]"""
import sys, json
from catalog import *
from parse_request import parse
from resolver import resolve
from retrieve import retrieve, export_mp3

OUT = ROOT / "output"


def route(text, use_llm=True, max_files=2, export=True, generate=True):
    cat = Catalog()
    p = parse(text, cat, use_llm)
    R = resolve(p, cat)
    res = dict(request=text, parsed=p, resolved=R, decision=None, files=[], hits=[], message="")
    if R["status"] in ("unknown", "conflict") or not (R["raga"] or R["title_key"]):
        res["decision"] = "clarify"
        res["message"] = ("; ".join(R["notes"]) or "Could not identify a piece or raga in your request.")
        if R.get("suggest_raga"):
            res["message"] += f" Did you mean {cat.title_display[R['title_key']]} in {R['suggest_raga']}?"
        return res
    hits = retrieve(R, cat)
    res["hits"] = hits
    play = [h for h in hits if h["audio"]]
    exact = [h for h in play if h["level"] == "L1"]
    raga = R["raga"]
    can_gen = raga in GENERATOR_READY
    gen_note = ("C-MODE generator is available for this raga (Phase 2)" if can_gen else
                "a grammar exists but the generator is not enabled yet for this raga" if raga in GRAMMAR_RAGAS else
                "no notation/grammar for this raga: retrieval only, cannot generate")
    if exact:
        res["decision"] = "retrieved"
        if R["voice"]:
            want = "higher-pitched" if R["voice"] == "female" else "lower-pitched"
            match = [h for h in exact if h["range"] == want]
            if match:
                exact = match + [h for h in exact if h not in match][:0]
        if export:
            d = OUT / raga_slug(text)[:50]
            res["files"] = [str(export_mp3(h, d, i + 1)) for i, h in enumerate(exact[:max_files])]
        top = exact[0]
        res["message"] = (f"Found {len(exact)} existing recording(s) of {top['title']} ({raga}, {top['tala']}). "
                          f"Returning {min(len(exact), max_files)}." +
                          (f" Voice '{R['voice']}': ordered by tonic range (heuristic, not a gender label)." if R["voice"] else ""))
    else:
        res["decision"] = "not_found"
        if can_gen and generate and export:
            from v2.generate import generate as gen
            inst = "tone"
            d = OUT / raga_slug(text)[:50]; d.mkdir(parents=True, exist_ok=True); f = d / f"generated_{raga}_{inst}.mp3"
            g = gen(raga, f, inst)
            res.update(decision="generated", files=[str(f)], generated=g)
            res["message"] = (f"No existing recording matched, so a new {raga} piece was composed from the {raga} varnam's own cells "
                              f"(strict mode, {inst}, real gamaka curves). It rearranges real material, so it stays close to the original varnam; "
                              "it is not a recording of a singer and has not been checked by a musician.")
            return res
        near = [h for h in play if h["level"] == "L2"]
        res["message"] = (f"No recording of this exact request. {len(near)} other audio recording(s) in {raga}; "
                          f"{len([h for h in hits if h['level']=='L3'])} pitch-only archive item(s). Generation: {gen_note}.")
    return res


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    r = route(" ".join(a), use_llm="--no-llm" not in sys.argv)
    print(json.dumps({k: r[k] for k in ("parsed", "resolved", "decision", "message", "files")}, indent=1, ensure_ascii=False))
    for h in r["hits"][:6]:
        print(h["level"], h["source"], h["artist"], h["tonic_hz"], h["range"], h["audio"] and "audio" or "pitch-only")
