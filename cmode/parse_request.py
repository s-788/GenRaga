"""Step 1: free text -> raw fields. Two interchangeable layers:
   (a) local LLM via Ollama (strict JSON, validated);  (b) deterministic n-gram matcher (always available).
The parser only EXTRACTS what the user wrote; the resolver decides what is true."""
import json, re
import requests
from rapidfuzz import fuzz, process
from catalog import *

OLLAMA = "http://localhost:11434"
PREFERRED = ["qwen2.5:3b-instruct", "qwen2.5:3b", "llama3.2:3b", "qwen2.5:7b-instruct", "llama3.1:8b"]
FEMALE = {"female", "woman", "women", "lady", "ladies", "girl", "feminine", "soprano"}
MALE = {"male", "man", "men", "gents", "boy", "masculine"}
SCHEMA_KEYS = ("title", "raga", "tala", "form", "voice")

PROMPT = """Extract fields from a Carnatic music request. Reply with ONLY a JSON object with keys:
title (composition name or null), raga (raga name or null), tala (tala name or null),
form (varnam/kriti/thillana/etc or null), voice ("female", "male" or null).
Copy words as the user wrote them. Use null when not stated. Never invent values.
Request: """


def installed_model():
    try:
        names = [m["name"] for m in requests.get(f"{OLLAMA}/api/tags", timeout=2).json()["models"]]
    except Exception:
        return None
    for p in PREFERRED:
        if p in names:
            return p
    return None  # smollm:135m etc. are too small to trust


def llm_parse(text, model):
    try:
        out = requests.post(f"{OLLAMA}/api/generate", timeout=120, json=dict(
            model=model, prompt=PROMPT + text, format="json", stream=False,
            options=dict(temperature=0))).json()["response"]
        d = json.loads(out)
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    clean = {k: (d.get(k) if isinstance(d.get(k), str) and d.get(k).strip() else None) for k in SCHEMA_KEYS}
    # anti-hallucination: any extracted string must literally appear (fuzzily) in the user's text
    nt = norm(text)
    for k, v in list(clean.items()):
        if v and k != "voice" and fuzz.partial_ratio(norm(v), nt) < 80:
            clean[k] = None
    return clean


def _ngrams(words, nmax=4):
    for n in range(nmax, 0, -1):
        for i in range(len(words) - n + 1):
            yield i, n, " ".join(words[i:i + n])


def rule_parse(text, cat: Catalog):
    words = norm(text).split()
    out = dict.fromkeys(SCHEMA_KEYS)
    ws = set(words)
    out["voice"] = "female" if ws & FEMALE else "male" if ws & MALE else None
    used = set()
    for i, w in enumerate(words):
        if w in FORM_WORDS or w in ("tala", "talam", "thalam"):
            used.add(i)
            if w in FORM_WORDS:
                out["form"] = w
    for a in sorted(TALA_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{a}\b", " ".join(words)) and not (a == "adi" and "adi" not in words):
            out["tala"] = a; break
    # raga: word(s) right before 'raga/ragam' first, else best fuzzy token against raga vocabulary
    for i, w in enumerate(words):
        if w in RAGA_NOISE - {"in", "the", "of"} and i > 0:
            out["raga"] = words[i - 1]; used.add(i - 1)
    # title: longest n-gram with strong fuzzy match to a catalog title
    best = (0, None, None)
    for i, n, g in _ngrams(words):
        if len(g) < 5 or any(j in used for j in range(i, i + n)):
            continue
        m = process.extractOne(g, cat.titles, scorer=fuzz.ratio)
        if m and m[1] >= 86 and (m[1] * (1 + 0.15 * n)) > best[0]:
            best = (m[1] * (1 + 0.15 * n), g, (i, n))
    if best[1]:
        out["title"] = best[1]
        i, n = best[2]; used |= set(range(i, i + n))
    if not out["raga"]:
        for i, n, g in _ngrams(words, 2):
            if any(j in used for j in range(i, i + n)) or len(g) < 4:
                continue
            m = process.extractOne(g, cat.ragas, scorer=fuzz.ratio)
            if m and m[1] >= 88:
                out["raga"] = g; break
    return out


def parse(text, cat: Catalog, use_llm=True):
    model = installed_model() if use_llm else None
    if model:
        d = llm_parse(text, model)
        if d:
            d["_parser"] = f"llm:{model}"
            r = rule_parse(text, cat)          # fill gaps the LLM left empty, from the same text
            for k in SCHEMA_KEYS:
                d[k] = d[k] or r[k]
            return d
    d = rule_parse(text, cat); d["_parser"] = "rules"
    return d
