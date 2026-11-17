import sys, shutil
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from router import route, OUT
import parse_request, resolver
from catalog import Catalog

def run(t, **k): return route(t, use_llm=False, export=False, **k)

def test_1_exact_varnam():
    r = run("Ninnukori Mohanam"); assert r["decision"] == "retrieved" and len([h for h in r["hits"] if h["level"]=="L1"]) == 4

def test_2_conflict():
    r = run("Ninnukori in Kalyani"); assert r["decision"] == "clarify" and "mohanam" in r["message"].lower()

def test_3_misspelling():
    r = run("Ninukori mohana ragam"); assert r["decision"] == "retrieved" and r["resolved"]["raga"] == "mohanam"

def test_4_abhogi_varnam():
    r = run("abhogi varnam"); assert r["decision"] == "retrieved" and all(h["raga"] == "abhogi" for h in r["hits"] if h["level"] == "L1")

def test_5_archive_only():
    r = run("todi varnam")
    assert r["resolved"]["raga"] == "todi"
    assert any(h["level"] == "L3" and h["audio"] is None for h in r["hits"]) or r["decision"] in ("retrieved", "not_found")
    assert "cannot generate" in r["message"] or r["decision"] == "retrieved"

def test_6_unknown():
    r = run("play zzzqqq xylophone"); assert r["decision"] == "clarify" and r["files"] == []

def test_7_bad_llm_output(monkeypatch=None):
    # hallucinated field not present in text must be dropped
    parse_request.requests.post = lambda *a, **k: type("R", (), {"json": lambda s: {"response": '{"title":"Zebra Song","raga":"mohanam","tala":null,"form":null,"voice":null}'}})()
    d = parse_request.llm_parse("Mohanam please", "x"); assert d["title"] is None and d["raga"] == "mohanam"
    parse_request.requests.post = lambda *a, **k: type("R", (), {"json": lambda s: {"response": "not json"}})()
    assert parse_request.llm_parse("Mohanam", "x") is None

def test_8_voice_returns_matching_range():
    r = route("Ninnukori Mohanam female", use_llm=False, export=False)
    assert r["hits"][0]["range"] == "higher-pitched"

def test_9_mp3_export():
    r = route("Ninnukori Mohanam male", use_llm=False)
    import pathlib; assert r["files"] and all(pathlib.Path(f).suffix == ".mp3" and pathlib.Path(f).stat().st_size > 1e5 for f in r["files"])

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            try: f(); print("PASS", n)
            except Exception as e: print("FAIL", n, repr(e))
