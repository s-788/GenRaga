"""Shared helpers for the C-MODE project (paths, raga-name normalisation, notation parsing)."""
import re, unicodedata
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
VARNAM = ROOT / "carnatic_varnam_1.1"
CMD = ROOT / "carnatic"
RRD = ROOT / "archive (1)"
UNIFIED = ROOT / "carnatic_unified"
PHASE1 = ROOT / "phase1_outputs"


def raga_slug(name: str) -> str:
    """'Mōhanaṁ' -> 'mohanam', 'Śrī' -> 'sri', 'Kāṁbhōji' -> 'kambhoji'."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("ḷ", "l").replace("ṛ", "r")
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s


# ---------------------------------------------------------------- notation
SWARA_LETTERS = "SRGMPDN"


def load_ordered_mapping(path):
    """Load a YAML mapping keeping DUPLICATE keys (needed for per-singer 'structure')."""
    node = yaml.compose(open(path, encoding="utf-8"))
    out = []
    for k, v in node.value:
        out.append((k.value, yaml.safe_load(yaml.serialize(v)) if not hasattr(v, "value") or v.id != "mapping"
                    else [(kk.value, _scalar(vv)) for kk, vv in v.value]))
    return out


def _scalar(n):
    return yaml.safe_load(n.value) if n.value != "" else None


def parse_token(tok: str):
    """'S^' -> ('S', +1); 'N_' -> ('N', -1); ',' -> (',', 0)."""
    tok = str(tok).strip()
    if tok == ",":
        return ",", 0
    octv = 1 if tok.endswith("^") else -1 if tok.endswith("_") else 0
    return tok.rstrip("^_"), octv


def variant_map(moorchanas: dict) -> dict:
    """letter -> variant (R -> R2) using the raga's arohana/avarohana."""
    m = {}
    for note in moorchanas["arohana"] + moorchanas["avarohana"]:
        base, _ = parse_token(note)
        m[base[0]] = base  # 'R2' keyed by 'R'
    return m
