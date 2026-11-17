"""Knowledge tables built from carnatic_unified/: what titles/ragas/talas exist and which recordings back them."""
import re
import pandas as pd
from common import *

GRAMMAR_RAGAS = {p.stem for p in (PHASE1 / "grammars").glob("*.json")}
GENERATOR_READY = {"mohanam", "kalyani", "abhogi", "sri", "saveri", "sahana", "begada"}      # All 7 notated & gamaka-aligned ragas
TALA_ALIASES = {"adi": "Ādi", "aadi": "Ādi", "rupaka": "Rūpaka", "rupakam": "Rūpaka", "atta": "Aṭṭa",
                "misra chapu": "Miśra chāpu", "khanda chapu": "Khaṇḍa chāpu", "triputa": "Ādi"}
FORM_WORDS = {"varnam": "Varnam", "varnams": "Varnam", "kriti": "Kriti", "krithi": "Kriti", "kirtana": "Kriti",
              "thillana": "Thillana", "tillana": "Thillana", "mangalam": "Mangalam", "shloka": "Shloka", "javali": "Javali"}
RAGA_NOISE = {"raga", "ragam", "raagam", "raaga", "ragaa", "in", "the", "of"}


def norm(s: str) -> str:
    return raga_slug(s).replace("_", " ")


class Catalog:
    def __init__(self):
        r = pd.read_csv(UNIFIED / "recordings.csv", keep_default_na=False)
        r["title_key"] = r.title.map(norm)
        self.rec = r
        self.ragas = sorted(set(r[r.raga != ""].raga) - {"ragamalika"})
        self.raga_display = (r[r.raga != ""].groupby("raga").raga_raw.first().to_dict())
        t = r[(r.source != "rrd") & (r.title_key != "")]
        self.titles = sorted(set(t.title_key))
        self.title_display = t.groupby("title_key").title.first().to_dict()

    def same_composition(self, key):
        """All title keys naming the same piece (e.g. 'sarasuda ninne' == 'sarasuda ninne kori')."""
        return {k for k in self.titles if k == key or k.startswith(key + " ") or key.startswith(k + " ")}

    def title_info(self, key):
        ks = self.same_composition(key)
        d = self.rec[self.rec.title_key.isin(ks) & (self.rec.source != "rrd")]
        return dict(ragas=sorted(set(d.raga) - {""}), talas=sorted(set(d.tala) - {""}),
                    forms=sorted(set(d.form) - {""}), n=len(d))
