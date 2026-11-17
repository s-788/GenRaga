"""Raga classifier on recorded pitch-class histograms (40 ragas, 570 recordings, sqrt features, logistic regression)."""
import pickle
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from common import *
from phase1 import semi
CACHE = ROOT / "corpus_data/raga_clf.pkl"
EDGES = np.arange(-50, 1151, 10)


class CalibratedRagaClassifier:
    def __init__(self, model, prior_log, prior_weight=0.7):
        self.model = model
        self.classes_ = model.classes_
        self.prior_log = prior_log
        self.prior_weight = prior_weight
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_

    def predict_proba(self, X):
        logits = X @ self.model.coef_.T + self.model.intercept_ - self.prior_weight * self.prior_log
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def predict(self, X):
        p = self.predict_proba(X)
        return self.classes_[np.argmax(p, axis=1)]


def train():
    r = pd.read_csv(UNIFIED / "recordings.csv", keep_default_na=False); H = pd.read_csv(UNIFIED / "pitch_class_hist_120.csv", index_col=0)
    r = r[r.uid.isin(H.index) & (r.raga != "") & (r.raga != "ragamalika") & (r.source != "varnam")]
    X = [np.sqrt(H.loc[u].values) for u in r.uid]; y = list(r.raga.values)
    vr = pd.read_csv(UNIFIED / "recordings.csv", keep_default_na=False); vr = vr[vr.source == "varnam"]      # solo-voice varnam recordings add Abhogi
    for u, rg in zip(vr.uid, vr.raga):
        f = ROOT / f"corpus_data/contours/{u}.npy"
        if f.exists():
            c = np.load(f); c = c[~np.isnan(c)]; h, _ = np.histogram(np.mod(c + 50, 1200) - 50, bins=120, range=(-50, 1150)); X.append(np.sqrt(h / h.sum())); y.append(rg)
    y = np.array(y); vc = pd.Series(y).value_counts(); keep = np.isin(y, vc[vc >= 7].index)
    m = LogisticRegression(C=10, max_iter=3000).fit(np.array(X)[keep], y[keep])
    classes = list(m.classes_)
    counts = pd.Series(y[keep]).value_counts().loc[classes].values
    prior_log = np.log(counts / len(y[keep]))
    clf_obj = CalibratedRagaClassifier(m, prior_log, prior_weight=0.7)
    pickle.dump(clf_obj, open(CACHE, "wb")); return clf_obj


def load(): return pickle.load(open(CACHE, "rb")) if CACHE.exists() else train()


def sym_hist(raga, events, sigma=25.0):
    """duration-weighted symbolic note usage -> 120-bin histogram (gaussian smear mimics recorded pitch)."""
    ctr = EDGES[:-1] + 5.0; h = np.zeros(120)
    for tok, d in events:
        c = (semi(raga, tok.rstrip("^_")) * 100) % 1200
        dd = np.minimum(np.abs(ctr - c), 1200 - np.abs(ctr - c)); dd = np.minimum(dd, np.abs(ctr - c - 1200)); dd = np.minimum(dd, np.abs(ctr - c + 1200))
        h += d * np.exp(-0.5 * (dd / sigma) ** 2)
    return h / max(h.sum(), 1e-9)
