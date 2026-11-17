"""Phase E2: learn gamaka (pitch residual around the written note) with a small 1-D conv net (torch/MPS).
 input per 10 ms frame: skeleton pitch (octaves), note-onset, position-in-note, note length, raga one-hot
 target: real pitch - skeleton pitch (cents), voiced frames only.   Held-out singer = prasanna.
 python gamaka_model.py   -> corpus_data/gamaka_model.pt, gamaka_eval.json"""
import json
import numpy as np, pandas as pd, torch, torch.nn as nn
from common import *
from phase1 import semi
from transcribe import DATA, HOP

RAG = ("mohanam", "kalyani", "abhogi"); W = 256; CLIP = 400.0
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def encode_events(raga, ev, n):
    """ev: [(pitch_cents, t0_frame, t1_frame)] contiguous. -> X [C, n]"""
    X = np.zeros((4 + len(RAG), n), np.float32); X[4 + RAG.index(raga)] = 1
    for k, (p, a, b) in enumerate(ev):
        a, b = max(0, a), min(n, b)
        if b <= a: continue
        L = b - a; X[0, a:b] = p / 1200.0; X[1, a:min(n, a + 3)] = 1.0
        X[2, a:b] = np.linspace(0, 1, L, endpoint=False); X[3, a:b] = min(L, 150) / 150.0
    return X


def build(uid_filter=None):
    A = pd.read_csv(DATA / "aligned_notes_dtw.csv"); data = []
    for uid, g in A.groupby("uid"):
        if uid_filter and not uid_filter(g.singer.iloc[0]): continue
        R = np.load(DATA / f"contours/{uid}.npy"); n = len(R); raga = g.raga.iloc[0]
        g = g.sort_values(["entry", "ncycle", "t0"]); t0 = (g.t0 / HOP).astype(int).to_numpy().copy(); t1 = (g.t1 / HOP).astype(int).to_numpy().copy()
        for i in range(len(g) - 1): t1[i] = max(t0[i] + 1, t0[i + 1])                         # contiguous: note lasts until next onset
        ev = [(float(semi(raga, s) * 100 + 1200 * o), a, b) for s, o, a, b in zip(g.swara, g.octave, t0, t1)]
        X = encode_events(raga, ev, n); S = X[0] * 1200.0; Y = (R - S).astype(np.float32)
        M = (~np.isnan(R)) & (np.abs(np.nan_to_num(Y)) < CLIP) & (X[3] > 0); Y = np.where(M, Y, 0)
        lo, hi = t0.min(), min(n, t1.max()); data.append((uid, g.singer.iloc[0], raga, X[:, lo:hi], Y[lo:hi], M[lo:hi].astype(np.float32)))
    return data


def windows(data, stride=64):
    Xs, Ys, Ms = [], [], []
    for _, _, _, X, Y, M in data:
        for s in range(0, X.shape[1] - W, stride):
            if M[s:s + W].mean() > .4: Xs.append(X[:, s:s + W]); Ys.append(Y[s:s + W]); Ms.append(M[s:s + W])
    return [torch.tensor(np.stack(a)) for a in (Xs, Ys, Ms)]


class Net(nn.Module):
    def __init__(self, cin=4 + len(RAG), h=64):
        super().__init__()
        L = [nn.Conv1d(cin, h, 9, padding=4), nn.GELU()]
        for d in (1, 2, 4, 8, 16, 1):
            L += [nn.Conv1d(h, h, 5, padding=2 * d, dilation=d), nn.GELU()]
        L += [nn.Conv1d(h, 1, 1)]; self.f = nn.Sequential(*L)
    def forward(self, x): return self.f(x)[:, 0] * 100.0          # cents


def lossf(p, y, m):
    l1 = (m * (p - y).abs()).sum() / m.sum().clamp(min=1)
    sp, sy = (m * p).std() if False else (p * m).abs().mean(), (y * m).abs().mean()      # keep amplitude close to the real one
    return l1 + 0.3 * (sp - sy).abs()


def evaluate(net, data):
    r = {"mae_flat": [], "mae_model": [], "std_real": [], "std_model": []}
    for _, _, _, X, Y, M in data:
        with torch.no_grad(): P = net(torch.tensor(X)[None].to(DEV))[0].cpu().numpy()
        m = M > 0; r["mae_flat"].append(np.abs(Y[m]).mean()); r["mae_model"].append(np.abs(P[m] - Y[m]).mean())
        r["std_real"].append(Y[m].std()); r["std_model"].append(P[m].std())
    return {k: float(np.mean(v)) for k, v in r.items()}


def main():
    torch.manual_seed(0); tr = build(lambda s: s != "prasanna"); te = build(lambda s: s == "prasanna")
    X, Y, M = windows(tr); Xt, Yt, Mt = windows(te); print("train windows", len(X), "test windows", len(Xt), "device", DEV)
    net = Net().to(DEV); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4); hist = []
    for ep in range(40):
        perm = torch.randperm(len(X)); net.train(); tot = 0
        for i in range(0, len(X), 64):
            ix = perm[i:i + 64]; x, y, m = X[ix].to(DEV), Y[ix].to(DEV), M[ix].to(DEV)
            x = x + 0.0                                                          # (no input noise: skeleton is exact)
            l = lossf(net(x), y, m); opt.zero_grad(); l.backward(); opt.step(); tot += l.item() * len(ix)
        net.eval()
        with torch.no_grad(): vl = lossf(net(Xt.to(DEV)), Yt.to(DEV), Mt.to(DEV)).item()
        hist.append((ep, tot / len(X), vl))
        if ep % 5 == 4: print(ep + 1, "train %.2f  heldout-singer %.2f" % (tot / len(X), vl), flush=True)
    torch.save(net.state_dict(), DATA / "gamaka_model.pt")
    res = dict(heldout_singer="prasanna", train=evaluate(net, tr), heldout=evaluate(net, te)); json.dump(res, open(DATA / "gamaka_eval.json", "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
