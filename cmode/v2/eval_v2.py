"""Phase F: objective evaluation of generated pieces. python -m v2.eval_v2 -> v2_outputs/{audio,eval_summary.json,eval_pieces.csv,blind_test}"""
import json, subprocess, tempfile, random
import numpy as np, pandas as pd, parselmouth
from common import *
from v2 import clf
from v2.generate import compose
from v2.render import render, contour, bank
from v2.objectives import events
OUT = ROOT / "v2_outputs"; (OUT / "audio").mkdir(parents=True, exist_ok=True); RAGAS = ("mohanam", "kalyani", "abhogi")
CL = clf.load(); CLS = list(CL.classes_)


def track(path, tonic):
    wav = tempfile.mktemp(suffix=".wav"); subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ar", "32000", wav], check=True)
    p_obj = parselmouth.Sound(wav).to_pitch_ac(time_step=.01, pitch_floor=tonic * .6, pitch_ceiling=tonic * 3, very_accurate=True)
    f = p_obj.selected_array["frequency"]
    times = p_obj.xs()
    c = np.where(f > 0, 1200 * np.log2(np.where(f > 0, f, 1) / tonic), np.nan)
    return c, times


def hist(c):
    c = c[~np.isnan(c)]; h, _ = np.histogram(np.mod(c + 50, 1200) - 50, bins=120, range=(-50, 1150)); return h / max(h.sum(), 1)


def verbatim_run(lib, toks):
    cells = [tuple(toks[i:i + 4]) for i in range(0, len(toks), 4)]; best = 0
    for st in lib.streams.values():
        sc = [tuple(st[i:i + 4]) for i in range(0, len(st), 4)]; idx = {}
        for i in range(len(sc)): idx.setdefault(sc[i], []).append(i)
        for i in range(len(cells)):
            for j in idx.get(cells[i], []):
                k = 0
                while i + k < len(cells) and j + k < len(sc) and cells[i + k] == sc[j + k]: k += 1
                best = max(best, k)
    return best


def main():
    rows, pieces = [], []
    for raga in RAGAS:
        for seed in range(6):
            toks, rep, lib = compose(raga, seed=seed)
            tri = [tuple(toks[i:i + 12]) for i in range(0, len(toks) - 11, 4)]
            inst = "tone"
            f = OUT / f"audio/{raga}_{inst}_{seed}.mp3"; info = render(raga, toks, f, inst, seed=seed)
            tr, times = track(f, info["tonic"])
            c_target = np.interp(times, np.arange(len(info["cents"])) / 100.0, info["cents"])
            m = ~np.isnan(tr) & (times <= len(info["cents"]) / 100.0)
            e = np.abs(((tr[m] - c_target[m] + 600) % 1200) - 600)
            p_trk = CL.predict_proba(np.sqrt(hist(tr))[None])[0]; p_int = CL.predict_proba(np.sqrt(hist(info["cents"]))[None])[0]
            sel = [CLS.index(r) for r in RAGAS]
            dur = float(subprocess.run(["afinfo", str(f)], capture_output=True, text=True).stdout.split("estimated duration:")[1].split()[0])
            rows.append(dict(raga=raga, inst=inst, seed=seed, dur_s=dur, expected_s=len(toks) / 4 * .75, voiced=float(m.mean()),
                             pitch_err_med=float(np.median(e)), pitch_err_p90=float(np.percentile(e, 90)),
                             pitch_acc_50c=float((e < 50).mean()), curve_cov=info["curve_coverage"],
                             id40_tracked=CLS[int(p_trk.argmax())] == raga, id3_tracked=RAGAS[int(np.argmax(p_trk[sel]))] == raga,
                             id40_intended=CLS[int(p_int.argmax())] == raga, id3_intended=RAGAS[int(np.argmax(p_int[sel]))] == raga,
                             p_target_tracked=float(p_trk[CLS.index(raga)]),
                             new_3cell=float(np.mean([t not in lib.tri for t in tri])), longest_verbatim_cells=verbatim_run(lib, toks), cells=len(toks) // 4))
            print(raga, seed, flush=True)
    D = pd.DataFrame(rows); D.to_csv(OUT / "eval_pieces.csv", index=False)
    g = D.groupby(["raga", "inst"]).mean(numeric_only=True).round(3); print(g.to_string())
    S = dict(pieces=len(D), pitch_err_median_cents=float(D.pitch_err_med.mean()), pitch_err_p90=float(D.pitch_err_p90.mean()),
             pitch_acc_within_50c=float(D.pitch_acc_50c.mean()), voiced=float(D.voiced.mean()),
             length_ratio=float((D.dur_s / D.expected_s).mean()), id40_tracked=float(D.id40_tracked.mean()), id3_tracked=float(D.id3_tracked.mean()),
             id3_intended=float(D.id3_intended.mean()), new_3cell_windows=float(D.new_3cell.mean()), longest_verbatim_cells_mean=float(D.longest_verbatim_cells.mean()),
             longest_verbatim_cells_max=int(D.longest_verbatim_cells.max()), cells_per_piece=float(D.cells.mean()))
    json.dump(S, open(OUT / "eval_summary.json", "w"), indent=1); print(json.dumps(S, indent=1))
    # blind pack
    bt = OUT / "blind_test"; bt.mkdir(exist_ok=True)
    for p in bt.glob("*"): p.unlink()
    items = [(r, "tone", s) for r in RAGAS for s in (0, 1, 2, 3)]; random.Random(4).shuffle(items); key = []
    for n, (r, i, s) in enumerate(items, 1):
        subprocess.run(["cp", str(OUT / f"audio/{r}_{i}_{s}.mp3"), str(bt / f"clip_{n:02d}.mp3")]); key.append((f"clip_{n:02d}.mp3", r, i))
    pd.DataFrame(key, columns=["file", "raga", "instrument"]).to_csv(bt / "ANSWER_KEY_open_after_listening.csv", index=False)
    open(bt / "README.txt", "w").write("12 shuffled clips: 3 ragas (Mohanam, Kalyani, Abhogi) x 4 pieces (clean musical tone with gamaka curves and tanpura drone).\nWrite the raga you hear for each clip, and rate 1-5 how authentic it sounds, THEN open ANSWER_KEY_open_after_listening.csv.\n")


if __name__ == "__main__":
    main()
