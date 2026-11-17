"""Build the unified Carnatic dataset (carnatic_unified/) from
   (1) carnatic_varnam_1.1 (CompMusic varnams: audio + notation + tala)
   (2) carnatic/           (CompMusic Carnatic corpus subset: audio, pitch, tala, sections, melodic phrases)
   (3) archive (1)/        (40-raga x 12 recordings pitch/confidence/tonic corpus, no audio)
Audio is NOT copied (5 GB); manifest holds paths relative to the project root."""
import json, re, subprocess, xml.etree.ElementTree as ET
import numpy as np, pandas as pd, yaml
from common import *

RRD_HOP_ASSUMED = 512 / 44100  # ASSUMPTION: archive(1) ships no hop metadata -> flagged in manifest
CMD_HOP = 196 / 44100
HIST_BINS = 120  # 10-cent bins over one octave, range -50..1150 cents (semitone-centred)
rel = lambda p: str(Path(p).relative_to(ROOT))


def cents(f, tonic):
    f = np.asarray(f, float)
    ok = f > 0
    c = np.full(f.shape, np.nan, np.float32)
    c[ok] = 1200 * np.log2(f[ok] / tonic)
    return c


def pc_hist(c):
    c = c[~np.isnan(c)]
    h, _ = np.histogram(np.mod(c + 50, 1200) - 50, bins=HIST_BINS, range=(-50, 1150))  # bin 0 centred on Sa
    return (h / max(h.sum(), 1)).astype(np.float32)


def afinfo_dur(mp3):
    try:
        o = subprocess.run(["afinfo", str(mp3)], capture_output=True, text=True).stdout
        return float(re.search(r"estimated duration: ([\d.]+)", o).group(1))
    except Exception:
        return np.nan


def main():
    (UNIFIED / "contours").mkdir(parents=True, exist_ok=True)
    (UNIFIED / "symbolic").mkdir(exist_ok=True)
    rows, hists, tala_rows, sec_rows, phr_rows, notes_rows, struct_rows = [], [], [], [], [], [], []

    # ------------------------------------------------ (1) varnam
    tonics = yaml.safe_load(open(VARNAM / "Notations_Annotations/annotations/tonics.yaml"))
    NOT = VARNAM / "Notations_Annotations/notations"
    for mp3 in sorted((VARNAM / "Audio").glob("*.mp3")):
        m = re.search(r"by-(.+)-in-(.+)-raaga", mp3.name)
        singer, raga = m.group(1), m.group(2)
        uid = f"varnam_{raga}_{singer}"
        svl = VARNAM / f"Notations_Annotations/annotations/taalas/{raga}/{singer}.svl"
        n_t = 0
        if svl.exists():
            root = ET.parse(svl).getroot()
            sr = float(root.find(".//model").get("sampleRate"))
            for p in root.iter("point"):
                lab = p.get("label")
                tala_rows.append((uid, "manual_cycle_start" if lab.endswith(".1") else "auto_akshara_div",
                                  int(p.get("frame")) / sr, lab)); n_t += 1
        sfile = NOT / raga / f"{singer}.yaml"
        rows.append(dict(uid=uid, source="varnam", title=yaml.safe_load(open(NOT / f"{raga}.yaml"))["title"],
                         artist=singer, raga_raw=raga, raga=raga_slug(raga), tala="Ādi", form="Varnam",
                         duration_s=afinfo_dur(mp3), tonic_hz=tonics.get(singer), audio_path=rel(mp3),
                         pitch_path="", contour_path="", hop_s=np.nan, hop_assumed=False, pitch_source="",
                         has_tala=n_t > 0, has_sections=sfile.exists(), has_mphrases=False, has_notation=True,
                         is_ragamalika=False))
        if sfile.exists():
            for i, (sec, cnt) in enumerate(load_ordered_mapping(sfile)[2][1]):
                struct_rows.append((uid, raga_slug(raga), singer, i, sec, cnt))

    # symbolic notation: raga-level yaml (base notation, avartanam x 8 aksharas x 4 subdivisions)
    scale_rows = []
    for y in sorted(NOT.glob("*.yaml")):
        d = yaml.safe_load(open(y, encoding="utf-8"))
        raga = raga_slug(y.stem)
        vm = variant_map(d["moorchanas"])
        scale_rows.append(dict(raga=raga, arohana=" ".join(d["moorchanas"]["arohana"]),
                               avarohana=" ".join(d["moorchanas"]["avarohana"]), varnam_title=d["title"]))
        for sec, avs in d.items():
            if sec in ("title", "mbid", "moorchanas"):
                continue
            for a, av in enumerate(avs):
                for k, cell in enumerate(av):
                    for s, tok in enumerate(cell):
                        base, octv = parse_token(tok)
                        sw = base if base == "," else vm.get(base[0], base)
                        notes_rows.append((raga, sec, a, k, s, str(tok), sw, octv))
    pd.DataFrame(notes_rows, columns="raga section avartanam akshara sub raw swara octave".split()
                 ).to_csv(UNIFIED / "symbolic/varnam_notation.csv", index=False)
    pd.DataFrame(struct_rows, columns="uid raga singer order section n_avartanams".split()
                 ).to_csv(UNIFIED / "symbolic/varnam_structure.csv", index=False)
    scales = pd.DataFrame(scale_rows)

    # ------------------------------------------------ (2) CMD
    for js in sorted(CMD.glob("*/*.json")):
        j = json.load(open(js, encoding="utf-8"))
        base = str(js)[:-5]
        cid = js.parent.name
        uid = f"cmd_{cid}"
        ragas = [r["name"] for r in j.get("raaga", [])]
        lead = ([a["artist"]["name"] for a in j.get("artists", []) if a.get("lead")]
                or [a["name"] for a in j.get("album_artists", [])] or [""])[0]
        tonic = float(open(base + ".ctonic.txt").read().split()[0])
        pv = Path(base + ".pitch-vocal.txt")
        pf, src = (pv, "vocal") if pv.exists() else (Path(base + ".pitch.txt"), "melody")
        arr = pd.read_csv(pf, sep=r"\s+", header=None).values
        c = cents(arr[:, 1], tonic)
        np.savez_compressed(UNIFIED / f"contours/{uid}.npz", cents=c, hop_s=np.float32(arr[1, 0] - arr[0, 0]),
                            tonic_hz=np.float32(tonic))
        hists.append((uid, pc_hist(c)))
        for kind, suf in (("sama", "sama-manual.txt"),):
            p = Path(f"{base}.{suf}")
            if p.exists():
                sama = [float(x) for x in open(p).read().split()]
                tala_rows += [(uid, "sama", t, "") for t in sama]
        sp = Path(base + ".sections-manual-p.txt")
        if sp.exists() and sp.stat().st_size:
            for line in open(sp, encoding="utf-8"):
                a = line.rstrip("\n").split("\t")
                if len(a) >= 4:
                    sec_rows.append((uid, float(a[0]), float(a[2]), a[3]))
        mp = Path(base + ".mphrases-manual.txt")
        has_ph = mp.exists() and mp.stat().st_size > 0
        if has_ph:
            for line in open(mp, encoding="utf-8"):
                a = line.split()
                if len(a) >= 4:
                    phr_rows.append((uid, raga_slug(ragas[0]) if ragas else "", float(a[0]), float(a[1]),
                                     float(a[2]), a[3], len(a[3])))
        rows.append(dict(uid=uid, source="cmd", title=j.get("title", ""), artist=lead,
                         raga_raw="|".join(ragas), raga=("ragamalika" if len(ragas) > 1 else raga_slug(ragas[0])) if ragas else "",
                         tala=(j.get("taala") or [{}])[0].get("name", ""), form=(j.get("form") or [{}])[0].get("name", ""),
                         duration_s=j["length"] / 1000, tonic_hz=tonic, audio_path=rel(base + ".mp3"),
                         pitch_path=rel(pf), contour_path=rel(UNIFIED / f"contours/{uid}.npz"), hop_s=float(arr[1, 0] - arr[0, 0]),
                         hop_assumed=False, pitch_source=src, has_tala=Path(base + ".sama-manual.txt").exists() and Path(base + ".sama-manual.txt").stat().st_size > 0,
                         has_sections=sp.exists() and sp.stat().st_size > 0, has_mphrases=has_ph, has_notation=False,
                         is_ragamalika=len(ragas) > 1))
    print("cmd done", flush=True)

    # ------------------------------------------------ (3) RRD archive(1)
    for rdir in sorted(p for p in RRD.iterdir() if p.is_dir()):
        for rec in sorted(p for p in rdir.iterdir() if p.is_dir()):
            stem = rec.name
            artist, album, title = (stem.split("---") + ["", "", ""])[:3]
            uid = f"rrd_{raga_slug(rdir.name)}_{len([r for r in rows if r['source']=='rrd' and r['raga']==raga_slug(rdir.name)]):02d}"
            g1 = lambda suf: next(rec.glob(f"*---{suf}.txt"))
            tonic = float(open(g1("tonic")).read().split()[0])
            f = np.loadtxt(g1("pitches"))
            cf = np.loadtxt(g1("confidences"))
            c = cents(f, tonic)
            c[cf < 0.5] = np.nan  # voicing decision: confidence >= 0.5 (our choice, documented)
            np.savez_compressed(UNIFIED / f"contours/{uid}.npz", cents=c, conf=cf.astype(np.float16),
                                hop_s=np.float32(RRD_HOP_ASSUMED), tonic_hz=np.float32(tonic))
            hists.append((uid, pc_hist(c)))
            rows.append(dict(uid=uid, source="rrd", title=title.replace("_", " "), artist=artist.replace("__", ". ").replace("_", " "),
                             raga_raw=rdir.name, raga=raga_slug(rdir.name), tala="", form="", duration_s=len(f) * RRD_HOP_ASSUMED,
                             tonic_hz=tonic, audio_path="", pitch_path=rel(g1("pitches")),
                             contour_path=rel(UNIFIED / f"contours/{uid}.npz"), hop_s=RRD_HOP_ASSUMED, hop_assumed=True,
                             pitch_source="melody", has_tala=False, has_sections=False, has_mphrases=False,
                             has_notation=False, is_ragamalika=False))
    print("rrd done", flush=True)

    rec = pd.DataFrame(rows)
    rec.to_csv(UNIFIED / "recordings.csv", index=False)
    pd.DataFrame(tala_rows, columns="uid kind t_s label".split()).to_csv(UNIFIED / "symbolic/tala_annotations.csv", index=False)
    pd.DataFrame(sec_rows, columns="uid start_s dur_s section".split()).to_csv(UNIFIED / "symbolic/sections.csv", index=False)
    pd.DataFrame(phr_rows, columns="uid raga start_s flag dur_s phrase n_notes".split()).to_csv(UNIFIED / "symbolic/cmd_mphrases.csv", index=False)
    H = pd.DataFrame([h for _, h in hists], index=[u for u, _ in hists], columns=[f"b{i}" for i in range(HIST_BINS)])
    H.index.name = "uid"; H.to_csv(UNIFIED / "pitch_class_hist_120.csv", float_format="%.6f")

    g = rec[~rec.is_ragamalika & (rec.raga != "")].groupby("raga")
    ragas = g.agg(n_recordings=("uid", "count"), n_varnam=("source", lambda s: (s == "varnam").sum()),
                  n_cmd=("source", lambda s: (s == "cmd").sum()), n_rrd=("source", lambda s: (s == "rrd").sum()),
                  display=("raga_raw", "first")).reset_index()
    ragas = ragas.merge(scales, on="raga", how="left")
    ragas.sort_values("n_recordings", ascending=False).to_csv(UNIFIED / "ragas.csv", index=False)
    print(rec.groupby("source").size(), "\nragas:", len(ragas))


if __name__ == "__main__":
    main()
