# carnatic_unified

Unified manifest over three sources. Audio is referenced, not copied (paths relative to project root).

| source | what | n | audio | pitch | tala | notation |
|---|---|---|---|---|---|---|
| `varnam` | carnatic_varnam_1.1 (7 ragas, 5 singers, Adi tala) | 28 | yes | no (extract in Phase 3) | cycle annotations (.svl) | yes (swara YAML) |
| `cmd` | carnatic/ (CompMusic kritis etc.) | 197 | yes | yes (vocal for 47, else melody) | sama times (132), sections (110) | expert melodic phrases (100 recs) |
| `rrd` | archive (1)/ (40 ragas x 12 recordings) | 480 | no | yes (+confidence) | no | no |

Files
- `recordings.csv`  one row per recording (uid, source, raga slug, tala, form, tonic_hz, paths, flags)
- `ragas.csv`       per-raga counts per source + arohana/avarohana where notation exists (7 ragas)
- `contours/<uid>.npz`  `cents` (re. tonic, NaN = unvoiced) at native hop `hop_s`; rrd also has `conf`
- `pitch_class_hist_120.csv`  octave-folded 10-cent histogram, bin 0 centred on Sa (validated: Mohanam peaks at S R2 G2 P D2 in both cmd and rrd)
- `symbolic/varnam_notation.csv`  long-format swara grid (raga, section, avartanam, akshara, sub, swara, octave)
- `symbolic/varnam_structure.csv` performed section order/lengths per singer (duplicate YAML keys preserved)
- `symbolic/cmd_mphrases.csv`, `sections.csv`, `tala_annotations.csv`

Caveats (read before using)
1. `rrd` hop is NOT given in the source. `hop_s=512/44100` is an ASSUMPTION (`hop_assumed=True`); hop-independent features (histograms) are safe, time-based ones are not.
2. `rrd` voicing = confidence >= 0.5 (our choice). `cmd` unvoiced = pitch 0.
3. `cmd` `pitch_source=melody` may contain accompaniment; prefer `vocal` rows for gamaka modelling.
4. Raga slugs are diacritic-stripped; ragamalika kritis are labelled `ragamalika`. Same-name ragas across sources are merged (e.g. 6 of 7 varnam ragas also have cmd+rrd recordings; abhogi has no rrd).
5. Varnam notation is ONE composition per raga (12-20 avartanams); singers share it.
