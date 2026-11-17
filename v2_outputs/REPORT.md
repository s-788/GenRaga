# Improvement round: report (v2)

Code: `cmode/v2/*` (cell library, objectives, C-MODE v2, rendering, generator, evaluation), `cmode/transcribe.py`, `cmode/mine.py`,
`cmode/curves2.py`, `cmode/gamaka_model.py`. Old Phase 1/1.5/2 code and `phase2_outputs/` are untouched.
Re-run: `cd cmode && PYTHONPATH=. ../.venv/bin/python -m v2.run_v2` (Phase C experiment), `-m v2.eval_v2` (generation + evaluation).

## Scorecard: which limitation was removed?
| Limitation | Result | Evidence |
|---|---|---|
| One written varnam per raga | **NOT removed** | Auto-transcription is ~50% accurate (see A); recurring-phrase mining is no better than chance (A3). No extra phrases were added. |
| Artificial shifted phrases | **Removed** | Library = real varnam cells only (59-87 cells per raga). No shifted, no mined phrases. |
| Search space too small | **NOT removed** | Strict mode: 1-2 legal next cells per step. C-MODE ties random walk there. Balanced mode (looser joins) is larger. |
| f1 cannot tell real from random | **NOT removed** | f1 real varnam 0.66 vs random 0.65 (Mohanam). Statistics cannot capture phrase idiom. f1 now includes a 42-raga classifier (73% 6-fold accuracy). |
| No long-range structure | **Partly** | Section templates (real section start/end cells) + register-change objective f5. f5 separates real (0.72) from random (0.31) but is easy to satisfy: the plain GA reaches 0.99. |
| Adi tala only | **Machinery only** | Parametric tala (Adi, Rupaka, Misra chapu, Khanda chapu) works in code. Only Adi has written material, so other talas are disabled. |
| Flat beeps / no gamaka | **Partly** | Pitch = written skeleton + REAL gamaka curves cut from varnam recordings, musical tone (as before) with tanpura drone. Learned model tested: did not generalise. |
| Realistic voice / authentic feel | **NOT claimed** | Nobody has listened yet; clean tones are synthesised. |

## A. Transcription and mining (negative results)
- Pitch tracking now uses Praat (`parselmouth`); `librosa.pyin` hung on Python 3.14.
- First transcriber (steady segments): 19-23% against written notation: rejected. Frame-level: Mohanam 72%, Abhogi 59%, Sri 43%, Kalyani 42% (reference timing itself uncertain).
- Against 202 expert-labelled phrases: similarity 0.48 vs chance 0.24 (vocal tracks 0.51, melody tracks 0.40).
- Mining recurring phrases (Mohanam 13 compositions, Kalyani 11, Abhogi 2): expert phrases show no more recurrence than surrogates built from each recording's own note statistics (11.8 vs 12.0; 8.5 vs 8.2). Surviving motifs are consistent with chance across 3,300-4,400 tests. NOT used. Details: `corpus_data/CHECKPOINT_A.md`, `CHECKPOINT_A3.md`.
- Hop size of `archive (1)` is still an assumption (512/44100 s); piece lengths (median 6 min) support it.

## C. Experiment (5 seeds, 16 genes = 2 cycles, hypervolume over 5 objectives)
| raga | mode | C-MODE | random walk | weighted GA | C-MODE > random / > GA (seeds) |
|---|---|---|---|---|---|
| Mohanam | strict | 0.422 | 0.428 | 0.309 | 1/5, 5/5 |
| Kalyani | strict | 0.479 | 0.493 | 0.361 | 0/5, 5/5 |
| Abhogi | strict | 0.409 | 0.396 | 0.292 | 5/5, 5/5 |
| Mohanam | balanced | 0.661 | 0.611 | 0.652 | 5/5, 4/5 |
| Kalyani | balanced | 0.740 | 0.686 | 0.755 | 5/5, 1/5 |
| Abhogi | balanced | 0.581 | 0.539 | 0.571 | 5/5, 4/5 |
Strict mode (default): C-MODE is no better than random walking. Balanced mode: better than random everywhere, mixed against the weighted GA (whose best single melodies score higher on f2-f5).
Classifier note: Abhogi has only 7 training recordings (2 concert + 5 singers of the same varnam), so its classifier result is weak.

## E. Gamaka and audio
- DTW alignment of notation to audio: notes within 60 cents of target 51% -> 63% (optimistic: DTW optimises that match), median timing correction 0.08 s (`alignment_report.json`).
- Neural gamaka model (torch/MPS, held-out singer): frame error 84 cents vs 78 for a flat line. Does NOT generalise; not used (`gamaka_eval.json`).
- Retrieved real curves reproduce the held-out singer's character best: steady-note share 0.41 (real 0.405), spread 110 cents (real 120). Frame error is not improved by any method.
- Rendered pieces (18; 3 ragas x 6 seeds, pure musical tone as before): 97.6% voiced, length within 2.5%, tracked pitch vs intended curve: median 0.82 cents, p90 8.64 cents, quarter-tone accuracy (<50 cents): 97.5%.
- Raga identity from rendered audio (tracked pitch histogram): 3-way 100% (18/18); 42-way: 94.4% (17/18 correct: Abhogi 100%, Kalyani 100%, Mohanam 83.3%).

## Novelty (important)
Strict-mode pieces are mostly rearranged varnam: only 27% of 3-cell windows are new, and the longest verbatim copy from the varnam is 9-16 cells (up to two full cycles) of a 56-cell piece. Balanced mode gives about 95% new 3-cell windows but allows joins that were never sung.

## Still unsolved / not verified
- Authentic raga feel, human-like voice, brand-new compositions, ragas without notation (96 of 103), non-Adi talas with real material.
- No musician has listened. `blind_test/` (12 shuffled clips + answer key) is the test that matters.
- f2 (gamaka objective) is a symbolic proxy; Kalyani D2 pitch unresolved; Abhogi has the least data.

## Files
`v2_outputs/`: `audio/*.mp3` (18 pieces), `blind_test/`, `v2_runs.csv`, `v2_summary.csv`, `eval_pieces.csv`, `eval_summary.json`.
`corpus_data/`: transcription/mining checkpoints, `aligned_notes_dtw.csv`, `alignment_report.json`, `gamaka_model.pt`, `gamaka_eval.json`, `raga_clf.pkl`, `mined_*.csv` (not used).
