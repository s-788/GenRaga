# Phase 2 report: C-MODE (Mohanam, Kalyani, Abhogi)

Code: `cmode/phase2/{library,objectives,engine,baselines,export,run_phase2}.py`.
Run: `cd cmode && PYTHONPATH=. ../.venv/bin/python -m phase2.run_phase2`  (note: this clears `phase2_outputs/` first, including this file)
Tempo: 0.75 s per beat (about 80 bpm, as in the real varnams); set by `AKSHARA_S` in `phase2/export.py`.
Setup: 2 avartanams (8 genes), population 60, 100 generations, 5 seeds, equal evaluation budget (6,060) for all methods.

## Design
- Gene = one 8-subdivision slot: an observed pair phrase, or two cells joined at a legal junction. Contour-shifted cells/pairs (+-1, +-2 scale degrees) enlarge the small vocabulary.
- DE: mutation `E[a] + F(E[b]-E[c])` in a 9-d phrase-feature space, snapped to the nearest filler legal after the previous gene; binomial crossover; repair keeps every individual legal and exactly 32 subdivisions per cycle.
- Selection: NSGA-II. No Markov/probability model.
- Objectives: f1 raga fidelity (recorded pitch profile + interval shape + expert phrases); f2 gamaka proxy (symbolic only); f3 tala alignment; f4 diversity.

## Results (mean of 5 seeds, hypervolume)
See `phase2_summary.csv` and `phase2_runs.csv` for the exact numbers of this run (earlier run: Mohanam 0.812 / 0.802 / 0.789, Kalyani 0.800 / 0.783 / 0.773, Abhogi 0.844 / 0.837 / 0.836 for C-MODE / random CFG / weighted GA).
All final individuals are valid; none is a verbatim copy of a corpus avartanam. C-MODE's margin is small (0.01-0.02), and the plain GA's individual melodies score higher on f2-f4.

## Caveats
1. f1 cannot separate real melodies from random recombinations of corpus phrases.
2. More than half of evolved genes come from shifted (synthetic) phrases.
3. f2 is a symbolic proxy; no gamaka model exists.
4. The corpus is one varnam per raga.
5. Previews are flat sine tones with a drone (no gamaka, no voice) and have not been listened to.
6. Phase 1 pitch labels were corrected from recordings: Abhogi Ga = 300 c, Mohanam Ga = 400 c, Kalyani Ga = 400 c and Ni = 1100 c. Begada/Sahana/Saveri/Sri labels are unverified.

## Files
`<raga>_pareto_front.csv`, `<raga>_best_swara.txt`, `<raga>_best.mid`, `<raga>_best_preview.mp3`, `phase2_runs.csv`, `phase2_summary.csv`, `objective_discrimination.json`.
