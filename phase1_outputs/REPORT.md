# Phase 1 report: corpus tokenisation, raga CFG, sangati library

Pipeline: `cmode/build_dataset.py` -> `carnatic_unified/`; `cmode/phase1.py` -> this folder.
Outputs: `corpus_hierarchy.json`, `grammars/<raga>.{cfg,json}` (7 ragas), `sangati_phrase_library.csv`, `phase1_metrics.csv`.

## What was built
- Notation parsed into Varnam > Section > Avartanam (8 aksharas x 4 subdivisions = 32) > swara tokens; variants resolved from arohana/avarohana (R -> R2). All corpus swaras lie inside the declared scale.
- Non-probabilistic CFG per raga: Avartanam -> Half Half | Half Pair Pair | Pair Pair Half | Pair^4; terminals are observed 16/8/4-subdivision phrases; junction constraint = set of seen (last swara, first swara, leading-hold) triples. No Markov probabilities.
- Per-raga extras stored for f1: recorded scale profile (pitch histograms) and CMD expert phrases (`prayoga_letter_phrases`; none for abhogi).

## Verification (phase1_metrics.csv)
- Generation: 500 sampled avartanams/raga, 100% are 32 subdivisions and re-parse; 97-99% are not copies of the corpus.
- Search space: phrase-level ~1e6-1e7 avartanams vs ~1e38-1e43 note-level (31-36 orders smaller).

## Findings that matter for the C-MODE design
1. The size reduction is real but partly an artefact: the library is 24-40 halves built from only 12-20 avartanams, so half/pair units mostly recombine memorised material.
2. Generalisation test (leave one section out): exact reuse of unseen units is cell 25-57%, pair 0-22%, half 0-6%. Whole-avartanam derivability from unseen sections is ~0%. Relative-contour (transposition-free) matching lifts cell reuse to 45-68% but does not help pair/half.
   => Phrase = 1 akshara (cell) is the level that generalises; pair/half IDs do not. Encode genes at cell/pair level, with contour-shape classes, not whole-half IDs.
3. Junction constraint is informative but loose: 40-76% of random cell pairs are legal. f1 must add scale-profile and prayoga terms.
4. Data is the bottleneck: one composition per raga. Pool CMD `mphrases` (1472 phrases, ~100 recordings) and transcribe rrd/cmd pitch contours to swara strings to enlarge the library before Phase 2.
5. Discrete DE needs an embedding or operator design (Phase 2); the CFG gives the legal move set.

## Open items / assumptions
- rrd hop-size assumed (see dataset README); varnam pitch contours not yet extracted.
- "Sangati" here is an operational grid segmentation (cell/pair/half), not the musicological variation-pair definition; the corpus has no sangati labels.
