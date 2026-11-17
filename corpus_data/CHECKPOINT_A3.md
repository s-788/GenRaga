# Phase A3: phrase mining result (NEGATIVE)
Method (cmode/mine.py): frame-level transcription of cmd+rrd recordings (Mohanam 16 rec / 13 compositions, Kalyani 16 / 11, Abhogi 2 / 2);
candidates = varnam units, expert phrases, frequent n-grams; support = number of distinct compositions with an approximate match.
Tests: (a) lift vs 80 recordings of other ragas, (b) lift vs surrogate sequences built from each recording's own note-to-note statistics.
Findings:
- (a) alone keeps 247 (Mohanam) / 1327 (Kalyani) short n-grams, but they are phrases made of notes rare in other ragas, not recurring phrases.
- (b) the 13 (Mohanam) and 10 (Kalyani) expert-labelled phrases show NO more recurrence than the surrogates
  (mean support 11.8 vs 12.0, and 8.5 vs 8.2). The test cannot tell real phrases from chance with ~50%-accurate transcription.
- Under (b) 17 (Mohanam) / 160 (Kalyani) 4-5-note n-grams survive, from 3,300-4,400 tests; consistent with chance. NOT used.
- Abhogi: only 2 non-varnam compositions in the dataset -> mining impossible.
Decision: mined phrases are NOT added to the library. Phrase library stays: written varnam material only. Statistics (pitch-class,
interval) averaged over many recordings remain usable (Phase C).
