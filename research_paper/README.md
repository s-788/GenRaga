# GenRaga — IEEE conference manuscript

## Main files

- `GenRaga_IEEE_Conference.tex`: full editable IEEEtran manuscript, including the approved eight sections and their subsections, original abstract, mathematical definitions, four vector flowcharts, one results chart, tables, algorithm outline, and 12 numbered references.
- `GenRaga_IEEE_Conference.pdf`: compiled **9-page** IEEEtran conference manuscript.
- `Diagrams_and_Recreation_Guide.md`: Mermaid versions, captions, and plain-text recreation instructions for all four flowcharts; description of the results chart.
- `analyze_saved_results.py`: reproducible descriptive analysis of saved project records. It does not execute the music-generation experiments.
- `tables/`: computed CSV summaries, table fragments, and source-file SHA-256 audit.
- `figures/`: vector hypervolume chart and PNG preview.
- `validation_report.json`: automated checks of page count, citation order, cross-references, layout overflow, and unchanged numerical-source hashes.

The compiled draft has been checked for the agreed 8–10-page length, all 12 cited references, all figure/table cross-references, and text beyond page edges. Representative pages and a nine-page contact sheet were also inspected visually. These checks establish document consistency, not independent validation of the project's scientific claims.

## Evidence policy

The manuscript uses the project files in the parent directory as primary evidence. It distinguishes v2 results from earlier proposals, the optimizer benchmark from the longer audio-generation protocol, and structural validity from perceptual authenticity. Dataset numbers, experiment settings, and renderer behavior were cross-checked against selected implementation files. The original project sources and saved outputs were not modified.

The revised abstract uses **7.89–8.08%**, calculated from unrounded saved-run values, rather than the earlier approximation based on rounded summary means. Its broader raga-identification result is explicitly an internal consistency check. It contains no claim of independent listener validation, human-like voice synthesis, universal superiority, or first-ever evolutionary Carnatic generation.

The bibliography contains identifiable publications previously checked against primary publication pages, scholarly records, or the supplied original papers. Metadata verification is not equivalent to full-text review of every publication. In particular, the supplied gamaka PDF is the 2022 version; only bounded bibliographic mention is made of the separately verified 2025 publication. Exact release references for the local Varnam and RRD snapshots have not been guessed.

## Reproduce descriptive tables and chart

From the project root:

```sh
.venv/bin/python research_paper/analyze_saved_results.py
```

The script requires NumPy and Matplotlib already available in the project environment. Standard deviations use `ddof=1` over five seeds. It reads the 90 saved method runs and 18 audio-evaluation records. It computes descriptive means, sample SDs, relative gains, and paired win counts; it does not infer statistical significance or create new experimental observations.

## Compile

Use a LaTeX installation containing IEEEtran, PGF/TikZ, AMS math, graphics, booktabs, and hyperref. From this directory:

```sh
pdflatex -interaction=nonstopmode -halt-on-error GenRaga_IEEE_Conference.tex
pdflatex -interaction=nonstopmode -halt-on-error GenRaga_IEEE_Conference.tex
```

Alternatively upload this directory, including `tables/` and `figures/`, into an IEEEtran-compatible LaTeX editor. References are embedded with `thebibliography`, so BibTeX is not required. Compile after regenerating table fragments if the source result files change.

## Required author review before submission

This is a complete **manuscript draft**, not a claim of conference readiness or acceptance. Resolve the following:

1. Replace author, affiliation, location, and email placeholders. Confirm author order and supervisor co-authorship; a supervisor's name is not added automatically.
2. Select the conference and check its actual page limit, anonymity policy, formatting, and AI-assistance disclosure requirements. An IEEE-style 8–10-page target is not a universal conference allowance.
3. Verify exact dataset release identities, licences, source overlap, attribution wording, and permissions for sharing derived audio or pitch curves. Add authoritative release citations rather than guessing them.
4. Confirm the saved outputs correspond to the intended code snapshot. Historical diagnostic values are labeled as report-level findings rather than newly rerun experiments.
5. Review the limitations with the project supervisor: only one notated composition per raga; three benchmarked ragas; Adi-only evidence; weak objective discrimination; dependent internal classifier; no completed listener study; octave-wrapped audio error; and unchecked joins between separately evolved audio sections.
6. Consider equalized archive/front retention, operator/selection ablations, composition/singer-separated testing, and uncertainty analysis before making stronger comparative claims. These are recommendations, not completed experiments.
7. Conduct musician evaluation before claiming idiomatic authenticity, aesthetic quality, or human-like performance. A prepared blind-listening pack is not itself a completed study.

## Source traceability

| Manuscript content | Primary project evidence |
|---|---|
| Project topic, original objectives | Final Panel Review 1 and Panel Review 2 presentations |
| Current implementation and proposal differences | `Project_Study_Guide.docx`; selected `cmode/` files |
| Dataset counts | `carnatic_unified/recordings.csv`, `ragas.csv`, symbolic tables |
| Representation and phrase coverage | `cmode/phase1.py`, `phase1_outputs/phase1_metrics.csv` |
| V2 library, features, constraints | `cmode/v2/library.py` |
| Five exact objective definitions | `cmode/v2/objectives.py`, `clf.py`, `tala.py` |
| DE, GA, random walk, and hypervolume | `cmode/v2/engine.py`, `cmode/phase2/engine.py`, `run_v2.py` |
| Benchmark results | `v2_outputs/v2_runs.csv`, regenerated `tables/hypervolume_summary.csv` |
| Full-piece generation and audio metric definitions | `cmode/v2/generate.py`, `render.py`, `eval_v2.py` |
| Audio outcomes and reuse | `v2_outputs/eval_pieces.csv`, `eval_summary.json` |
| Alignment and learned-gamaka experiment | `corpus_data/alignment_report.json`, `gamaka_eval.json` |
| Transcription/mining negative results | `corpus_data/CHECKPOINT_A.md`, `CHECKPOINT_A3.md` |
| Rounded objective-diagnostic observations | `v2_outputs/REPORT.md` |
| Routing behavior | `cmode/router.py` |

Hashes of principal numerical sources are recorded in `tables/evidence_audit.json` for review. All research-paper artifacts are contained in this directory; temporary typesetting tools are separate from the project environment.
