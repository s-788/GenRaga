# GenRaga: An Evolutionary AI Framework for Carnatic Music Generation

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Website](https://img.shields.io/badge/Live%20Studio-GitHub%20Pages-success.svg)](https://s-788.github.io/GenRaga/)
[![Ragas Supported](https://img.shields.io/badge/Ragas-7%20Foundation%20Ragas-purple.svg)](https://s-788.github.io/GenRaga/)
[![Validity](https://img.shields.io/badge/Structural%20Validity-100%25-brightgreen.svg)]()

> **Interactive Web Studio:** [https://s-788.github.io/GenRaga/](https://s-788.github.io/GenRaga/)  
> **Authors:** Shrinidhi Maheshwaran (`cb.sc.p2cse25033@cb.students.amrita.edu`) & Dr. Ritwik M (`m_ritwik@cb.amrita.edu`)  
> **Institution:** Department of Computer Science and Engineering, Amrita School of Computing, Amrita Vishwa Vidyapeetham, Ettimadai, India.

---

## 🎵 Overview

**GenRaga** is an evolutionary artificial intelligence framework for authentic, rule-compliant Carnatic music generation. It formulates melody composition as a **Constrained Multi-Objective Differential Evolution (C-MODE)** problem over corpus-derived rhythmic phrase cells, combined with **empirical microtonal gamaka synthesis** extracted from professional concert recordings.

```
┌─────────────────────────┐     ┌────────────────────────┐     ┌───────────────────────────┐
│  Consolidated Corpus    │ ──► │  C-MODE Optimization   │ ──► │  Empirical Gamaka Synth   │
│  705 Tracks, 103 Ragas  │     │  DE in R^9 + NSGA-II   │     │  27,731 Aligned Notes     │
│  7 Foundation Grammars  │     │  5 Musical Objectives  │     │  Violin / Veena / Tone    │
└─────────────────────────┘     └────────────────────────┘     └───────────────────────────┘
```

---

## ✨ Key Capabilities

1. **7 Foundation Ragas Supported:** *Mohanam, Kalyani, Abhogi, Begada, Sahana, Saveri, Sri*.
2. **Multi-Timbral 3-Instrument Audio Synthesis:**
   * 🎻 **Violin Rendition:** 14 harmonic overtones + 5.3 Hz natural vibrato + gamaka glides + tanpura drone.
   * 🪕 **Veena Rendition:** 16 harmonics + plucked exponential decay attack + gamaka modulation + tanpura drone.
   * 🪈 **Vocal / Tone Rendition:** Additive harmonic spectrum + microtonal curves + tanpura drone.
3. **Dynamic Tanpura Sruti Calibration:** Standard Panchama-sruti ($Sa+Pa$) for 6 ragas, and dedicated Madhyama-sruti ($Sa+Ma$) for $Pa$-omitted ragas (*Abhogi*).
4. **Pareto Multi-Objective Search:** Optimizes across 5 competing musical criteria (Raga Melodic Fidelity $f_1$, Gamaka Readiness $f_2$, Tala Metric Alignment $f_3$, Melodic Diversity $f_4$, and Structure $f_5$).
5. **100% Structural Legality:** Enforces exact 32-subdivision Adi tala meter and swara grammar.

---

## 📊 Benchmark Results Across All 7 Ragas

Evaluated under strictly equal candidate-evaluation budgets (3,660 evaluations per run):

| Raga | Scale Trajectory | Unique Cells | C-MODE Hypervolume | Random Walk Baseline | Weighted GA Baseline | C-MODE Gain vs. Random | Validity |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Begada** | $S\ R_2\ G_2\ M_1\ P\ D_2\ N_2$ | 131 | **0.5116** | 0.3929 | 0.5078 | **+30.2%** | **100%** |
| **Sri** | $S\ R_2\ G_2\ M_1\ P\ D\ N_2$ | 74 | **0.5909** | 0.4553 | 0.6130 | **+29.8%** | **100%** |
| **Mohanam** | $S\ R_2\ G_2\ P\ D_2$ | 73 | **0.6063** | 0.5358 | 0.6374 | **+13.2%** | **100%** |
| **Sahana** | $S\ R_2\ G_3\ M_1\ P\ D_2\ N_2$ | 85 | **0.6245** | 0.5717 | 0.5612 | **+9.2%** | **100%** |
| **Saveri** | $S\ R_1\ G_2\ M_1\ P\ D_1\ N_2$ | 94 | **0.5237** | 0.4851 | 0.4942 | **+8.0%** | **100%** |
| **Abhogi** | $S\ R_2\ G_1\ M_1\ D_2$ | 79 | **0.5917** | 0.5497 | 0.6088 | **+7.6%** | **100%** |
| **Kalyani** | $S\ R_2\ G_2\ M_2\ P\ D_2\ N_2$ | 87 | **0.6819** | 0.6724 | 0.7403 | **+1.4%** | **100%** |

* **Pitch Accuracy:** Median cents error of **0.824 cents** with **97.46% of frames within quarter-tone accuracy (<50 cents)**.
* **Raga Classification:** Calibrated 42-class classifier achieves **94.4% (17/18) accuracy** on synthesized audio.
* **Field Study / Turing Test:** **74.4% of GenRaga compositions** were classified as human-composed in a double-blind study ($N=30$).

---

## 🚀 Quickstart & CLI Usage

### 1. Installation
```bash
git clone https://github.com/s-788/GenRaga.git
cd GenRaga
pip install -r requirements.txt
```

### 2. Generate a Complete Composition with Audio
```python
from v2.generate import generate

# Generate Mohanam on Violin
res = generate("mohanam", "output/mohanam_violin.mp3", instrument="violin")
print("Generated Swara Notation:\n", res["swara"])

# Generate Kalyani on Veena
res = generate("kalyani", "output/kalyani_veena.mp3", instrument="veena")

# Generate Abhogi on Pure Tone
res = generate("abhogi", "output/abhogi_tone.mp3", instrument="tone")
```

### 3. Natural-Language Query Router
```bash
# Retrieves authentic recording if present; otherwise generates
python cmode/router.py "Varnam in Sahana"
```

---

## 📂 Project Directory Structure

```text
GenRaga/
├── docs/                              # Live Web Studio (GitHub Pages)
│   ├── index.html                     # Prompt-to-swara & 3-instrument web player
│   ├── studio_data.json               # Swara notation & audio registry
│   └── audio/                         # 21 pre-rendered audio tracks (7 ragas x 3 instruments)
├── cmode/                             # Core Python algorithmic package
│   ├── v2/                            # C-MODE evolutionary engine & objectives
│   ├── build_dataset.py               # Manifest aggregator
│   ├── curves2.py                     # 27,731 notes DTW alignment pipeline
│   └── router.py                      # Request router
├── research_paper/                    # IEEE Conference Publication
│   ├── GenRaga_IEEE_Conference.pdf    # 8-page compiled PDF
│   ├── GenRaga_IEEE_Conference.tex    # LaTeX source with 40 references
│   └── CSE25033_User_Study.pdf        # Formal User Study & Turing Test Report
├── CSE25033_website.url               # University submission shortcut for Web Studio
├── CSE25033_repository.url            # University submission shortcut for GitHub Repo
├── CSE25033_pdf.pdf                   # User study submission PDF
├── requirements.txt                   # Dependencies
└── README.md                          # Documentation
```

---

## 📜 Citation

```bibtex
@inproceedings{maheshwaran2026genraga,
  author    = {Shrinidhi Maheshwaran and Ritwik M},
  title     = {GenRaga: Constrained Multi-Objective Differential Evolution for Carnatic Melody Generation},
  booktitle = {Proceedings of the IEEE International Conference},
  year      = {2026},
  address   = {Coimbatore, India}
}
```

---
**License:** MIT License. All concert performance recordings from Carnatic Varnam 1.1, CompMusic, and Saraga are utilized under Creative Commons Attribution licenses.
