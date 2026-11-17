# Phase A checkpoint: how accurate is automatic transcription?
Tool: cmode/transcribe.py (Praat pitch -> frame-level quantisation to the raga's pitch map -> merged notes).

1. Against the WRITTEN varnam notation (28 recordings, timing from tala ticks):
   - first version (steady-segment rule): note accuracy 19-23% -> rejected.
   - frame-level version: Mohanam 72%, Abhogi 59%, Sri 43%, Kalyani 42% (4 recordings checked). The reference itself is uncertain
     (equal sub-division of beats; Kalyani notated Ma is heard as Pa, Ni as Sa).
2. Against 202 EXPERT-labelled melodic phrases in concert recordings (alignment-free): similarity 0.48 vs chance 0.24
   (vocal-pitch tracks 0.51, melody tracks 0.40). Per raga 0.39-0.52.
Conclusion: transcription is clearly better than chance but about half of the notes are wrong. Individual transcribed
phrases cannot be trusted; only recurring patterns across many recordings can.
