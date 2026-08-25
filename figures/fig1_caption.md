# Figure 1: PHerc0139 title line, native 113 keV, four ways

`fig1_titleline_comparison.png`, region A_titleline_left (x1050 y900 w1300 h400
on the title segment's 9.362 um grid; region frozen 2026-08-05, unchanged
since).

Panels, top to bottom:
1. **113 keV mean, layers 8-20**: the native-resolution scan. Nothing readable
   by eye.
2. **The 78 keV / 2.4 um ink map** published with the segment, the map the
   title was read from (*Philodemus, On Gods*).
3. **Control checkpoint** (identical training minus the native-dense data) on
   the 113 keV input.
4. **dense_native-016000** (this repo) on the same 113 keV input.

Measured against panel 2, restricted to valid pixels in this region: ink =
published 78 keV map (resized onto the 113 keV grid) >= 128, background = all
other valid pixels, no ignore band (this differs from the frozen benchmark's
lo/hi class construction). `fig1_metrics.py` in this directory recomputes
every number in the table from the prediction maps:

| | AUC | pixel recall @ matched 5% FP | ink components hit (>=30% covered) |
|---|---|---|---|
| control | 0.8703 | 52.5% | 4 / 7 |
| dense_native-016000 | **0.9036** | **59.4%** | 4 / 7 |

The plain reading: the native-dense data buys pixel-level fidelity (+3.3 AUC,
+6.9 recall) on the letters that are already found, and **no additional
letters** on this line. The remaining gap to panel 2 is dominated by the
acquisition (2.4 um / 78 keV versus 9.362 um / 113 keV), not by the model.
