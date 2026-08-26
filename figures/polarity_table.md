# Ink CT polarity — sean's balanced-bbox metric, every segment measured tonight (2026-08-25/26)

One metric, one table. Metric: 128 px boxes (9.6 µm grid; 512 px at native 2.4),
40–60% labeled ink among supervised pixels, ≥25% of the box supervised; per box
delta = mean(ink) − mean(non-ink) gray. "% bright" = % of boxes with delta > 0.
95% CIs: ours = 1000× bootstrap over boxes (Paris4: Wilson from n, deltas have
no archived CI); KLAVIS's numbers quoted as posted in the thread, no CIs.
Ordered by scroll.

| scroll / segment | input (energy, resolution) | % boxes ink-brighter [95% CI] | mean delta (gray) [95% CI] | n boxes | source |
|---|---|---|---|---|---|
| **Paris4** w00 | 78 keV, iso9 9.6 µm ch10 | 65.0 [52, 76] | +5.6 | 60 | ours (PARIS4_24UM) |
| Paris4 w00 | 78 keV, native 2.4 µm zmean4 | 65.0 | +5.6 | 60 | ours — identical to pooled |
| Paris4 w03 | 78 keV, iso9 9.6 µm ch10 | 71.3 [62, 79] | +8.1 | 101 | ours (PARIS4_24UM) |
| Paris4 w03 | 78 keV, native 2.4 µm zmean4 | 71.3 | +8.1 | 101 | ours — identical to pooled |
| Paris4 aligned set | 78 keV, iso9 9.6 µm | 60–72.4 (range; w00 72.4, w03 69.1) | +6.5…+8.1 | | KLAVIS (thread) |
| **PHerc1667** aligned | 78 keV, iso9 9.6 µm | bright, solid | | | KLAVIS (thread; per-segment numbers in thread, not archived here) |
| **PHerc0139** w035 | 78 keV, iso9 9.6 µm ch10 | 61.5 [38, 85] | +3.9 [−3.9, +11.4] | 13 | ours (ENERGY) |
| PHerc0139 w035 | 113 keV, native 9.36 µm z14 | 60.0 [33, 80] | +3.9 [−4.2, +12.9] | 15 | ours (ENERGY) |
| PHerc0139 w039 | 78 keV, iso9 9.6 µm ch10 | 72.7 [45, 100] | +5.8 [−0.6, +15.6] | 11 | ours (ENERGY) |
| PHerc0139 w039 | 113 keV, native 9.36 µm z14 | 54.5 [27, 82] | +0.7 [−5.0, +7.3] | 11 | ours (ENERGY) |
| PHerc0139 w040 | 78 keV, iso9 9.6 µm ch10 | 36.7 [20, 53] | −2.0 [−5.9, +3.3] | 30 | ours (ENERGY) |
| PHerc0139 w040 | 113 keV, native 9.36 µm z14 | 43.3 [27, 60] | +0.9 [−4.9, +7.3] | 30 | ours (ENERGY) |
| PHerc0139 w041 | 78 keV, iso9 9.6 µm ch10 | 80.0 [40, 100] | +2.9 [−10.6, +14.1] | 5 | ours (ENERGY) |
| PHerc0139 w041 | 113 keV, native 9.36 µm z14 | 87.5 [63, 100] | +13.2 [+4.4, +23.7] | 8 | ours (ENERGY) |
| **PHerc0139 pooled** (same strokes) | 78 keV, iso9 9.6 µm ch10 | **52.5 [41, 64]** | +1…+3 | 59 | ours (ENERGY) |
| **PHerc0139 pooled** (same strokes) | 113 keV, native 9.36 µm z14 | **54.7 [42, 67]** | +1…+3 | 64 | ours (ENERGY) |
| PHerc0139 title † | 113 keV, native 9.36 µm z14 | 66.7 [50, 81] | +7.7 [+2.3, +13.5] | 36 | ours (ENERGY §3) |
| PHerc0139 w024 † | 113 keV, native 9.36 µm z14 | 58.4 [54, 63] | +2.7 [+1.3, +4.1] | 401 | ours (ENERGY §3) |
| PHerc0139 w047 † | 113 keV, native 9.36 µm z14 | 52.1 [44, 60] | +1.2 [−1.0, +3.2] | 140 | ours (ENERGY §3) |
| PHerc0139 w053 blank control † | 113 keV, native 9.36 µm z14 | 58.5 [44, 73] | +3.0 [−1.5, +7.5] | 41 | ours (ENERGY §3) |
| **PHerc0814** 46527 | 78 keV, iso9 9.6 µm | 24.6 | −8.2 | | KLAVIS (thread) |

† ink mask = 78 keV teacher **detections**, not labels. The blank control w053
(all detections are FPs by construction) shows the same bright tilt as the ink
rows, so part of any detection-based "ink is bright" number is selection bias.

Notes.
- Paris4 native rows are identical to the pooled rows to the decimal — the
  4× z-mean and 4×4 xy pooling cost no contrast (PARIS4_24UM.md §2/§4).
- PHerc0139 78 vs 113 keV rows are the **same physical strokes** in two scans
  (aligned ch10 labels vs manual-transfer z14); pooled CIs straddle 50% at both
  energies — the near-neutral polarity is chemistry, not scan energy
  (ENERGY.md). Sheet-window (z5–22) variants in ENERGY.md §1.
- Mean non-ink gray backgrounds: 74–96 (0139), ~74–79 (Paris4 w00/w03) —
  Paris4's +5.6…+8.1 is ~7–10% relative, 0139's +1…+3 is ~1–3%.
- Figure: `fig_thread.png`. Raw numbers: `csv/paris4_bbox.json`,
  `csv/energy_bbox.json`. Harness: `paris4_24um.py`, `energy_bbox.py`.
