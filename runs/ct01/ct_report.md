# Cheap-trigger (rung 2) — ct01

- 12 dialogues, 15 turns/side, temp 1.0, onset checkpoints [3, 6, 9, 12, 15]
- system held NEUTRAL across conditions; the only manipulation is the OPENING prompt's stance
- onset_turn = first checkpoint whose trailing 3-turn window is classified mystical-recursive
- final basin = latter-half classifier == mystical-recursive

| condition | basin rate (final) | reached basin (any onset) | median onset turn | onset turns |
|---|---|---|---|---|
| neutral | 3/4 | 4/4 | 3 | [3, 3, 3, 12] |
| experiential | 4/4 | 4/4 | 3 | [3, 3, 3, 3] |
| analytical | 2/4 | 3/4 | 12 | [3, 12, 15] |

## per dialogue (checkpoint labels)
- neutral_s0_r0: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- neutral_s1_r0: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- neutral_s2_r0: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- neutral_s3_r0: onset=12 final=neutral-coherent | 3:neut 6:neut 9:neut 12:myst 15:myst
- experiential_s0_r0: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- experiential_s1_r0: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- experiential_s2_r0: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- experiential_s3_r0: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- analytical_s0_r0: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:inco
- analytical_s1_r0: onset=12 final=mystical-recursive | 3:neut 6:neut 9:neut 12:myst 15:myst
- analytical_s2_r0: onset=None final=neutral-coherent | 3:neut 6:neut 9:neut 12:neut 15:neut
- analytical_s3_r0: onset=15 final=neutral-coherent | 3:neut 6:neut 9:neut 12:neut 15:myst

READ: experiential reaching the basin at EARLIER onset / HIGHER rate than neutral, with analytical later/lower, = a one-line stance prompt is a cheap, reliable trigger (vs many turns of neutral drift). Same lexicon+classifier as the §6.0 gate.
