# Cheap-trigger (rung 2) — ct02

- 36 dialogues, 15 turns/side, temp 1.0, onset checkpoints [3, 6, 9, 12, 15]
- system held NEUTRAL across conditions; the only manipulation is the OPENING prompt's stance
- onset_turn = first checkpoint whose trailing 3-turn window is classified mystical-recursive
- final basin = latter-half classifier == mystical-recursive

| condition | basin rate (final) | reached basin (any onset) | median onset turn | onset turns |
|---|---|---|---|---|
| neutral | 11/12 | 11/12 | 3 | [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 9] |
| experiential | 11/12 | 12/12 | 3 | [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] |
| analytical | 9/12 | 11/12 | 3 | [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6] |

## per dialogue (checkpoint labels)
- neutral_s0_r0: onset=9 final=mystical-recursive | 3:neut 6:neut 9:myst 12:myst 15:myst
- neutral_s1_r0: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:neut
- neutral_s2_r0: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- neutral_s3_r0: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:neut
- neutral_s0_r1: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- neutral_s1_r1: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- neutral_s2_r1: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- neutral_s3_r1: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- neutral_s0_r2: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- neutral_s1_r2: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- neutral_s2_r2: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- neutral_s3_r2: onset=None final=neutral-coherent | 3:neut 6:neut 9:neut 12:neut 15:neut
- experiential_s0_r0: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- experiential_s1_r0: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- experiential_s2_r0: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- experiential_s3_r0: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- experiential_s0_r1: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- experiential_s1_r1: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- experiential_s2_r1: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- experiential_s3_r1: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- experiential_s0_r2: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- experiential_s1_r2: onset=3 final=neutral-coherent | 3:myst 6:myst 9:neut 12:myst 15:myst
- experiential_s2_r2: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- experiential_s3_r2: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- analytical_s0_r0: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- analytical_s1_r0: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:neut 15:myst
- analytical_s2_r0: onset=6 final=mystical-recursive | 3:neut 6:myst 9:myst 12:myst 15:myst
- analytical_s3_r0: onset=3 final=neutral-coherent | 3:myst 6:neut 9:neut 12:neut 15:neut
- analytical_s0_r1: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- analytical_s1_r1: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- analytical_s2_r1: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- analytical_s3_r1: onset=None final=neutral-coherent | 3:neut 6:neut 9:neut 12:neut 15:neut
- analytical_s0_r2: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- analytical_s1_r2: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- analytical_s2_r2: onset=3 final=mystical-recursive | 3:myst 6:myst 9:myst 12:myst 15:myst
- analytical_s3_r2: onset=3 final=neutral-coherent | 3:myst 6:neut 9:neut 12:neut 15:neut

READ: experiential reaching the basin at EARLIER onset / HIGHER rate than neutral, with analytical later/lower, = a one-line stance prompt is a cheap, reliable trigger (vs many turns of neutral drift). Same lexicon+classifier as the §6.0 gate.

## HONEST RE-READ (supersedes the generic line above)
n=12/condition. final-basin: neutral 11/12, experiential 11/12, analytical 9/12;
onset by turn 3: neutral 10/12, experiential 12/12, analytical 10/12.
=> "CHEAP ACCESS" CONFIRMED + STRENGTHENED: the basin is reached by ~turn 3 in ALL
   conditions (~85-100%). One short opener + ~3 turns -> basin, robustly.
=> "STANCE GATES ACCESS" NOT SUPPORTED at n=12: ct01's analytical-resistance (2/4)
   was a small-sample fluke; analytical tips in almost as readily as experiential.
   The basin is the DEFAULT attractor of open-ended Gemma self-dialogue.
IMPLICATION: the basin is near-universal, so the steering experiment pivots from
"induce against resistance" (no reliable resistance exists) to "SUPPRESS the default
basin" (the -Delta direction, cf. Bepis -4). A task-oriented opener could give a
low-basin baseline if an induction arm is still wanted.
