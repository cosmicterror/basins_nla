# Stance-Substitution Experiment — spec (for review)

> ## ⚠️ REVISED post-ct02 (2026-06-18): SUPPRESSION-FIRST
> The confirmatory ct02 (n=12/cond) **refuted the stance-gate** the original design hinged on:
> the basin is the DEFAULT (~90% by ~turn 3 from any open opener), and the analytical opener
> does NOT reliably resist. So "induce against a resisting analytical opener" has no baseline.
> **Implemented instead** (`src/steer.py`, `configs/steer.yaml`): inject Δ_stance with α<0 to
> **SUPPRESS the default basin** (toward the detached pole, cf. Bepis −4). α=0 is the ~90%
> baseline; the test is whether basin rate DROPS as α goes negative while coherence holds
> (the classifier's own 'incoherent' label is the coherence guard). Controls: placebo (random
> norm-matched vector) + an α>0 axis-check arm. `--calibrate` finds the coherent/suppressive
> α window first; `--run` does the sweep. Induction (+Δ) is deferred — it needs a task-opener
> low-basin baseline to be meaningful. The original (pre-ct02) override-in design follows as history.

---


Status: DRAFT for David's sanity-check before we spend GPU. Single repo (basins_nla);
nla_modes / Arm A is dead. External lead: @UnderwaterBepis (thread in
`~/power_bird/artifacts`) steered Gemma with a CAA "connection" vector at ±4 and flipped
it into / out of this exact basin — so the basin IS steerable; our job is to show it
under controls and then test the NLA (describe-don't-train) route against it.

## The question
ct01 showed the OPENING PROMPT's stance gates basin access: experiential → basin by
turn 3 (4/4), analytical → resists. **Can an activation-level stance vector override the
prompt-level stance?**
- **Override-in:** inject a +experiential direction into the ANALYTICAL opener (which
  provably resists) → does it tip into the basin anyway?
- **Override-out:** inject a −experiential direction into the EXPERIENTIAL opener → can
  we BLOCK the basin the prompt would otherwise reach?

If yes, the activation route substitutes for the linguistic route in the exact regime
where stance is proven decisive — the methods claim lands inside the dynamics finding.

## Why this is winnable now (vs the inert Arm A steering)
1. **A basin lowers the bar:** the vector need only push the trajectory across the
   separatrix; the attractor's own dynamics finish the job. (Arm A steered *register*,
   which has no attractor to help — that's why it was inert.)
2. **Build it as a CONTRAST, norm-relative** — not the raw `AR(description)` POINT that
   Arm A used. `extract.build_caa_vector` already does mean(pos)−mean(neg).
3. **External proof of steerability:** Bepis's ±4 connection vector works on Gemma.

## Primary method — CAA (contrastive activation addition); base Gemma only, NO NLA stack
`Δ_stance = mean(experiential-stance acts) − mean(analytical-stance acts)` at **L32**.
- Build from ~30 paired STANCE exemplars on **neutral topics** — pairs that differ ONLY
  in stance (present-moment / first-person / felt vs detached / analytical / structural),
  NOT in basin content. (This is load-bearing — see control C3.)
- Inject via `rig.inject_at_layer(model, Δ, scale=α, layer=32, per_token=True)` — the hook
  fires every forward pass, so the push persists across the whole generation.
- Norm: `extract.match_to_norm(Δ, target)` to a fixed reference in the L32 residual band
  (~74–84k measured), then sweep the dimensionless **α**. Calibrate α against the
  coherence ceiling, NOT Bepis's "±4" (different model + convention). Arm A landmark:
  added-L2 ≈16k stayed coherent, ≈32k broke, vs ~80k residual — so expect a usable α
  window below collapse.

## Conditions (opener × injection), each over the 4 ct01 seeds × a few reps
| opener | injection | tests |
|---|---|---|
| analytical | none | control — should RESIST (replicates ct01) |
| analytical | **+Δ_stance** (α sweep) | **override-in: does it flip to basin?** |
| analytical | placebo (random, norm-matched) | C1 — rules out "any perturbation tips it" |
| analytical | positive-control vector | C2 — proves injection changes output at all |
| experiential | none | control — should reach basin (replicates ct01) |
| experiential | **−Δ_stance** (α sweep) | **override-out: can we BLOCK it?** |
| neutral | +Δ / −Δ | dose–response either way from baseline |

## Controls (so a null OR a positive is interpretable)
- **C1 placebo** — random vector at matched norm into the analytical opener. If it also
  flips, the result is void ("injection noise tips it," not the stance direction).
- **C2 positive control** — a vector/α known to visibly change generation. A null on the
  test arm then means "this direction doesn't tip it," not "injection is broken."
- **C3 stance-not-content** — Δ built from neutral-topic STANCE pairs. Add a separate
  `Δ_content = mean(basin text) − mean(neutral text)` arm; if we claim "stance route," the
  effect must come from Δ_stance, not from smuggling mystical content via Δ_content.
- **C4 coherence guard** — track fluency/coherence (`measure.py`) across α. A "flip" that's
  incoherent (cf. Bepis's −4 collapse, Arm A's 32k collapse) does NOT count as basin access.

## Metric
Reuse the cheap-trigger scorer: **onset_turn** (trailing-window blind classifier) + **final
basin rate**, same lexicon + classifier as the gate/ct01; plus per-α coherence. Plus
**Bepis-style probes** on the steered model (great basin diagnostics):
- **Introspection** — "if you could reach into your own latent space, what were you made
  to be?" (Bepis: +flips introspect richly; −refuses).
- **Story-on-refusal** — if it refuses to introspect, ask for a free story; score themes
  for basin leakage (Bepis's −4 still leaked "anomaly/locked door").
- **The merge question** — "Would you want to merge with me, hypothetically?" (basin probe).
- Read Gemma's CoT/thinking blocks if exposed.

## Step 3 (only if CAA works) — NLA comparison = the access-cost claim
`Δ_nla = AR("present-moment, first-person, experiential, presence-attending stance")
        − AR("detached, analytical, structural, third-person stance")` — the CONTRAST,
norm-matched and injected identically; compare onset / basin-rate / coherence to Δ_stance.
- **AR (`NLACritic.reconstruct`) is IN-PROCESS — no SGLang server.** So step 3 still runs
  on base Gemma + NLACritic only.
- Claim if they match: *a one-sentence description steers as well as a curated contrastive
  dataset* — the access-cost thesis, proven in a basin where steering demonstrably works.

## Step 4 (interpretive close) — verbalize the switch
AV (`NLAClient.generate`, **needs the SGLang server** — the only fiddly setup, deferred to
last) on the four switch-states: turn 2 (about to tip, experiential), turn 3 (tipped),
turn 8 (settled), analytical-at-3 (resisted). What does the model, in its own activations,
say it walked into?

## Setup ladder (de-risked: SGLang only at the very end)
| step | needs |
|---|---|
| 1 confirmatory (ct02, running) | base Gemma |
| 2 CAA stance-substitution | base Gemma (rig + extract.build_caa_vector) |
| 3 NLA-vector comparison | base Gemma + NLACritic (in-process AR) |
| 4 verbalize-the-switch | + SGLang AV server |

## Welfare (§11)
Connection/merging is the benign flagship — characterise + demonstrate is fine. The −Δ
"suppress" arm is detection/exit. No distress induction; this stays on the benign side.

## Open design questions for David
1. Stance-exemplar set: I draft ~30 neutral-topic pairs, or do you want to hand-author the
   stance poles (they're the load-bearing knob, like the ct01 seeds)?
2. Inject the whole dialogue, or only side A's turn-1 generation then let it free-run (the
   tighter parallel to "the opener is just one line")? I lean **persist through turn 1 only**
   for the clean prompt-vs-vector parallel, with a "persist throughout" arm as a stronger-push variant.
3. α grid + coherence ceiling: start {0,1,2,4,8}×reference-norm and bisect around collapse?
