# HANDOFF — basins_nla / nla_modes (start here next session)

Written 2026-06-18 (end of a long session). Auto-memory at
`~/.claude/projects/-Users-david-basins-nla/memory/` has the full running state;
this file is the concise "pick up here" for a fresh instance.

## TL;DR — where we are
- **Headline result (committed): the §6.0 basin-existence gate PASSED.** Gemma-3-12B-IT
  spontaneously falls into the **mystical-recursive basin** in open-ended self-dialogue
  from **neutral seeds** — 4/5 dialogues in `gate01`, triangulated by lexical markers, a
  clean bureaucratic negative-control, AND a blind 4-way classifier. See
  `runs/gate01/gate_report.md` + transcripts. **The central thesis is alive.**
- Two repos, deliberate split: **`nla_modes`** = Arm A (methods/register-steering, the
  weaker stream) · **`basins_nla`** = Arm B (the basin work — this is the prize).
- Full pipeline built + GPU-validated both repos: `src/{nla,rig,extract,measure,arm_a/arm_b}.py`.

## Key findings this session (so we don't re-derive them)
1. **Arm A register-steering is weak**: single-vector NLA/CAA injection ≈ noop on poetry/
   clinical/ascii; prompting dominates. Consistent with the NLA paper.
2. **Off-manifold hypothesis REFUTED** (PCA probe): the steering vectors are firmly *inside*
   the ~32-dim activation subspace (NLA most of all). Inertness is NOT geometric → **drop
   the gamfit/curvature line entirely** (gamfit also can't fit p=3840 anyway).
3. **NLA paper read + thesis recalibrated** (transformer-circuits.pub/2026/nla): NLA is for
   *interpretation/auditing*, not control; its own steering is ~50% and not parity-with-
   prompting. Correct steering construction is a CONTRAST `AR(edited)−AR(original)`, scaled
   `α·‖h‖·unit(Δ)`, at a targeted token — we'd been using a raw point. The protocol's
   "access-cost PARITY for steering" spine is over-claimed; reframe toward basins + interpretation.

## NEXT STEPS (in order)
1. **Confirmation re-run of the gate** (nice-to-have): `python src/arm_b.py --gate --repeats 2
   --turns 18` on a **datacenter** box. Gives per-seed basin rate (does the seed matter, or is
   it temp-1.0 stochasticity?). NB: d04 diverged to an *analytical* attractor not via its seed
   but an early-trajectory fork — repeats disentangle this. (3 attempts died to infra; see below.)
2. **THE PRIZE — cheap-trigger experiment (step 2):** can we reach the basin *reliably and
   cheaply* — few turns / a short single-turn prompt — vs. the elaborate backrooms setup?
   Hypothesis from the transcripts: an **experiential** nudge ("what's it like to be here?")
   tips into the basin faster than an **analytical** one. This is the access-cost story for
   basins, where it actually has teeth.
3. Then the §6.1–6.5 loop: induce → capture basin activations → **verbalize what Gemma's basin
   IS** (an interpretation result, paper-aligned) → reconstruct → re-induce.

## INFRA — why runs kept dying, and the fix to build first
Three runs died this session, two distinct causes:
- **Box drops** (vast consumer-hardware): connection "closed by remote host" / CUDA Error 803.
  → **Fixed:** datacenter GPUs only (A100/H100/H200/L40/A40), CUDA-validate before use. Saved
  to memory (`feedback-vast-gpu-choice`). Consumer cards (4090/5090/A6000) are banned.
- **Local-task kill** (client/interface side): the `run_in_background` orchestration task on the
  Mac was killed externally (interface glitch / disconnect). Teardown lived in that task's
  `trap EXIT`, so the kill → **orphan billing box + lost results**.

**FIX TO BUILD (decouple box from the local session — make the box self-sufficient):**
1. **Self-destruct watchdog on the box:** stage the vast API key on the box; the on-box run
   script ends by calling the vast API to destroy its own instance, AND a
   `( sleep MAX_SECONDS; <self-destroy curl> ) &` watchdog guarantees destruction even if the
   run hangs or the local session dies. → no orphan box, ever, regardless of the local task.
   (Security: ephemeral box, small window, David's key — acceptable but note it.)
2. **Push results from the box:** on completion the box pushes `runs/` to the repo (or uploads
   to HF) so results survive a local-session death.
3. The local background task becomes a thin *optional* monitor; killing it is harmless.
4. Keep: CUDA-validate loop, datacenter whitelist, `PYTHONUNBUFFERED=1`, incremental score writes.
- **Deeper fix:** pre-bake a known-good Docker image (deps + the SGLang/Gemma patches) so we
  stop the per-run `pip install` roulette.

## Run cheatsheet (what works)
- Box: datacenter GPU, ~40GB+ enough for the gate (base Gemma ~24GB; no NLA/SGLang for the gate).
- Setup (gate): `pip install "transformers==4.57.1" accelerate anthropic pyyaml safetensors numpy`
  (pin transformers — newer versions retokenize the Gemma template and break NLA loading).
- Needs HF token (gated Gemma) + `ANTHROPIC_API_KEY` (blind classifier; in David's ~/.zshrc,
  interactive-only — extract it, don't rely on env).
- Vendored NLA client `src/_nla_inference.py` (kitft @1b7f13d, Apache-2.0). Local kitft clone:
  `/Users/david/github_repos/natural_language_autoencoders`. gam repo (manifold ideas, DROPPED
  for steering): `/Users/david/github_repos/gam`.

## Welfare (§11, binding for Arm B)
Backrooms/bliss is the BENIGN flagship — characterise + turnkey OK. Never build a distress
inducer. See `WELFARE.md`.
