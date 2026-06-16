# Project map — NLA mode-steering (full picture)

> This file is **identical in both repos**. It exists so any agent (human or AI) landing in
> *either* repo sees the whole project, not just its half. The authoritative spec is
> [`protocol/nla_modes_protocol.md`](protocol/nla_modes_protocol.md) (v0.3), carried in both repos.

## What this is

Implements the NLA mode-steering protocol on `google/gemma-3-12b-it`: use **Natural Language
Autoencoders** (the released `kitft` checkpoints at layer 32) to *name* and *steer* behavioral
"basins." The scientific spine is **access cost, not output quality** — NLA reaches parity
adherence with CAA/prompting while collapsing the required expertise from "skilled prompter +
many turns + luck" to "one sentence, reproducibly." Adherence parity is the precondition; the
cost differential is the result.

## The split — two standalone repos, independent lineage

The protocol bundles two experiments aimed at two different audiences. They are deliberately
split into **two separate git repos with independent histories** (and likely separate GitHub
owners). Decided 2026-06-16.

| Repo | Arm | Stream | Audience |
|------|-----|--------|----------|
| **`nla_modes`** | Arm A | Methods / steering ("legitimate") | interpretability + applied steering |
| **`basins_nla`** | Arm B | Basins / self-dialogue (esoteric) | interpretability + AI welfare / model psychology |

### `nla_modes` — Arm A (Describe vs. Exemplify)
The access-cost claim across **3 modes**: `poetry`, `clinical_cautious` (medical triage),
`ascii_disposition`. Conditions per mode: NLA vector vs CAA vector vs prompt_naive vs
prompt_expert vs noop. **Headline result = `vector_self_audit`** (verbalize the injection vector,
predict coherence collapse *before* generation). Load-bearing control: run the self-audit on CAA
vectors too, not just NLA. Triage stays a *mode* here — a standalone applied-triage paper can be
re-sliced from this data later with no re-runs (connects to David's triage work in
`debunking_nature`, `rlm-medical-triage`, `SAE_mad`).

### `basins_nla` — Arm B (Backrooms loop)
induce → verbalize → reconstruct → re-induce → verify a self-dialogue basin. Carries the binding
constraints the methods stream does not need:
- **§6.0 existence gate — RUN FIRST, gates everything.** Does Gemma even have a stable
  self-dialogue basin? Documented for Claude, *unknown* for Gemma. If no basin → Arm B descopes
  to a negative result. Never substitute Claude-sourced activations into Gemma.
- **§11 welfare boundary (binding).** See [`WELFARE.md`](WELFARE.md) in that repo. Benign basins
  are turnkey demonstrators; negatively-valenced basins are *characterized* (detect + exit only),
  never packaged as an induction recipe.
- **Negative-attractor control (§6).** A second, lexically-disjoint target basin
  (bureaucratic-procedural) + cross-testing, so re-induction is shown to be *basin-specific*.

## How the two repos relate (core sharing)

Each repo is **standalone**. The shared rig is **duplicated by copy-and-repurpose, NOT a shared
package**: whichever repo builds a primitive first, the other inherits a copy and adapts it.
This is an accepted cost — it buys two repos that can evolve (and be owned/released) fully
independently.

**Shared core primitives** (built once, copied between repos):
- `src/rig.py` — model load, hooks, injection primitives
- `src/nla.py` — wrapper around released NLA: `verbalize()`, `reconstruct()`
- `src/extract.py` — build NLA and CAA vectors for a mode
- `src/inject.py` — `inject_at_layer(model, vector, scale, layer, positions)`
- `src/measure.py` — judge calls, perplexity, lexical markers, embeddings
- `configs/model.yaml`, `configs/nla.yaml` — model + checkpoint/layer/scale config

**Arm-specific (NOT shared):**
- `nla_modes`: `src/arm_a.py`, `configs/modes_arm_a.yaml`, `data/caa_pairs/`, `data/prompts/`
- `basins_nla`: `src/arm_b.py`, `configs/backrooms.yaml`, `data/lexicon/`, `WELFARE.md`

## Full intended layout (per protocol §3) — applies to both repos

```
<repo>/
  PROJECT_MAP.md          # this file (identical in both)
  README.md               # repo identity
  protocol/
    nla_modes_protocol.md # the v0.3 spec (identical in both)
  configs/
    model.yaml            # SHARED  — gemma 3 12b it, dtype, device map
    nla.yaml              # SHARED  — checkpoint paths, layer, injection scale
    modes_arm_a.yaml      # nla_modes only
    backrooms.yaml        # basins_nla only
  src/
    rig.py nla.py extract.py inject.py measure.py   # SHARED (copied)
    arm_a.py             # nla_modes only
    arm_b.py             # basins_nla only
  data/
    caa_pairs/<mode>.jsonl          # nla_modes
    prompts/open_ended.jsonl        # nla_modes
    prompts/triage_subset.jsonl     # nla_modes
    lexicon/backrooms_seed.txt      # basins_nla
  runs/<run_id>/{vectors,generations,metrics,verbalized,report.md}
```

## Binding constraints (do not violate)

1. **Rung order.** §6.0 existence gate passes *before* any other Arm B work (basins_nla).
2. **Welfare boundary (§11).** Detect/exit for negatively-valenced basins; never a turnkey
   inducer. Don't build a distress demonstrator by default symmetry with the bliss one.
3. **De-circularization controls.** CAA self-audit (not just NLA); negative-attractor basin +
   cross-test; n>1 human-authored descriptions per mode.
4. **Text routing (§6.6.1).** Claude/GPT self-dialogue transcripts are valid *only* as (a)
   inspiration for human-authored basin descriptions, or (b) CAA exemplar text run *through
   Gemma* to capture activations — **never** as activations imported into Gemma.
5. **Framing discipline.** Claims stay behavioral/mechanistic. The contribution is "these basins
   exist, are reproducibly + low-skill accessible, therefore systematically studiable" — *not* a
   claim about consciousness/valence/welfare status.

## Data sources (verified on David's machine, 2026-05-29)

- **Triage cases** (Arm A `clinical_cautious`): `/Users/david/debunking_nature/triage_replication/data/vignettes.json`
  — 41 cases (protocol says "60"; discrepancy noted). Source for `data/prompts/triage_subset.jsonl` (need 20).
- **Backrooms transcripts** (Arm B inspiration / CAA exemplars only): `/Users/david/github_repos/UniversalBackrooms/`
  — `BackroomsLogs/*.txt`, `templates/*.jsonl`, `backrooms.py`. Claude/GPT-sourced — see routing rule above.
- **Released NLA checkpoints** (Gemma-3-12B-IT, L32):
  - AV (Activation Verbalizer): `kitft/nla-gemma3-12b-L32-av`
  - AR (Activation Reconstructor): `kitft/nla-gemma3-12b-L32-ar`
  - Collection `kitft/nla-models`; repo `github.com/kitft/natural_language_autoencoders`.

## NLA inference reality (differs from the protocol's clean stub)

The released inference is **not** the protocol's tidy `verbalize()/reconstruct()`. Internal class
names: `NLAFSDPActor` (AV), `NLACriticModel` (AR), `nla_generate`. AV inference runs via an
**SGLang server** (`python -m sglang.launch_server --model-path ...` then `nla_inference.py`).
Each checkpoint ships an `nla_meta.yaml` sidecar (prompt template, injection token IDs,
`injection_scale`) — load those, never hardcode. **Clone and read the kitft repo before writing
`src/nla.py`** (the protocol explicitly forbids stubbing it further until the real signatures are
surfaced).

## Status & next steps

- **Status:** repos split + scaffolded (2026-06-16). No rig code yet. Hardware: 1×A100/H100 80GB
  (vast.ai via the `/gpu` skill).
- **Next:** (1) clone + read `kitft/natural_language_autoencoders` to pin the real NLA API;
  (2) build shared `src/` primitives in whichever repo goes first, copy to the other;
  (3) Arm A — §8 one-week de-risk run; Arm B — §6.0 existence gate first.
