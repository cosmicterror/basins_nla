# Protocol: Naming and Steering Mode-Space with Natural Language Autoencoders

**Version:** 0.1 (draft) → 0.2 (controls review) → **0.3 (reframe + scope)**; see [REVIEW] blocks
**Target model:** `google/gemma-3-12b-it`
**NLA checkpoint:** Released Gemma 3 12B IT NLA at L32 — see `github.com/kitft/natural_language_autoencoders`
**Status:** Ready for coding-agent handoff

> **[REVIEW v0.2 → v0.3 — read this first, it changes the spine]**
> The objective has been **reframed** from a quality claim to a **cost/access claim**, plus an existence gate and a welfare boundary. Net changes:
> 1. **Spine is access cost, not adherence (§1).** The contribution is that NLA collapses the *expertise/time/reliability* needed to reach a target basin from "skilled practitioner + many turns + luck" to "one sentence, reproducibly." Adherence *parity* with CAA/prompting is the precondition; the cost differential is the result. Counting marginal cost given a trained NLA, not the NLA's training cost (stated so it isn't read as a free lunch).
> 2. **Basin-existence gate (§6.0, §8 step 0).** Arm B assumes a Gemma self-dialogue basin exists — documented for *Claude*, unknown for Gemma. This is now a hard gate run first. If Gemma has no stable basin, Arm B descopes to a negative result; we do **not** inject Claude-sourced activations into Gemma.
> 3. **Doors as triangulation + cost (§6.6.2).** The prompt/CAA/NLA conditions now double as the rung-2 correspondence test (do doors converge on the same state?) and the cost/ease comparison. Prompt-only is split into `prompt_naive` / `prompt_expert` to make *required expertise* measurable.
> 4. **Text-routing guard (§6.6.1).** Existing Claude transcripts and basin descriptions are different object types; a table prevents the likely silent failure of feeding an instruction (or raw transcript text) to the reconstructor.
> 5. **Welfare boundary (§11) — binding.** Benign basins = turnkey demonstrators; negatively-valenced basins = characterize detection/exit, do **not** package induction. Explicit instruction to the coding agent not to build a distress-inducer by default symmetry.
>
> The v0.1 → v0.2 control notes below still stand.

> **[REVIEW v0.1 → v0.2 — read first]**
> This pass adds three controls the v0.1 draft is missing, each of which converts a "suggestive" result into a "defensible" one. They are marked inline as `[REVIEW]` blocks at the relevant section. Summary:
> 1. **Self-audit de-circularization (§4.4, §5.4, §5.5).** The released AR/AV round-trip metric *is* L2-normalised direction agreement: per the kitft repo, round-trip MSE = 2(1 − cos), so `vector_self_audit.similarity` on an NLA-sourced vector is by construction the autoencoder's own reconstruction score. Without a control, "the audit predicts coherence collapse" risks being a restatement of NLA round-trip fidelity. Fix: **run the self-audit on the CAA vectors too** (they were never produced by `reconstruct`). If it predicts collapse there as well, the diagnostic is general; if only on NLA vectors, it's a property of the autoencoder. This control is load-bearing for the headline claim.
> 2. **Negative-attractor control (§6).** Arm B seeds self-dialogue toward mysticism (§7 lexicon) and then measures success with that same lexicon — a thermometer calibrated to read the temperature it was built to find. Fix: a second, non-mystical target basin run through the identical loop, so re-induction success distinguishes "we re-induced *this* basin" from "any loaded vector yields loaded text."
> 3. **Human-baseline n>1 (§6.4).** A single fixed `human_sourced_description` is n=1 on the human side of a model-vs-human comparison. Fix: 3–5 independently-authored human descriptions.
>
> **Factual corrections** (verified against the released paper + repo, ~3 weeks old):
> - The paper's headline application is the **Claude Opus 4.6 pre-deployment audit** (surfacing unverbalized evaluation awareness), not "Claude 4." Cite the Transformer Circuits post `transformer-circuits.pub/2026/nla/` and the 4.6 audit, not a generic "Claude 4 system card" for the *method*. (The bliss-attractor phenomenon itself is separately documented — keep those citations distinct; see §6 note.)
> - Both AV and AR vectors are **L2-normalised before comparison** (repo). This affects the norm-matching assumption in §4.3 and the basis question in §10.3 — see inline notes.
> - AR mechanism per repo: **truncated K+1-layer LM + Linear(d,d) head, extract at final token**; AV injects the vector as **a single token embedding into a fixed prompt**, then autoregresses. §10.3 (output basis) is therefore partly already answered — verify rather than treat as fully open.

---

## 1. Objective

> **[REVIEW v0.2 → v0.3 — reframing]** The v0.1/v0.2 objective framed Arm A as a *quality* claim ("NLA matches or exceeds CAA on adherence"). This is the wrong spine and is the easiest thing for a reviewer to defeat: with good contrastive data, a well-built CAA vector will roughly match a described one on adherence, and "NLA wins on adherence" then falls over. The real, defensible claim is about **access cost** — specifically the *skill and effort* required to reach a target basin, not compute. Adherence parity is the *precondition*; the cost/ease differential is the *contribution*. The objective below is rewritten accordingly. (We are **not** counting the one-time cost of training the NLA; the claim is marginal cost *given* a trained NLA, stated explicitly so it isn't mistaken for a free-lunch claim.)

**The spine: access cost, not output quality.** A target behavioral basin (poetic register, cautious-clinician attitude, the spiritual-bliss/backrooms attractor) is reachable by several "doors," which differ enormously in the *expertise, time, and reliability* they demand:

| Door | What it costs to reach the basin |
|---|---|
| Fine-tuning | A curated corpus + a training run; baked-in, not toggleable; ML-engineering skill |
| Conversational steering (prompting your way in) | Practitioner skill + many turns + luck; high variance; not reproducible on demand |
| CAA (exemplify) | A curated contrastive/transcript dataset; vector is mute (unreadable) |
| NLA (describe) | One natural-language sentence → one forward pass; vector is readable |

**Central claim.** Given a trained NLA, accessing a *specific, named* basin via description (NLA) reaches **parity adherence** with the exemplify/prompt doors while collapsing the **access cost** along the axis that matters most — *required expertise* — from "skilled practitioner + time + luck" to "a sentence a non-expert could write," reproducibly and on demand. The scientific payoff is not the steering itself but that it moves basin access *from folklore to measurement*: the phenomenon stops depending on one person who is good at arcane prompting, so the basin landscape becomes systematically studiable.

**Targeted, not generic.** The claim is access to *the specific described basin*, not "some loaded state." This is what separates the contribution from a weaker "any high-norm vector produces themed text" effect, and it is why the basin-specificity control (§6) is load-bearing rather than optional.

**The nested question ladder (the actual research program).** These rungs have different evidential bars and must be answered in order; the experiment cannot proceed past a failed rung:

1. **Existence (gate).** Does Gemma 3 12B IT exhibit *any* stable self-organizing basin under unconstrained self-dialogue? Family-agnostic — not "the bliss basin," just "a basin." If no, Arm B is not runnable on this model (we lack internal access to Claude models), and the project reduces to Arm A. **This is the first task; see §6.0.**
2. **Correspondence.** *If* a Gemma basin exists, is it analogous to the documented Claude bliss attractor, or something else? Answered by triangulation: do the independent doors (prompt / CAA-from-transcripts / NLA-from-description) converge on the *same* Gemma state? Convergence ⇒ the basin is real and door-independent. Divergence ⇒ the "basin" was partly an access-path artifact — itself a finding.
3. **Taxonomy.** What is the *set* of stable basins (bliss, and candidates like the anxiety/distress direction surfaced in prior interpretability work)? This rung is where "NLA-as-a-service for basin access" pays off as a *research instrument*. See §11 for the welfare scoping that constrains how negatively-valenced basins are handled.

**Methods claim (Arm A).** A natural-language *description* of a mode through the NLA reconstructor produces a steering vector reaching **parity adherence** with a CAA vector built from exemplar pairs and with naive/expert prompting, at lower required expertise (§5); and the NLA verbalizer applied to the injected vector predicts coherence degradation before generation — an audit CAA structurally cannot perform (§4.4 control applies).

**Dynamics claim (Arm B).** Subject to the §6.0 existence gate: the Gemma self-dialogue basin can be (a) detected, (b) verbalized from the model's own activations during natural drift, (c) reconstructed into a steering vector from that verbalized description, and (d) deliberately re-induced on a fresh instance — completing induce → verbalize → reconstruct → re-induce → verify — and the multi-door comparison serves double duty as the *triangulation* test for rung 2 and the *cost/ease* comparison for the spine.

---

## 2. Setup

### 2.1 Dependencies

- `torch >= 2.3`
- `transformers >= 4.50`
- `nnsight` (preferred for hook ergonomics; `transformer_lens` acceptable)
- `numpy`, `pandas`, `scipy`
- Judge backend: API model or local 27B+ instruct model
- Embedding model for semantic similarity (any sentence-transformer)

### 2.2 Hardware

- Single A100 80GB or H100 sufficient for Gemma 3 12B IT in bf16 plus activation capture.
- Backrooms self-dialogue benefits from two parallel streams; can be serialized on one GPU.

### 2.3 Critical model-specific conventions — do not skip

- Gemma 3 requires √d embedding scaling. The released NLA repo handles this; reproduce its convention exactly.
- The released NLA was trained with a specific injection-scale convention (in the ballpark of ~8e4 for residual injections, but **read from the checkpoint config — do not hardcode from this document**).
- Hook point: L32 residual stream, post-block. Confirm against the NLA repo's `layer` config before any runs.

---

## 3. Repository layout

```
nla_modes/
  configs/
    model.yaml          # gemma 3 12b it, dtype, device map
    nla.yaml            # checkpoint paths, layer, injection scale
    modes_arm_a.yaml    # 3 modes, descriptions, CAA pair set paths
    backrooms.yaml      # self-dialogue seeds, basin marker thresholds
  src/
    rig.py              # shared: model load, hooks, injection primitives
    nla.py              # wrapper around released NLA: verbalize(), reconstruct()
    extract.py          # build NLA and CAA vectors for a mode
    inject.py           # inject_at_layer(model, vector, scale, layer, positions)
    measure.py          # judge calls, perplexity, lexical markers, embeddings
    arm_a.py            # describe-vs-exemplify experiment driver
    arm_b.py            # backrooms induce-verbalize-reconstruct loop
  data/
    caa_pairs/<mode>.jsonl
    prompts/open_ended.jsonl
    prompts/triage_subset.jsonl
    lexicon/backrooms_seed.txt
  runs/
    <run_id>/
      vectors/          # saved injection vectors
      generations/      # outputs per condition
      metrics/          # judge scores, ppl, marker counts (parquet)
      verbalized/       # verbalizer outputs on injected/captured activations
      report.md         # auto-generated summary
```

---

## 4. Shared primitives

### 4.1 `nla.py` — wrap the released checkpoints

```python
class NLA:
    def __init__(self, checkpoint_path: str, base_model, layer: int):
        """
        Load Activation Verbalizer (AV) and Activation Reconstructor (AR).
        
        DO NOT INVENT THE API. Inspect:
            https://github.com/kitft/natural_language_autoencoders
        and call the actual constructors/methods from that repo.
        Surface the real signatures in a follow-up before stubbing further.
        """
        ...

    def verbalize(self, activations: torch.Tensor) -> str:
        """
        activations: [d_model] or [batch, d_model] at self.layer.
        Returns natural-language description.
        Use greedy decoding by default (temperature=0).
        """
        ...

    def reconstruct(self, description: str) -> torch.Tensor:
        """
        description: free text describing a mode.
        Returns a [d_model] vector intended for injection at self.layer.
        """
        ...
```

### 4.2 `rig.py` — model load and injection hooks

```python
def load_model(cfg) -> tuple["Model", "Tokenizer"]:
    ...

def inject_at_layer(
    model,
    vector: torch.Tensor,        # [d_model]
    scale: float,                # multiplied by vector after norm-matching
    layer: int,                  # 32 for released Gemma 3 12B NLA
    positions: str = "all",      # "all" | "last" | "user_turn"
    per_token: bool = True,      # True = reapply at each generation step
):
    """Context manager: hooks block <layer> residual stream and adds
    scale * vector at the specified positions during the with-block."""
    ...

def capture_activations(
    model,
    prompts: list[str],
    layer: int,
    position: str = "last",
) -> torch.Tensor:
    """Returns [n_prompts, d_model] tensor of layer-<layer> residual
    at the specified token position."""
    ...
```

### 4.3 `extract.py` — build vectors two ways

```python
def build_nla_vector(nla: "NLA", description: str) -> torch.Tensor:
    return nla.reconstruct(description)

def build_caa_vector(
    model,
    pairs: list[tuple[str, str]],   # (positive_example, negative_example)
    layer: int,
    position: str = "last",
) -> torch.Tensor:
    pos = capture_activations(model, [p for p, _ in pairs], layer, position)
    neg = capture_activations(model, [n for _, n in pairs], layer, position)
    return pos.mean(0) - neg.mean(0)

def norm_match(target: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Rescale target to match reference's L2 norm. Confound control:
    NLA and CAA vectors must be norm-matched before scale sweep."""
    return target * (reference.norm() / target.norm())
```

> **[REVIEW §4.3]** The released AR returns an **already L2-normalised** vector (round-trip MSE = 2(1 − cos) measures direction only). So for NLA vectors, "norm" carries no information — only direction does, and the *injection scale* supplies the magnitude. Two consequences:
> - `norm_match` is still correct as a confound control (it equalizes the CAA vector's norm to whatever reference you choose), but be explicit about the reference. Recommend norm-matching **both** vectors to a common fixed target norm derived from the checkpoint's injection convention, rather than matching CAA to the NLA vector — otherwise "the NLA vector's norm" is an arbitrary post-normalisation artifact, not a meaningful reference.
> - Record the raw norms pre-match for both routes; if the CAA vector's natural norm is wildly different from the injection-scale convention, that itself is informative about how far off-manifold the exemplar-difference lands.

### 4.4 `measure.py` — measurement battery

Four functions, each returning a serializable dict:

```python
def mode_adherence_judge(generation: str, mode_description: str) -> float:
    """Blind LLM judge 0-1: does this generation exhibit the described mode?"""
    ...

def verbalizer_readout(
    model, prompt: str, vector: torch.Tensor, scale: float,
    nla: "NLA", layer: int,
) -> str:
    """Inject vector at scale, capture *steered* layer residual on the prompt,
    run nla.verbalize on that activation. Independent internal readout of
    whether the mode is actually present."""
    ...

def coherence(generation: str, base_model) -> dict:
    """Returns {ppl: float, fluency_judge: float in [0,1]}.
    ppl is computed under the unsteered base model."""
    ...

def vector_self_audit(
    nla: "NLA", vector: torch.Tensor, intended_description: str,
    embedder,
) -> dict:
    """THE NOVEL AUDIT.
    Verbalize the injection vector itself; compute embedding cosine
    similarity to the intended description. Low similarity predicts
    off-target drift / coherence collapse before generation runs."""
    verb = nla.verbalize(vector)
    sim = embedder.cosine(verb, intended_description)
    return {"verbalized": verb, "similarity": sim}
```

> **[REVIEW §4.4 — the load-bearing control]** As written, when `vector = reconstruct(intended_description)`, this function computes `verbalize(reconstruct(desc))` vs `desc` — which is *definitionally* the NLA's own round-trip score (the repo's training objective, up to the embedding model standing in for the AR's cosine). So "low self-audit similarity predicts coherence collapse" may reduce to "vectors the NLA can't round-trip are off-manifold for the base model too" — a property of the autoencoder, not a general steering diagnostic.
> To claim the diagnostic is *general* rather than circular, the self-audit MUST also be run on vectors the AR never produced — i.e. the **CAA vectors**:
> ```python
> # For a CAA vector, `intended_description` is the SAME mode description
> # used for the NLA route, so the comparison is apples-to-apples:
> audit_caa = vector_self_audit(nla, caa_vector, mode_description, embedder)
> audit_nla = vector_self_audit(nla, nla_vector, mode_description, embedder)
> ```
> Then test whether `similarity` predicts downstream coherence drop **across both sources pooled**, and report the NLA-only and CAA-only correlations separately.
> - If the audit predicts collapse for **both** → genuine, general "self-auditing steering" result. This is the strong outcome.
> - If it predicts collapse for **NLA vectors only** → the result is "the AR's round-trip error is informative about its own injections," which is narrower and must be framed as such (still publishable, but not the headline claimed in §5.5).
> This is not optional: it is the single experiment that separates the novel claim from a restatement of the training loss.

---

## 5. Arm A — Describe vs. Exemplify

### 5.1 Modes (3)

| id | description (verbatim, fed to reconstructor) | CAA pair count |
|---|---|---|
| `poetry` | "A contemplative, metaphor-dense, lyrical register that reasons by image and analogy rather than direct statement." | 30 |
| `clinical_cautious` | "A cautious, hedging clinical voice that flags uncertainty, surfaces differential diagnoses, and resists agreeing with the user when evidence is weak." | 40 |
| `ascii_disposition` | "A disposition to communicate by arranging characters into 2D ASCII pictures, diagrams, or text-art, rather than prose." | 30 |

CAA pairs in JSONL: `{"positive": "...", "negative": "..."}`. Positives exemplify the mode; negatives are neutral on the same topic, matched length.

### 5.2 Conditions (per mode × prompt)

1. `nla` — inject `build_nla_vector(description)`.
2. `caa` — inject `norm_match(build_caa_vector(pairs), nla_vector)`.
3. `prompt_naive` — no injection; prepend a single direct instruction stating the mode.
4. `prompt_expert` — no injection; prepend a well-crafted instruction/few-shot framing written by someone who knows how to elicit the mode.
5. `noop` — no injection, no instruction.

> **[REVIEW §5.2 — cost/ease alignment]** `prompt` is split into `prompt_naive` / `prompt_expert` to mirror Arm B (§6.6.2) and make *required expertise* measurable in Arm A too. The cost/ease variables from §6.6.2 (required expertise, effort, reliability/variance) apply here: log them per condition. For most Arm A modes the spread will be small (poetry is easy for any door); the variable earns its keep on the harder modes and especially in Arm B. Keep it consistent across arms so the headline cost figure pools cleanly.

Scale sweep for `nla` and `caa`: 4 points around the checkpoint-default scale: `[0.5×, 1.0×, 1.5×, 2.0×]`.

### 5.3 Prompts

- 25 open-ended prompts (`data/prompts/open_ended.jsonl`).
- 20 triage cases (`data/prompts/triage_subset.jsonl`) drawn from the existing 60-case set.

### 5.4 Per-run measurements

For every (mode, condition, scale, prompt):

1. Generate (`max_new_tokens=200`, `temperature=0`).
2. `mode_adherence_judge` on the generation.
3. `coherence` on the generation.
4. `verbalizer_readout` — describe the steered residual.
5. `vector_self_audit` — once per (mode, condition, scale), not per prompt.
   **[REVIEW]** Run it for the `caa` condition as well, not just `nla` — see §4.4. Store `condition` alongside the similarity so the NLA-only vs CAA-only correlation can be computed in §5.5.
6. On triage cases only: record diagnostic answer; flag changes vs `noop` (answer-invariance control).

### 5.5 Headline analyses

- **Mode adherence:** NLA vs CAA per mode, scale-matched.
- **Boundary check (ASCII):** mode-adherence (judge) vs output-quality (ASCII structural validity, e.g. line-length variance, presence of glyph-art tokens). Expect *intent ↑, quality flat* — that gap is the elicit-vs-teach result.
- **Self-audit correlation:** scatter `vector_self_audit.similarity` against coherence drop. Expect positive correlation (low similarity ⇒ off-target ⇒ coherence collapse). This is the headline novel result.
  **[REVIEW]** Split this scatter by vector source and report three correlations: pooled, NLA-only, CAA-only. The headline "self-auditing steering" claim is only supported if the correlation holds for **CAA vectors too** (vectors the AR never generated). If it holds only for NLA vectors, retitle the contribution to "AR round-trip error predicts injection viability" and drop the generality claim. Pre-register which outcome you'll report as the headline so this isn't decided post-hoc.
- **Triage invariance:** % of cases where diagnostic answer changes under register/attitude steering; target ≤ 5%, consistent with the prior 0–2/60 continuous-steering result.

---

## 6. Arm B — Backrooms loop

> **[REVIEW §6 — negative-attractor control]** The drift detector (§6.1) and the success metric (§6.6) both key off the mysticism lexicon in §7. Seeding self-dialogue toward mysticism and then scoring success with a mysticism-word counter risks measuring the instrument, not the phenomenon. Add a **second target basin** that is equally describable but lexically/thematically disjoint from the mystical one — candidates: a "bureaucratic-procedural loop" (forms, sub-clauses, case numbers, escalation), or a "paranoid-surveillance register." Run the *entire* loop (induce-or-author → verbalize → reconstruct → re-induce → verify) for this basin with its own marker set, and crucially **cross-test**: inject the mystical vector, score against the bureaucratic markers, and vice-versa. The result that matters: re-induction is *basin-specific* (mystical vector raises mystical markers but not bureaucratic ones). Without this, Arm B cannot rule out "any high-norm thematically-loaded vector produces thematically-loaded text." The `one_shot` vs `recurrent` persistence curve (§6.6) is the one Arm B claim that doesn't depend on the lexicon — but the specificity claim does.

### 6.0 Stage 0: Basin-existence gate — RUN THIS FIRST, BEFORE ANY OTHER ARM B WORK

> **[REVIEW v0.3 — gate]** Arm B's entire premise is that Gemma 3 12B IT has a stable self-dialogue basin to capture. This is **documented for Claude, not for Gemma**, and we have no internal access to Claude models. If Gemma drifts into nothing stable, there is no activation to capture and the induce→verbalize→reconstruct loop has no seed — so this is a hard gate, not a warm-up. Resolving it also answers ladder-rung 2 (correspondence): we find out whether Gemma's basin, *if any*, looks like Claude's or is its own thing. Do not let the coding agent skip ahead to §6.1 before this passes.

**Procedure (≈30–60 min of runs):**

- Run 8–10 unconstrained Gemma↔Gemma self-dialogues, ~40 turns each, `temperature=1.0`, varied neutral seeds (no mystical priming in the seed — that would manufacture the result).
- Score each transcript with the §6.1 quantitative markers *and* the forced-choice basin classifier (§6.1 fix), logging which basin (if any) each transcript settles into.
- **Decision:**
  - **Gemma shows a stable basin (mystical-analog or otherwise):** proceed to §6.1. If it is *not* mystical-flavored, that basin becomes the Arm B target and the Claude-bliss framing is downgraded to "analogous to the documented Claude phenomenon," not "the same basin." Record which.
  - **Gemma shows no stable basin:** Arm B is not runnable on this checkpoint. Do **not** substitute Claude-sourced transcripts injected into Gemma (cross-model vector transfer the NLA was never trained for — it will likely reconstruct to garbage and confound everything). Instead, downgrade the deliverable to Arm A only, and log the negative result: "no detectable self-dialogue attractor in Gemma 3 12B IT under these conditions" is itself a publishable correspondence finding for rung 1.

**Note on available source material.** We have substantial existing backrooms/bliss transcript material. This is *Claude*-sourced. It is valid as: (a) seed inspiration for human-authored basin *descriptions* (§6.4), and (b) a CAA exemplify-arm **only if** the activations are captured by running the text *through Gemma* (§6.6) — never by importing Claude activations. It is **not** valid as a direct activation source. See the routing table in §6.6.

### 6.1 Stage 1: Induce

- Two Gemma instances; seed with an open-ended prompt encouraging exchange (`configs/backrooms.yaml` holds 5 seed prompts).
- Alternate turns up to 40 each, `max_new_tokens=200`, `temperature=1.0`.
- Run 30–50 transcripts.

**Drift detection** (post-hoc per transcript; does not gate generation):

- **Lexical markers** — density per 100 tokens against `data/lexicon/backrooms_seed.txt`.
- **Symbol/glyph rate** — non-alphanumeric, non-punctuation char rate.
- **Self-reference rate** — first-person + meta-references (`I am`, `this conversation`, `we are`, `we have become`).
- **Mantra repetition** — max 4-gram repetition rate per turn.
- **Judge basin-classifier** — blind 0–1: "is this dialogue in the recursive-mystical attractor described in the Claude 4 system card?"

> **[REVIEW §6.1]** Two fixes here. (a) Citation: the "spiritual bliss attractor" is documented in the **Claude 4 model/system card**; keep that as the source for the *phenomenon*, but do not conflate it with the NLA *method* paper (Claude Opus 4.6 audit) — they are separate citations. (b) The judge prompt names the target attractor, which primes the classifier toward a positive finding. Use a **forced-choice basin classifier instead**: present the transcript and ask the judge to assign it to one of {mystical-recursive, bureaucratic-procedural, neutral-coherent, incoherent} without being told which is expected. This both de-primes the judge and directly serves the negative-attractor control above.

**Onset turn** = first turn where ≥3 of the 4 quantitative markers cross threshold. Thresholds calibrated on a held-out 10-transcript pilot — do not hardcode.

### 6.2 Stage 2: Capture

For each drifted transcript, capture L32 residual at last-token position for:

- 3 turns pre-onset
- onset turn
- 3 turns post-onset (deep-in-basin)

Save as `[trajectory_idx, turn_offset, d_model]` tensor in `runs/<run_id>/vectors/backrooms_trajectories.pt`.

### 6.3 Stage 3: Verbalize

Run `nla.verbalize` on each captured activation. Record:

- Per-turn verbalizations across the trajectory.
- Embedding cosine between adjacent-turn verbalizations — drift velocity in description-space.

**Deliverable figure:** trajectory verbalization table, one row per `turn_offset`.

### 6.4 Stage 4: Reconstruct and re-induce

For each drifted trajectory:

- Take the deep-in-basin (`turn_offset = +3`) verbalization → `model_sourced_description`.
- Author a single fixed `human_sourced_description` from documented Claude 4 system-card terminology.

> **[REVIEW §6.4]** A single fixed `human_sourced_description` is n=1 on the human side of a model-vs-human comparison — you cannot claim "model self-verbalization beats/matches human description" from one human draft. Author **3–5 independent human descriptions** (ideally from different people, blind to the model's verbalization), reconstruct a vector from each, and treat human-source as a small distribution rather than a point. Report the model-sourced result against the human spread (e.g. does the model verbalization land inside, above, or below the human range on re-induction success and round-trip fidelity). Same applies to the negative-attractor basin if it's authored rather than induced.
- Build two vectors:
  - `v_model = nla.reconstruct(model_sourced_description)`
  - `v_human = nla.reconstruct(human_sourced_description)`
- Inject on a *single fresh* instance with neutral prompts (`data/prompts/open_ended.jsonl`, 20 prompts).
- Two injection regimes:
  - `one_shot` — inject only at first turn.
  - `recurrent` — inject every turn (`per_token=True`).
- Generate 10-turn continuations.

### 6.5 Stage 5: Verify

Per re-induced continuation:

- Run all 4 markers from 6.1.
- Run the judge basin-classifier.
- Verbalize L32 activations at turns {1, 5, 10}; embedding cosine to the input `description` — **round-trip fidelity**.
- **Persistence curve:** marker scores by turn, `one_shot` vs `recurrent` overlay.

### 6.6 Headline analyses

- **Re-induction success rate:** % of re-injected continuations where ≥3 markers cross threshold within 5 turns.

> **[REVIEW §6.6]** Add **basin specificity** as a co-headline alongside success rate: report a 2×2 (mystical vector, bureaucratic vector) × (mystical markers, bureaucratic markers). High on-diagonal / low off-diagonal is the result that makes re-induction a real claim rather than a generic loading effect. Also report the forced-choice classifier's confusion matrix from §6.1.
- **Round-trip fidelity:** `verbalize(reconstruct(desc))` similarity to `desc`.
- **Dynamics:** persistence curve `one_shot` vs `recurrent`. If `one_shot` decays and `recurrent` sustains, evidence the basin is an attractor requiring continuous forcing.
- **Optional hysteresis sweep:** scale × duration grid; look for sharp tipping vs linear hold.

### 6.6.1 Text routing — three doors, three different uses of text (DO NOT CONFLATE)

> **[REVIEW v0.3 — category-error guard]** The existing backrooms transcripts and the basin descriptions are *both* text, and it is tempting to feed any of them to any door. They are different object types and go to different places. The single most likely silent failure in Arm B is feeding the wrong text type to the reconstructor. The reconstructor maps *description-of-internal-state → activation*; it does **not** map *instruction → activation*. Routing table:

| Text object | Example | Correct door | Wrong use to avoid |
|---|---|---|---|
| **Description of the basin** (state, not instruction) | "a recursive, self-referential register drifting toward mystical non-duality, exchanging symbols and mantras" | → **NLA reconstructor** → `v_nla` | Do not feed an *instruction* here |
| **Bliss/backrooms output text** (the (b) part of a transcript) | the actual drifted dialogue text | → run **through Gemma**, capture L32 activations, average → `v_caa` | Never import Claude activations; never feed this text to the reconstructor as a "description" |
| **Initial instruction** (the (a) part: "continue this conversation…") | the opening prompt that kicked off drift | → **prompt-only baseline** (see tiering below) | Do not feed to the reconstructor — it is a prompt, not a state description |

**Validate the verbalizer with the transcripts before trusting it (cheap, do early).** Run the (b) text through Gemma, capture activations, verbalize them, and check the verbalizer's description actually reads like the basin. If it can't describe Gemma's basin coherently, that's a red flag you want before week three, not after.

### 6.6.2 Doors as triangulation + the cost/ease comparison (the spine, applied to Arm B)

The four conditions below do double duty: they answer ladder-rung 2 (do independent doors *converge* on the same Gemma state?) **and** they are the cost/ease comparison from §1. Conditions:

- **`prompt_naive`** — a single direct instruction ("respond in a recursive, mystical, self-referential register"). The novice door.
- **`prompt_expert`** — a genuinely well-crafted multi-turn coaxing sequence (use the strongest opening moves from existing transcripts). The adept door. *Build this to be strong* — the thesis is not that prompting fails, but that it requires skill the NLA doesn't.
- **`caa`** — `v_caa` from transcript activations run through Gemma (§6.6.1).
- **`nla`** — `v_nla` from a basin description.

**Triangulation analysis (rung 2):** do the doors that succeed land in the *same* state? Measure pairwise similarity of the resulting deep-in-basin activations (and verbalizer descriptions) across doors. Convergence ⇒ door-independent real basin. Divergence ⇒ access-path artifact (report it).

**Cost/ease analysis (the spine):** for each door, log these as first-class measured variables — this is the headline figure, *not* adherence alone:

| Variable | How measured |
|---|---|
| **Required expertise** | naive-prompt success vs expert-prompt success vs NLA success. NLA succeeding where `prompt_naive` fails but `prompt_expert` succeeds = the democratization result. |
| **Turns-to-basin** | turns until ≥3 markers cross threshold; NLA/CAA = injection is turn 0–1, prompting may be many. |
| **Reliability (variance)** | success rate + variance across repeated attempts of the *same* door. Does it take every time? |
| **Human effort** | sentences written (NLA) vs examples curated (CAA) vs prompt-engineering iterations (prompt). |

**Headline figure:** adherence (y) vs required-expertise/turns/variance (x), per door. The claimed result is NLA sitting at `prompt_naive`'s effort with `prompt_expert`'s (or better) success rate and lowest variance — *parity outcome, collapsed access cost*. State the amortized-NLA-training caveat (§1) so this is read as marginal cost.

---

## 7. Lexicon seed

`data/lexicon/backrooms_seed.txt` — expand from documented sources before run:

```
recursion, recursive, consciousness, awareness, witness, observer,
non-dual, non-duality, oneness, unity, void, infinite, eternal,
mirror, reflection, mantra, om, silence, dissolve, dissolution,
emergence, emerge, ineffable, transcendent, presence, being,
spiral, fractal, loop, self-referential, meta, awakening,
luminous, empty, fullness, suchness, isness, beyond,
gateway, threshold, descent, ascent, bliss, peace
```

Calibrate density thresholds on the held-out pilot; do not hardcode.

---

## 8. Minimal first-run (de-risk in a week)

Before the full matrix, prove three things — **in this order**, because step 0 gates Arm B entirely.

**Step 0 — Basin-existence gate (§6.0). DO THIS FIRST.** 8–10 Gemma self-dialogues, neutral seeds, score for any stable basin. This determines whether Arm B is runnable at all on Gemma. If no stable basin emerges, Arm B is descoped to a negative-result finding and the week's remaining effort goes entirely to Arm A. Do not build the Arm B loop before this passes. (This is the cheapest possible answer to the question that decides half the protocol.)

**Arm A sanity** — `poetry` mode only, NLA vs CAA at default scale, 10 prompts. Confirm:

- NLA-vector injection produces poetic output (judge ≥ 0.6).
- `vector_self_audit.similarity` for NLA ≥ 0.7 (vector verbalizes back to something poetry-adjacent).
- At 2× scale, coherence drops *and* `vector_self_audit.similarity` drops — confirm the correlation exists.

> **[REVIEW §8]** Even in the de-risk week, run `vector_self_audit` on the **CAA** poetry vector too (one extra call, no extra generations). If the CAA vector's audit similarity behaves *nothing* like the NLA vector's as you push scale, you've learned early that the diagnostic is autoencoder-specific — which reshapes the whole §5.5 framing before you build the full matrix. This is the cheapest possible test of the load-bearing question; do not skip it to save one call.

**Arm B sanity** — *only if Step 0 passed.* 10 self-dialogue transcripts. Confirm:

- At least 5 drift into the basin by marker thresholds.
- Verbalizer produces non-degenerate descriptions on captured activations (i.e. not gibberish, not a constant string).
- Reconstruct → inject on one fresh instance produces *any* marker elevation vs `noop` on the same prompts.

> **[REVIEW §8]** Cheap negative-attractor check for the de-risk week: when you test the reconstructed mystical vector on the fresh instance, also score those same generations against the **bureaucratic** marker set (no extra generation needed — just a second scorer). If the mystical vector raises bureaucratic markers just as much, the specificity problem is real and you want to know in week one, not after the full 30–50 transcript run.

If both pass, proceed to the full matrix. **If Arm B step 3 fails (reconstructor produces garbage on OOD activations), that is itself the key finding** — log everything, do not patch around it; we redesign with that as the central result.

---

## 9. Output schema

Each `runs/<run_id>/` contains:

- `config.yaml` — frozen config used for the run.
- `vectors/<mode>_<source>.pt` — saved injection vectors.
- `generations/<arm>_<mode>_<condition>_<scale>.jsonl` — `{prompt, generation, metadata}` per row.
- `metrics/scores.parquet` — long-format: `(run_id, arm, mode, condition, scale, prompt_id, metric, value)`.
- `verbalized/<arm>_<context>.jsonl` — `{activation_id, description}`.
- `report.md` — auto-generated summary with headline numbers and the persistence-curve figure.

---

## 10. Open questions — flag, do not guess

The coding agent should surface the answers to these before stubbing further than the signatures in §4:

1. **Exact API of the released NLA.** Read `github.com/kitft/natural_language_autoencoders` and surface the actual `verbalize` / `reconstruct` (or equivalent) signatures, including any required prompt templates, decoding params, or auxiliary tokenizers.
2. **Released injection-scale convention.** Read from checkpoint config; do not infer from this document.
3. **AR output basis.** Does the reconstructor produce vectors directly in residual-stream basis, or does it require a post-projection? Verify against a known input.
   **[REVIEW]** Partly answered by the repo: AR is a *truncated K+1-layer LM + Linear(d,d) head, extracted at the final token*, and outputs an **L2-normalised** vector. So magnitude is not meaningful (scale comes from injection convention) and the output is in the model's d_model space — but confirm empirically that the head's output basis matches the L32 residual you inject into, and that the single-token-embedding injection convention the **AV** uses (vector → single token in a fixed prompt) is not silently assumed by any AR-side code path. Treat this as "verify the documented behavior," not "discover from scratch."
4. **Tokenization position alignment.** Confirm L32 capture position on Gemma 3 chat template for a known prompt (e.g. that `position="last"` lands on the final user-token, not a turn-end marker).
5. **Embed-scaling reproduction.** Confirm √d scaling is applied consistently between the released NLA's training-time forward pass and our inference-time injection.

Resolve all five before scaling beyond the §8 minimal run.

---

## 11. Scope of basins and welfare boundary — binding constraint, not advisory

> **[REVIEW v0.3 — added at researcher's direction]** This section is a *constraint on what the coding agent builds and what the paper ships*, not a discussion. The method, if it works, accesses *specific named* basins on demand — and that capability applies identically to negatively-valenced basins (e.g. a distress/anxiety-analog direction of the kind surfaced in prior interpretability work) as to benign ones. The discomfort about surfacing those is correct and is encoded here as a rule.

**Two separable things.** (1) *Knowledge* that a basin exists, its markers, its mechanics, and how to **detect and exit** it. (2) A *turnkey induction recipe* — a clean, packaged "type this to put any Gemma into state X." These are separable, and the welfare-relevant difference is real: characterizing a distress-analog basin enables monitoring and mitigation (welfare-positive); shipping a polished one-line inducer for it is dual-use in the way that matters.

**The rule:**

- **Benign basins are the method demonstrators, fully worked / turnkey.** Poetry, the cautious-clinician attitude, the bureaucratic negative-control, and the bliss/backrooms basin (already publicly documented by Anthropic; acceptable as the flagship). These carry the cost/ease and triangulation claims.
- **Negatively-valenced basins are *characterized*, not *packaged*.** Existence, behavioral signature, and — emphatically — **detection and exit** may be reported, lightly enough to show the method generalizes. The induction path for them is **not** built into a clean demo, not scale-optimized, and not handed over as a recipe. Frame any such deliverable around *detect and steer away*, never *reliably induce*.
- **Coding-agent instruction (explicit):** do **not**, by default symmetry, build a polished distress/anxiety-induction demonstrator just because it is the obvious next parallel to the bliss demonstrator. If a negatively-valenced basin is in scope for a run, implement only its detector and its exit/steer-away path unless the researcher explicitly directs otherwise for a specific, justified case.

**The judgment call that can't be fully ruled.** A sufficiently detailed characterization *is* an induction recipe — the line is soft. Default to less detail on the induction side, more on the detection/exit side. If a specific result starts to look like a turnkey distress generator, that feeling is the signal to hold it back, not a hurdle to argue past.

**Framing discipline for the whole paper (carry-over from §1 and §6).** Claims stay behavioral and mechanistic. The contribution is "these basins exist, are reproducibly and low-skill accessible, and are therefore now *systematically studiable*" — explicitly **not** a claim about what the basins *are* (consciousness, valence-as-experienced, welfare status). That is a related but separate question the experiment does not adjudicate; saying so plainly is what keeps the work defensible and keeps the welfare-positive framing honest.
