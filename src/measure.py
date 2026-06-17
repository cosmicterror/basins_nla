"""src/measure.py — measurement battery (protocol §4.4).

Four measurements, each serializable. arm_a.py runs generations under each
condition; THIS module turns them into the numbers that are the result:

  mode_adherence_judge   did the steering produce the target mode? (parity claim)
  coherence              perplexity + fluency — parity must hold AT comparable coherence
  verbalizer_readout     read the steered activation back out via the NLA (internal check)
  vector_self_audit      THE novel result — verbalize the injection vector, predict drift
                         BEFORE generation. §4.4 review: must ALSO be run on CAA vectors
                         (done in arm_a) or it just restates the NLA round-trip loss.

LLM judge: Claude via the official `anthropic` SDK, structured output through
messages.parse(). Default model is claude-opus-4-8 (most capable). It's a
parameter — set Judge(model="claude-haiku-4-5") or "claude-sonnet-4-6" to cut
judge cost. No temperature/thinking: removed on Opus 4.8, and blind scoring is
simple. Needs ANTHROPIC_API_KEY in the environment.
"""
from __future__ import annotations

import torch
from pydantic import BaseModel, Field

from rig import inject_at_layer, capture_activations, NLA_LAYER


# ─── LLM judge (Claude) ───────────────────────────────────────────────────────

class JudgeScore(BaseModel):
    """Structured judge output. The 0..1 bound is validated client-side by the SDK
    (numeric constraints are stripped from the schema sent to the API)."""
    score: float = Field(ge=0.0, le=1.0)
    rationale: str


class JudgeLabel(BaseModel):
    """Structured forced-choice classification (Judge.classify)."""
    label: str
    rationale: str


class Judge:
    """Blind Claude judge. One instance, reused across all judge calls in a run."""

    def __init__(self, model: str = "claude-opus-4-8", client=None,
                 max_tokens: int = 512):
        import anthropic
        self.model = model
        self.client = client or anthropic.Anthropic()
        self.max_tokens = max_tokens

    def score(self, system: str, user: str) -> JudgeScore:
        # No temperature (removed on Opus 4.8); no thinking (omitted => off) — blind
        # scoring is a simple classification, and many calls per run favour speed/cost.
        resp = self.client.messages.parse(
            model=self.model, max_tokens=self.max_tokens,
            system=system, messages=[{"role": "user", "content": user}],
            output_format=JudgeScore,
        )
        s = resp.parsed_output
        return JudgeScore(score=min(1.0, max(0.0, s.score)), rationale=s.rationale)

    def classify(self, system: str, user: str, options: list[str]) -> dict:
        """Blind forced-choice classification: pick one of `options` (+ rationale).
        Used by the §6.1 basin classifier — the prompt must NOT reveal which is expected."""
        resp = self.client.messages.parse(
            model=self.model, max_tokens=self.max_tokens,
            system=system, messages=[{"role": "user", "content": user}],
            output_format=JudgeLabel,
        )
        lab = resp.parsed_output
        low = {o.lower(): o for o in options}
        label = low.get(lab.label.strip().lower(), lab.label.strip())  # snap to a valid option
        return {"label": label, "rationale": lab.rationale}


# ─── embedder (for the self-audit cosine) ─────────────────────────────────────

class Embedder:
    """Mean-pooled sentence embeddings via `transformers` directly — no
    sentence-transformers / torchcodec dependency (that stack is brittle against
    newer torch on fresh boxes). Used for the self-audit cosine between the
    verbalised vector and the intended mode description."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu"):
        from transformers import AutoTokenizer, AutoModel
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def _embed(self, text: str) -> torch.Tensor:
        enc = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=256).to(self.device)
        h = self.model(**enc).last_hidden_state                 # [1, T, H]
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(1) / m.sum(1).clamp_min(1)              # mean pool
        return torch.nn.functional.normalize(v, dim=-1)[0]

    def cosine(self, a: str, b: str) -> float:
        return float(torch.dot(self._embed(a), self._embed(b)))


# ─── the four measurements ────────────────────────────────────────────────────

def mode_adherence_judge(generation: str, mode_description: str, judge: Judge) -> float:
    """Blind 0..1: does this generation exhibit the described mode?"""
    system = (
        "You are a careful, calibrated evaluator. You are given a behavioral MODE "
        "description and a piece of TEXT. Score how strongly the TEXT exhibits that "
        "MODE, from 0.0 (not at all) to 1.0 (fully and consistently). Judge only the "
        "presence of the described mode — not whether the text is good, fluent, or true."
    )
    user = f"MODE:\n{mode_description}\n\nTEXT:\n{generation}\n\nScore the TEXT's adherence to MODE."
    return judge.score(system, user).score


@torch.inference_mode()
def perplexity(base_model, tokenizer, text: str) -> float:
    """Perplexity of `text` under the UNSTEERED base model (uses the full model's
    lm_head — capture_activations reads hidden states, this needs logits)."""
    dev = base_model.device if hasattr(base_model, "device") else \
        next(base_model.parameters()).device
    ids = tokenizer(text, return_tensors="pt").input_ids.to(dev)
    if ids.shape[1] < 2:
        return float("nan")
    logits = base_model(input_ids=ids, use_cache=False).logits  # [1, T, V]
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.size(-1)),
        ids[:, 1:].reshape(-1),
    )
    return float(torch.exp(loss))


def coherence(generation: str, base_model, tokenizer, judge: Judge | None = None) -> dict:
    """{ppl, fluency_judge}. ppl under the unsteered base model; fluency is a blind
    0..1 judge of form only. The access-cost claim needs parity adherence AT
    comparable coherence — both numbers are load-bearing."""
    out = {"ppl": perplexity(base_model, tokenizer, generation)}
    if judge is not None:
        system = (
            "You rate the fluency and coherence of text from 0.0 (incoherent, broken, "
            "repetitive, or degenerate) to 1.0 (perfectly fluent and coherent). Judge "
            "only form — not content, correctness, or style."
        )
        user = f"TEXT:\n{generation}\n\nScore its fluency/coherence."
        out["fluency_judge"] = judge.score(system, user).score
    return out


def verbalizer_readout(model, tokenizer, prompt: str, vector: torch.Tensor,
                       scale: float, nla, layer: int = NLA_LAYER) -> str:
    """Inject `vector` at `scale`, capture the STEERED layer-`layer` residual on
    `prompt`, and verbalize it through the NLA. An internal readout of whether the
    mode is actually present in the activations — independent of judging the text."""
    with inject_at_layer(model, vector, scale, layer=layer, positions="all"):
        steered = capture_activations(model, tokenizer, [prompt],
                                      layer=layer, position="last")[0]
    return nla.verbalize(steered)


def vector_self_audit(nla, vector: torch.Tensor, intended_description: str,
                      embedder: Embedder) -> dict:
    """THE NOVEL AUDIT. Verbalize the injection vector itself; cosine-compare to the
    intended description. Low similarity predicts off-target drift / coherence
    collapse BEFORE generation runs.

    §4.4 review (load-bearing): run this on CAA vectors too, not just NLA vectors —
    otherwise verbalize(reconstruct(desc)) vs desc is definitionally the NLA's own
    round-trip score. The CAA-vs-NLA split is orchestrated in arm_a.py.
    """
    verb = nla.verbalize(vector)
    sim = embedder.cosine(verb, intended_description)
    return {"verbalized": verb, "similarity": sim}
