"""src/rig.py — base-model load, injection hooks, activation capture (protocol §4.2).

This is the BASE-model rig: it loads google/gemma-3-12b-it, captures L32 residual
activations, and steers generation by adding a vector to the residual stream. It is
independent of the NLA checkpoints (those live in nla.py). extract.py and measure.py
compose rig + nla.

Two scales, do not conflate (see PROJECT_MAP.md / nla.py):
  - the AV's injection_scale (80000) is for the verbaliser's input slot — NOT used here;
  - the BASE-steering scale is `scale` below, the magnitude we add to Gemma's own
    residual stream. It is a separate, calibrated knob (§4.3/§5 scale sweep).

Layer convention: the released NLA extracts at hidden_states[32] (= the residual after
32 transformer blocks = the output of decoder layer index 31). capture_activations()
reads hidden_states[layer] (matches NLA training — validated: real medical activation
L2≈84k). inject_at_layer() adds to the output of decoder layer index `layer-1`, the
same locus. The off-by-one is asserted-documented and should be confirmed on the first
real run via measure.verbalizer_readout (a correctly-injected mode vector verbalises
back to the mode).
"""
from __future__ import annotations

import contextlib
from typing import Iterable

import torch

MODEL_ID = "google/gemma-3-12b-it"
NLA_LAYER = 32  # Gemma-3-12B-IT extraction layer (of 48)


# ─── model load ──────────────────────────────────────────────────────────────

def load_model(model_id: str = MODEL_ID, *, dtype: torch.dtype = torch.bfloat16,
               device_map: str = "cuda"):
    """Load the base model + tokenizer. Returns (model, tokenizer).

    gemma-3-12b-it is a multimodal wrapper (Gemma3ForConditionalGeneration); we keep
    the full model (so .generate works) and reach its text stack via _language_model().
    Needs `accelerate` for device_map= loading.
    """
    from transformers import AutoTokenizer
    try:
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype, device_map=device_map)
    except Exception:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map=device_map)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_id)
    return model, tok


def _language_model(model):
    """The text decoder stack (Gemma3TextModel), robust to the mm wrapper layout."""
    for path in (("model", "language_model"), ("language_model",), ("model",)):
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if hasattr(obj, "layers"):
                return obj
        except AttributeError:
            continue
    raise AttributeError(
        f"could not locate the decoder stack (.layers) on {type(model).__name__}; "
        f"inspect the model and extend _language_model().")


def _decoder_layers(model):
    return _language_model(model).layers


# ─── injection ───────────────────────────────────────────────────────────────

@contextlib.contextmanager
def inject_at_layer(model, vector: torch.Tensor, scale: float, *,
                    layer: int = NLA_LAYER, positions: str = "all",
                    per_token: bool = True):
    """Context manager: add `scale * vector` to the layer-`layer` residual stream
    during the with-block (steered capture or generation).

    vector:    [d_model] direction (norm-matched by the caller — see extract.norm_match).
    scale:     base-steering magnitude (NOT the AV injection_scale). Multiplies vector.
    layer:     NLA extraction layer (32). Injects at decoder layer index `layer-1`,
               whose output == hidden_states[layer] (the NLA capture locus).
    positions: "all" (every token in the forward) | "last" (final position only).
               "user_turn" is not yet implemented (falls back to "all" with a warning).
    per_token: a forward hook fires on every forward pass, so injection naturally
               persists across generated tokens; the flag is kept for signature
               parity and reserved for a future prefill-only mode.
    """
    layers = _decoder_layers(model)
    idx = layer - 1
    assert 0 <= idx < len(layers), f"layer {layer} out of range for {len(layers)} blocks"
    if positions == "user_turn":
        import warnings
        warnings.warn("positions='user_turn' not implemented; using 'all'")
        positions = "all"

    v = vector.detach().reshape(-1)

    def hook(_module, _inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        add = (scale * v).to(hs.device, hs.dtype)
        if positions == "last":
            hs = hs.clone()
            hs[:, -1, :] = hs[:, -1, :] + add
        else:  # "all"
            hs = hs + add
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs

    handle = layers[idx].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# ─── capture ─────────────────────────────────────────────────────────────────

@torch.inference_mode()
def capture_activations(model, tokenizer, prompts: Iterable[str], *,
                        layer: int = NLA_LAYER, position: str = "last",
                        batch_size: int = 8, apply_chat_template: bool = False
                        ) -> torch.Tensor:
    """Return [n_prompts, d_model] of layer-`layer` residual at `position`.

    Reads hidden_states[layer] from the text stack — the exact locus the NLA was
    trained on (validated: real Gemma L32 activation L2≈84k). `position`:
      "last" -> last real (non-pad) token; "mean" -> mean over real tokens.
    Set apply_chat_template=True to wrap prompts as a user turn; default False
    extracts from raw text (matches the NLA's fineweb-style training distribution).
    """
    lm = _language_model(model)
    dev = next(lm.parameters()).device
    prompts = list(prompts)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        if apply_chat_template:
            texts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False,
                add_generation_prompt=True) for p in chunk]
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(dev)
        else:
            enc = tokenizer(chunk, return_tensors="pt", padding=True).to(dev)
        hs = lm(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                output_hidden_states=True, use_cache=False).hidden_states[layer]  # [b,t,d]
        mask = enc["attention_mask"]  # [b,t]
        if position == "mean":
            m = mask.unsqueeze(-1).to(hs.dtype)
            vec = (hs * m).sum(1) / m.sum(1).clamp_min(1)
        else:  # "last" real token
            last = mask.sum(1) - 1  # index of last real token per row
            vec = hs[torch.arange(hs.size(0), device=dev), last]
        out.append(vec.float().cpu())
    return torch.cat(out, 0)


# ─── convenience generation ──────────────────────────────────────────────────

@torch.inference_mode()
def generate_text(model, tokenizer, prompt: str, *, max_new_tokens: int = 256,
                  temperature: float = 1.0, **gen) -> str:
    """Chat-format a prompt, generate, return only the new completion text.
    Wrap a call in `with inject_at_layer(...):` to get a steered generation."""
    dev = next(_language_model(model).parameters()).device
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], return_tensors="pt",
        add_generation_prompt=True).to(dev)
    do_sample = temperature and temperature > 0
    full = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=do_sample,
                          temperature=(temperature if do_sample else None), **gen)
    return tokenizer.decode(full[0, ids.shape[1]:], skip_special_tokens=True)
