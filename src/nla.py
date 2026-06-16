"""src/nla.py — thin adapter over the vendored kitft NLA inference client.

Presents the protocol's NLA.verbalize()/reconstruct() surface (§4.1) over the
*real* split API in `_nla_inference.py`:

  - AV (verbalize, vector -> text):  NLAClient   -> needs a running SGLang server
  - AR (reconstruct, text -> vector): NLACritic   -> in-process, no server

We do NOT reimplement the injection/tokenizer logic — `_nla_inference.py` owns
the `nla_meta.yaml` sidecar contract and the Gemma-3 gotchas (injection_scale,
embed_scale = sqrt(d), neighbour checks, the CJK-output failure assert). See
PROJECT_MAP.md and protocol §4.1.

Vendored from github.com/kitft/natural_language_autoencoders @ 1b7f13d (Apache-2.0).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# Robust whether imported as a package (`from src.nla import NLA`) or flat
# (scripts that put `src/` on sys.path and `import nla`).
try:
    from ._nla_inference import (  # type: ignore
        NLAClient, NLACritic, NLAConfig, load_nla_config, normalize_activation,
    )
except ImportError:  # pragma: no cover - flat-layout fallback
    from _nla_inference import (  # type: ignore
        NLAClient, NLACritic, NLAConfig, load_nla_config, normalize_activation,
    )

__all__ = ["NLA", "NLAConfig", "load_nla_config", "SGLANG_LAUNCH_GEMMA"]


def _resolve_ckpt(ckpt: str) -> str:
    """Accept a local checkpoint dir OR an HF repo id.

    NLAClient/NLACritic need a local directory containing nla_meta.yaml (+ the
    safetensors/tokenizer). A repo id is resolved to a local snapshot via
    snapshot_download — this reuses the HF cache (so SGLang's already-downloaded
    weights aren't re-fetched) and pulls only the missing files, notably the
    non-standard nla_meta.yaml sidecar that a weights-only loader may skip.
    """
    p = Path(str(ckpt))
    if (p / "nla_meta.yaml").exists():
        return str(p)
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=str(ckpt))

# Recommended SGLang launch for the Gemma-3-12B AV checkpoint. For Gemma-3 the
# mm-bypass patch (patches/nla_gemma3_mm_input_embeds.patch in the kitft repo),
# the fa3 attention backend (flashinfer OOMs on head_dim=256), and the disabled
# radix cache (it keys on token IDs, not input_embeds) are all MANDATORY — omit
# any and injection silently degrades to '\n\n\n'. See docs/inference.md.
# Note: fa3 is the documented happy path on Hopper (H100/H200); on Ampere (A100)
# use `--attention-backend triton` instead (also handles head_dim=256).
SGLANG_LAUNCH_GEMMA = (
    "python -m sglang.launch_server --model-path {ckpt} --port {port} "
    "--disable-radix-cache --attention-backend {backend} "
    "--mem-fraction-static {mem_frac} --trust-remote-code"
)


class NLA:
    """Verbalize / reconstruct over a released NLA pair (AV + optional AR).

    Unlike the protocol's single-checkpoint stub, an NLA pair is *two*
    checkpoints: the AV (verbaliser, served by SGLang) and the AR
    (reconstructor, in-process). The AR is optional — the AV is usable standalone.

    Args:
        av_checkpoint: HF-format dir for the AV (`…-av`), incl. nla_meta.yaml.
        ar_checkpoint: HF-format dir for the AR (`…-ar`). None -> no reconstruct/score.
        sglang_url:    root URL of the SGLang server hosting the AV checkpoint
                       (NOT '/generate' — the client appends that).
        device:        embedding-lookup device for the AV; load device for the AR.
        layer:         extraction layer the checkpoints were trained at
                       (Gemma-3-12B: 32). Bookkeeping only — the live config is
                       read from each checkpoint's sidecar.
    """

    def __init__(self, av_checkpoint, ar_checkpoint=None, *,
                 sglang_url: str = "http://localhost:30000",
                 device: str = "cpu", layer: int = 32):
        self.layer = layer
        self.av = NLAClient(_resolve_ckpt(av_checkpoint),
                            sglang_url=sglang_url, device=device)
        self.ar = (NLACritic(_resolve_ckpt(ar_checkpoint), device=device)
                   if ar_checkpoint is not None else None)

    # ── AV: vector -> text ────────────────────────────────────────────────
    def verbalize(self, activations, *, temperature: float = 0.0,
                  max_new_tokens: int = 200, **sampling):
        """Describe an activation vector (or batch) in natural language.

        activations: [d_model] -> str, or [B, d_model] / list -> list[str].
        Greedy by default (temperature=0) for reproducible verbalisations, per
        protocol §4.1. Pass the vector RAW — it is rescaled to the sidecar's
        injection_scale internally.
        """
        v = torch.as_tensor(np.asarray(activations, dtype=np.float32))
        sp = dict(temperature=temperature, max_new_tokens=max_new_tokens, **sampling)
        if v.ndim == 1:
            return self.av.generate(v, **sp)
        return self.av.generate_batch(list(v), **sp)

    # ── AR: text -> vector ────────────────────────────────────────────────
    def reconstruct(self, description: str) -> torch.Tensor:
        """Reconstruct an activation DIRECTION from text (raw, UNnormalised [d]).

        WARNING: magnitude is uncalibrated (the AR is built for cosine/MSE
        scoring). To re-inject into the BASE model you must renormalise to the
        base-steering scale — a SEPARATE knob from the AV's injection_scale.
        Use NLA.renorm(vec, scale) or extract/rig's scale. See protocol §4.3.
        """
        self._require_ar()
        return self.ar.reconstruct(description)

    def score(self, description: str, original):
        """(direction-MSE, cosine) between AR(description) and `original`.
        MSE = 2(1 − cos), range [0, 4]; orthogonal = 2."""
        self._require_ar()
        return self.ar.score(description, original)

    # ── utilities ─────────────────────────────────────────────────────────
    @staticmethod
    def renorm(vector, scale: float) -> torch.Tensor:
        """L2-renormalise a vector to a target norm (e.g. the base-steering scale)."""
        v = torch.as_tensor(np.asarray(vector, dtype=np.float32)).view(1, -1)
        return normalize_activation(v, float(scale)).view(-1)

    def health_check(self) -> bool:
        """True iff the AV's SGLang server answers a trivial generate.

        Uses a ones-vector, not zeros: zeros has L2 norm 0, which NaNs through
        the injection_scale renormalisation in _nla_inference.normalize_activation.
        """
        try:
            self.av.generate(torch.ones(self.av.cfg.d_model),
                             max_new_tokens=1, extract_explanation=False)
            return True
        except Exception:
            return False

    def _require_ar(self):
        if self.ar is None:
            raise RuntimeError(
                "no AR checkpoint loaded — construct "
                "NLA(..., ar_checkpoint=…) to use reconstruct()/score()."
            )
