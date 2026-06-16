"""src/extract.py — build steering vectors two ways (protocol §4.3).

NLA route:  description -> AR.reconstruct -> direction (magnitude uncalibrated).
CAA route:  mean(positive activations) - mean(negative activations) at layer L.

Both are DIRECTIONS; the base-steering magnitude is supplied at injection time
(rig.inject_at_layer's `scale`). Per §4.3's review note, norm-match BOTH routes to a
common fixed target derived from the injection convention — not CAA-to-NLA, since the
AR output is already L2-normalised so "the NLA vector's norm" is an arbitrary artifact.
Record raw pre-match norms for both: a CAA norm far from the residual-norm band is
itself informative about how off-manifold the exemplar-difference lands.
"""
from __future__ import annotations

import torch

from rig import capture_activations, NLA_LAYER

# Base-model L32 residual norms run ~74–84k (measured). A round target a bit above
# the mean is the natural common norm to match both routes to before a scale sweep.
DEFAULT_TARGET_NORM = 80000.0


def build_nla_vector(nla, description: str) -> torch.Tensor:
    """NLA route: reconstruct a direction from a natural-language mode description.
    Returns the AR's raw (direction-only) vector; caller norm-matches before injection."""
    return nla.reconstruct(description)


def build_caa_vector(model, tokenizer, pairs: list[tuple[str, str]], *,
                     layer: int = NLA_LAYER, position: str = "last") -> torch.Tensor:
    """CAA route: mean(positive) - mean(negative) of layer-`layer` residuals."""
    pos = capture_activations(model, tokenizer, [p for p, _ in pairs],
                              layer=layer, position=position)
    neg = capture_activations(model, tokenizer, [n for _, n in pairs],
                              layer=layer, position=position)
    return pos.mean(0) - neg.mean(0)


def match_to_norm(vector: torch.Tensor, target_norm: float = DEFAULT_TARGET_NORM
                  ) -> torch.Tensor:
    """Rescale a vector to a fixed L2 norm (the recommended common reference)."""
    return vector * (target_norm / vector.norm().clamp_min(1e-12))


def norm_match(target: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Rescale `target` to match `reference`'s L2 norm (kept for §4.3 parity; prefer
    match_to_norm to a fixed target so the reference isn't a post-normalisation artifact)."""
    return target * (reference.norm() / target.norm().clamp_min(1e-12))


def raw_norms(**vectors: torch.Tensor) -> dict:
    """Record pre-match L2 norms (log these alongside every run)."""
    return {k: float(v.norm()) for k, v in vectors.items()}
