# Third-party notices

## `src/_nla_inference.py`

Vendored verbatim from the kitft Natural Language Autoencoders repo so this
repo stays standalone (no dependency on a separate package or the sibling repo).

- **Source:** https://github.com/kitft/natural_language_autoencoders
- **File:** `nla_inference.py` (unmodified)
- **Commit:** `1b7f13d9d8a37075cd2e5d1604eca57820216ed5`
- **License:** Apache-2.0 — https://github.com/kitft/natural_language_autoencoders/blob/main/LICENSE

`src/nla.py` is our own thin adapter over it (`NLAClient` / `NLACritic` →
the protocol's `NLA.verbalize()/reconstruct()`).

**To update:** re-copy `nla_inference.py` from upstream into
`src/_nla_inference.py` and bump the commit hash above.

Accompanying paper:

> Fraser-Taliente, Kantamneni, Ong et al., "Natural Language Autoencoders
> Produce Unsupervised Explanations of LLM Activations", Transformer Circuits,
> 2026. https://transformer-circuits.pub/2026/nla/index.html
