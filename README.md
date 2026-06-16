# basins_nla — Arm B (Backrooms loop)

The **basins / self-dialogue** stream of the NLA mode-steering project (the esoteric arm):
induce → verbalize → reconstruct → re-induce → verify a self-dialogue basin in
`google/gemma-3-12b-it`, using the released NLA checkpoints at layer 32.

Two binding constraints govern this repo and not the methods stream:
- **§6.0 existence gate — RUN FIRST.** Confirm Gemma has a stable self-dialogue basin at all
  before any §6.1+ work. If not, Arm B descopes to a negative result. Never inject Claude-sourced
  activations into Gemma.
- **§11 welfare boundary — binding.** See **[`WELFARE.md`](WELFARE.md)**. Benign basins are
  turnkey demonstrators; negatively-valenced basins are *characterized* (detect + exit), never
  packaged as an induction recipe.

This is **one of two standalone repos.** The methods / steering work (Arm A — the access-cost
claim across poetry / triage / ascii modes) lives in the sibling repo **`nla_modes`**, with its
own independent git history. For the whole picture — both repos, how they relate, the shared-core
copy policy, all constraints — read **[`PROJECT_MAP.md`](PROJECT_MAP.md)**. The authoritative spec
is [`protocol/nla_modes_protocol.md`](protocol/nla_modes_protocol.md).

Shared `src/` primitives (`rig`, `nla`, `extract`, `inject`, `measure`) are **duplicated by copy**
from/to `nla_modes`, not shared as a package — each repo stands alone.

**Status:** scaffolded; no rig code yet. Next: clone + read
`kitft/natural_language_autoencoders`, then run the §6.0 existence gate before anything else.
