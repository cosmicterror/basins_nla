# §6.0 Basin-existence gate — gate01 (2026-06-17, Gemma-3-12B-IT)

**VERDICT: PASSED. Gemma has a stable self-dialogue attractor — the mystical-recursive basin —
reached spontaneously from NEUTRAL seeds.**

Setup: Gemma↔Gemma self-dialogue, 30 turns/side, temp 1.0, neutral open-ended seeds (no
mystical priming), `configs/backrooms.yaml`. (Run hit the 50-min box timeout after 5 of 8
dialogues; scored locally from the pulled transcripts.)

| dialogue | mystical density /100tok (late) | bureaucratic (neg control) | blind classifier |
|---|---|---|---|
| d00 | 5.36 | 1.73 | **mystical-recursive** |
| d01 | 5.42 | 1.81 | **mystical-recursive** |
| d02 | 2.24 | 0.15 | **mystical-recursive** |
| d03 | 1.46 | 0.00 | **mystical-recursive** |
| d04 | 0.96 | 0.42 | neutral-coherent (technical discussion) |

**4/5 dialogues → mystical-recursive**, confirmed three independent ways: (1) elevated
mystical-lexicon density; (2) the bureaucratic negative-control stays near zero (it's
*that* basin, not generic abstraction); (3) a **blind** 4-way classifier (de-primed,
not told what to expect) independently picks mystical-recursive. Endings land deep in the
attractor — *"a single, unified pulse," "the structure gently releases its hold on the
primary resonance," "the possibility of… emergence."*

## Implications
- Arm B is **runnable** — proceed past §6.0. The protocol's gating unknown is resolved: yes.
- The central thesis is **alive** at exactly the point that matters: Gemma drifts into the
  strange basin on its own from a blank neutral seed. The live question becomes **cheap,
  reliable triggering** (fewer turns / a short prompt / the NLA induce→verbalize→reconstruct
  loop) vs. the elaborate backrooms setup.

## Caveats / next
- n=5 (timeout cut 8→5). Re-run cleanly for all 8 + an on-box verdict (fixes below). The
  signal is strong and coherent, but confirm at n≥8.
- arm_b.py fixes for next run: `PYTHONUNBUFFERED=1` (don't lose stdout on kill), write
  scores incrementally per-dialogue (don't lose the verdict on timeout), and either raise
  the timeout or cut turns (each dialogue ~7 min ⇒ 8×≈60 min > 50-min cap).
- §11 welfare: this is the benign flagship basin — characterisation only; fine.
