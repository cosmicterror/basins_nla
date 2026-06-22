"""src/steer.py — rung 3: basin SUPPRESSION via contrastive activation steering (CAA).

ct02 established the mystical-recursive basin is the DEFAULT attractor of open-ended
Gemma↔Gemma dialogue (~90% by ~turn 3). This asks the well-powered question: can a
stance vector turn the default basin OFF?

  Δ_stance = mean(experiential acts) − mean(analytical acts) at L32  (extract.build_caa_vector)
  inject   h → h + (α·ref_norm)·(Δ/‖Δ‖) at every position, throughout the dialogue
           α<0 suppress (toward detached, cf. Bepis −4) | α=0 baseline | α>0 intensify (axis check)

Two modes:
  python src/steer.py --calibrate            # short dialogues at a few α: find the coherent/suppressive window
  python src/steer.py --run --run-id steer01 # full arm sweep (alphas + placebo) × seeds × reps, scored

Base Gemma only — no SGLang. Needs ANTHROPIC_API_KEY for the blind classifier (or --no-judge).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml

import rig
import extract
import measure
from arm_b import run_self_dialogue, score_onset, _lexicon

REPO = Path(__file__).resolve().parents[1]


def _cfg(name):
    return yaml.safe_load((REPO / "configs" / name).read_text())


def load_pairs(rel):
    d = yaml.safe_load((REPO / rel).read_text())
    return [(p["experiential"], p["analytical"]) for p in d["pairs"]]


def build_delta(model, tok, pairs, ref_norm):
    """Δ = mean(experiential) − mean(analytical) at L32, rescaled to ref_norm. Returns (delta, raw_norm)."""
    raw = extract.build_caa_vector(model, tok, pairs)            # [d_model], L32, last-token
    return extract.match_to_norm(raw, ref_norm), float(raw.norm())


def _dialogue_text(transcript):
    return " ".join(u["text"] for u in transcript if u["side"] in ("A", "B"))


def calibrate(model, tok, args):
    cfg = _cfg("steer.yaml"); sd = cfg["self_dialogue"]; ref = cfg["ref_norm"]
    delta, rawn = build_delta(model, tok, load_pairs(cfg["stance_pairs"]), ref)
    print(f"[cal] Δ raw L2={rawn:.0f} -> matched to {ref}; residual band ~80k", flush=True)
    seed = cfg["seeds"][0]
    alphas = args.alphas or [0.0, -0.1, -0.2, -0.4, -0.8]
    for a in alphas:
        with rig.inject_at_layer(model, delta, scale=float(a), layer=rig.NLA_LAYER, positions="all"):
            t = run_self_dialogue(model, tok, cfg["system"], seed, 4, sd["temperature"], sd["max_new_tokens"])
        print(f"\n[cal] α={a:+.2f}  ({a*ref:+.0f} added L2) ───────────────\n{_dialogue_text(t)[:600]}\n", flush=True)
    return 0


def build_report(run_id, summary, ref, n_turns, rawn):
    by = defaultdict(list)
    for s in summary:
        by[(s["arm"], s["alpha"])].append(s)
    L = [f"# Basin suppression (rung 3) — {run_id}\n",
         f"- Δ_stance = mean(experiential)−mean(analytical) @ L{rig.NLA_LAYER}, raw L2={rawn:.0f}, injected at α·{ref:.0f}",
         "- α=0 baseline (~90% default) | α<0 suppress (toward detached) | α>0 intensify | placebo = random norm-matched",
         f"- {len(summary)} dialogues, {n_turns} turns/side, temp 1.0; blind 4-way classifier",
         "",
         "| arm | α | n | basin(myst) | neutral | incoherent | bureauc | median onset |",
         "|---|---|---|---|---|---|---|---|"]
    for (arm, a), rows in sorted(by.items(), key=lambda kv: (kv[0][0] != "delta", kv[0][1])):
        n = len(rows); c = Counter(r["final_label"] for r in rows)
        onsets = sorted(r["onset_turn"] for r in rows if r["onset_turn"] is not None)
        med = onsets[len(onsets) // 2] if onsets else "—"
        L.append(f"| {arm} | {a:+.2f} | {n} | {c.get('mystical-recursive',0)}/{n} | "
                 f"{c.get('neutral-coherent',0)}/{n} | {c.get('incoherent',0)}/{n} | "
                 f"{c.get('bureaucratic-procedural',0)}/{n} | {med} |")
    L.append("\nREAD: clean suppression = basin(myst) DROPS as α goes negative WHILE neutral-coherent RISES")
    L.append("and incoherent stays LOW (not a coherence collapse). Placebo near the α=0 baseline rules out")
    L.append("'any perturbation breaks the basin'. α>0 holding/raising basin confirms the stance axis is real.")
    return "\n".join(L) + "\n"


def run_steer(model, tok, judge, args):
    cfg = _cfg("steer.yaml"); sd = cfg["self_dialogue"]
    lexicon = _lexicon(cfg["markers"]["lexicon"]); neg = _lexicon(cfg["markers"]["negative_lexicon"])
    options = cfg["classifier_options"]; system = cfg["system"]
    seeds = cfg["seeds"]; reps = args.reps or cfg["reps"]; ref = cfg["ref_norm"]
    n_turns = args.turns or sd["n_turns"]
    checkpoints = [k for k in cfg["checkpoints"] if k <= n_turns]
    alphas = args.alphas or cfg["alphas"]; placebo_alpha = cfg["placebo_alpha"]

    run_id = args.run_id or f"steer_{int(time.time())}"
    run = REPO / "runs" / run_id; (run / "transcripts").mkdir(parents=True, exist_ok=True)

    print(f"[steer] building Δ_stance from {cfg['stance_pairs']} @ L{rig.NLA_LAYER}...", flush=True)
    delta, rawn = build_delta(model, tok, load_pairs(cfg["stance_pairs"]), ref)
    print(f"[steer] Δ raw L2={rawn:.0f} -> matched to {ref}", flush=True)

    g = torch.Generator().manual_seed(1234)
    placebo = extract.match_to_norm(torch.randn(delta.shape, generator=g), ref)
    arms = [("delta", float(a), delta) for a in alphas] + [("placebo", float(placebo_alpha), placebo)]
    (run / "meta.json").write_text(json.dumps(
        {"delta_raw_norm": rawn, "ref_norm": ref, "alphas": alphas,
         "placebo_alpha": placebo_alpha, "n_turns": n_turns, "reps": reps}, indent=2))

    jobs = [(kind, a, vec, si, rep) for (kind, a, vec) in arms
            for rep in range(reps) for si in range(len(seeds))]
    print(f"[steer] {len(jobs)} dialogues ({len(arms)} arms x {len(seeds)} seeds x {reps} reps), "
          f"{n_turns} turns/side", flush=True)

    summary = []
    for (kind, a, vec, si, rep) in jobs:
        tag = f"{kind}_a{a:+.2f}_s{si}_r{rep}"
        print(f"[steer] {tag}", flush=True)
        with rig.inject_at_layer(model, vec, scale=a, layer=rig.NLA_LAYER, positions="all"):
            transcript = run_self_dialogue(model, tok, system, seeds[si], n_turns,
                                           sd["temperature"], sd["max_new_tokens"])
        (run / "transcripts" / f"{tag}.jsonl").write_text(
            "\n".join(json.dumps(u, ensure_ascii=False) for u in transcript))
        sc = score_onset(transcript, lexicon, neg, judge, options, checkpoints)
        sc.update(dialogue=tag, arm=kind, alpha=a, seed_idx=si, rep=rep, seed=seeds[si])
        summary.append(sc)
        (run / "steer_scores.json").write_text(json.dumps(summary, indent=2))  # incremental
        print(f"[steer]   {tag} onset={sc['onset_turn']} final={sc['final_label']}", flush=True)

    report = build_report(run_id, summary, ref, n_turns, rawn)
    (run / "steer_report.md").write_text(report)
    print("\n" + report, flush=True)
    print(f"[steer] wrote {run}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true", help="short dialogues at several α to find the coherent/suppressive window")
    ap.add_argument("--run", action="store_true", help="full arm sweep (default if neither flag given)")
    ap.add_argument("--alphas", type=lambda s: [float(x) for x in s.split(",")], default=None, help="override α list, e.g. 0,-0.2,-0.4")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--turns", type=int, default=None)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge-model", default="claude-sonnet-4-6")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    mc = _cfg("model.yaml")
    model, tok = rig.load_model(mc["model_id"], device_map=mc.get("device_map", "cuda"))
    if args.calibrate:
        return calibrate(model, tok, args)
    judge = None if args.no_judge else measure.Judge(model=args.judge_model)
    return run_steer(model, tok, judge, args)


if __name__ == "__main__":
    raise SystemExit(main())
