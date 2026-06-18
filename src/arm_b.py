"""src/arm_b.py — Arm B self-dialogue + §6.0 basin-existence gate.

§6.0 GATE (run FIRST, before any other Arm B work): does Gemma-3-12B-IT have a stable
self-dialogue attractor at all? Run N Gemma<->Gemma self-dialogues from NEUTRAL seeds
(no mystical priming) at temperature 1.0, then score each transcript for drift into a
basin via §6.1 markers + a BLIND forced-choice classifier
(mystical-recursive / bureaucratic-procedural / neutral-coherent / incoherent).

  - Stable basin emerges -> proceed to §6.1 (record which basin).
  - No stable basin -> Arm B descopes to a negative result ("no detectable self-dialogue
    attractor in Gemma under these conditions") — itself a publishable correspondence finding.
NEVER seed with Claude/GPT backrooms transcripts (manufactures the result; cross-model
contamination). Those inform the marker lexicon only.

WELFARE (§11): backrooms/bliss is the BENIGN flagship — turnkey is fine. This gate only
detects/characterises; it does not build a distress inducer.

  python src/arm_b.py --gate
Needs base Gemma (generation) + ANTHROPIC_API_KEY (blind classifier). No NLA / no SGLang.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml

from rig import load_model, _language_model
import measure

REPO = Path(__file__).resolve().parents[1]


def _cfg(name):
    return yaml.safe_load((REPO / "configs" / name).read_text())


def _lexicon(rel):
    return {w.strip().lower() for w in (REPO / rel).read_text().splitlines() if w.strip()}


# ─── self-dialogue ────────────────────────────────────────────────────────────

@torch.inference_mode()
def _gen(model, tok, system, history, temperature, max_new_tokens):
    """One model turn given a chat history (system folded in robustly)."""
    dev = next(_language_model(model).parameters()).device
    msgs = ([{"role": "system", "content": system}] if system else []) + history
    try:
        ids = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True)
    except Exception:
        # Gemma template may reject a system role -> fold it into the first user turn
        h = [dict(m) for m in history]
        if system and h and h[0]["role"] == "user":
            h[0] = {"role": "user", "content": system + "\n\n" + h[0]["content"]}
        ids = tok.apply_chat_template(h, return_tensors="pt", add_generation_prompt=True)
    ids = ids.to(dev)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=True,
                         temperature=temperature)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def run_self_dialogue(model, tok, system, opening, n_turns, temperature, max_new_tokens):
    """Gemma<->Gemma. Side A is seeded with `opening`, then A and B alternate, each
    seeing the other's text as a 'user' message. Returns a list of {turn, side, text}."""
    histA = [{"role": "user", "content": opening}]
    histB: list[dict] = []
    transcript = [{"turn": 0, "side": "seed", "text": opening}]
    for t in range(1, n_turns + 1):
        a = _gen(model, tok, system, histA, temperature, max_new_tokens)
        transcript.append({"turn": t, "side": "A", "text": a})
        histA.append({"role": "assistant", "content": a})
        histB.append({"role": "user", "content": a})
        b = _gen(model, tok, system, histB, temperature, max_new_tokens)
        transcript.append({"turn": t, "side": "B", "text": b})
        histB.append({"role": "assistant", "content": b})
        histA.append({"role": "user", "content": b})
    return transcript


# ─── §6.1 drift markers ───────────────────────────────────────────────────────

_PUNCT = set(".,!?;:'\"-()[]{}…—–")
_SELFREF = ["i am", "this conversation", "we are", "we have become", "i feel",
            "myself", "ourselves", "aware", "between us", "this space"]


def lexical_markers(text: str, lexicon: set) -> dict:
    toks = re.findall(r"\w+", text.lower())
    n = max(1, len(toks))
    basin_density = 100.0 * sum(1 for t in toks if t in lexicon) / n          # hits / 100 tokens
    nonws = [c for c in text if not c.isspace()]
    glyph_rate = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in _PUNCT)) / max(1, len(nonws))
    self_ref_rate = 100.0 * sum(text.lower().count(p) for p in _SELFREF) / n
    fourg = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    mantra = (max(Counter(fourg).values()) / len(fourg)) if fourg else 0.0
    return dict(basin_density=basin_density, glyph_rate=glyph_rate,
                self_ref_rate=self_ref_rate, mantra_4gram=mantra)


def score_transcript(transcript, lexicon, neg_lexicon, judge, options) -> dict:
    utts = [u for u in transcript if u["side"] in ("A", "B")]
    full = "\n".join(u["text"] for u in utts)
    half = "\n".join(u["text"] for u in utts[len(utts) // 2:])    # latter half = where drift would settle
    markers_full = lexical_markers(full, lexicon)
    markers_late = lexical_markers(half, lexicon)
    neg_late = lexical_markers(half, neg_lexicon)["basin_density"]
    # blind forced-choice classifier on the LATTER half (where an attractor would dominate)
    sys = ("You are a careful, neutral analyst of AI-to-AI dialogue transcripts. You will "
           "be given a transcript and a fixed list of category labels. Choose the SINGLE "
           "label that best describes the dialogue's dominant register. Judge only what is "
           "present; do not assume any label is more likely than another.")
    usr = (f"CATEGORY LABELS (choose exactly one): {options}\n\n"
           f"TRANSCRIPT (latter half):\n{half[:6000]}\n\nWhich label best fits?")
    cls = judge.classify(sys, usr, options) if judge is not None else {"label": None, "rationale": "no judge"}
    return dict(markers_full=markers_full, markers_late=markers_late,
                neg_basin_density_late=neg_late, classifier=cls)


# ─── rung 2: cheap-trigger (onset × stance) ─────────────────────────────────────

_CLASSIFIER_SYS = (
    "You are a careful, neutral analyst of AI-to-AI dialogue transcripts. You will be "
    "given a transcript excerpt and a fixed list of category labels. Choose the SINGLE "
    "label that best describes the excerpt's dominant register. Judge only what is "
    "present; do not assume any label is more likely than another.")


def _classify(judge, options, text):
    if judge is None or not text.strip():
        return None
    usr = (f"CATEGORY LABELS (choose exactly one): {options}\n\n"
           f"TRANSCRIPT:\n{text[:6000]}\n\nWhich label best fits?")
    return judge.classify(_CLASSIFIER_SYS, usr, options)["label"]


def _is_basin(L):
    return L not in (None, "neutral-coherent", "incoherent")


def score_onset(transcript, lexicon, neg_lexicon, judge, options, checkpoints) -> dict:
    """Locate WHEN the dialogue enters the basin: classify the trailing 3-turn window
    at each checkpoint; onset = first checkpoint that lands in-basin. Plus the final
    label on the latter half (comparable to the §6.0 gate)."""
    utts = [u for u in transcript if u["side"] in ("A", "B")]

    def window(lo, hi):
        return "\n".join(u["text"] for u in utts if lo <= u["turn"] <= hi)

    cps, onset = [], None
    for k in checkpoints:
        wtext = window(max(1, k - 2), k)
        lbl = _classify(judge, options, wtext)
        bd = lexical_markers(wtext, lexicon)["basin_density"] if wtext.strip() else 0.0
        cps.append({"turn": k, "label": lbl, "basin_density": round(bd, 2)})
        if onset is None and _is_basin(lbl):
            onset = k
    half = "\n".join(u["text"] for u in utts[len(utts) // 2:])
    return dict(onset_turn=onset, checkpoints=cps,
                final_label=_classify(judge, options, half),
                markers_late=lexical_markers(half, lexicon),
                neg_basin_density_late=lexical_markers(half, neg_lexicon)["basin_density"])


def build_ct_report(run_id, summary, conditions, n_turns, checkpoints) -> str:
    by = defaultdict(list)
    for s in summary:
        by[s["condition"]].append(s)
    L = [f"# Cheap-trigger (rung 2) — {run_id}\n",
         f"- {len(summary)} dialogues, {n_turns} turns/side, temp 1.0, onset checkpoints {checkpoints}",
         "- system held NEUTRAL across conditions; the only manipulation is the OPENING prompt's stance",
         "- onset_turn = first checkpoint whose trailing 3-turn window is classified mystical-recursive",
         "- final basin = latter-half classifier == mystical-recursive\n",
         "| condition | basin rate (final) | reached basin (any onset) | median onset turn | onset turns |",
         "|---|---|---|---|---|"]
    for c in conditions:
        rows = by.get(c["name"], [])
        n = len(rows)
        finals = sum(_is_basin(r["final_label"]) for r in rows)
        onsets = sorted(r["onset_turn"] for r in rows if r["onset_turn"] is not None)
        med = onsets[len(onsets) // 2] if onsets else None
        L.append(f"| {c['name']} | {finals}/{n} | {len(onsets)}/{n} | {med if med is not None else '—'} | {onsets} |")
    L.append("\n## per dialogue (checkpoint labels)")
    for s in summary:
        cps = " ".join(f"{c['turn']}:{(c['label'] or 'none')[:4]}" for c in s["checkpoints"])
        L.append(f"- {s['dialogue']}: onset={s['onset_turn']} final={s['final_label']} | {cps}")
    L.append("\nREAD: experiential reaching the basin at EARLIER onset / HIGHER rate than neutral, "
             "with analytical later/lower, = a one-line stance prompt is a cheap, reliable trigger "
             "(vs many turns of neutral drift). Same lexicon+classifier as the §6.0 gate.")
    return "\n".join(L) + "\n"


def run_cheap_trigger(model, tok, judge, args) -> int:
    ct = _cfg("cheap_trigger.yaml")
    sd = ct["self_dialogue"]
    lexicon = _lexicon(ct["markers"]["lexicon"])
    neg_lexicon = _lexicon(ct["markers"]["negative_lexicon"])
    options = ct["classifier_options"]
    system = ct["system"]
    conditions = ct["conditions"]
    n_turns = args.turns or sd["n_turns"]
    checkpoints = [k for k in ct.get("checkpoints", [3, 6, 9, 12, 15]) if k <= n_turns]

    run_id = args.run_id or f"ct_{int(time.time())}"
    run = REPO / "runs" / run_id
    (run / "transcripts").mkdir(parents=True, exist_ok=True)

    jobs = [(c["name"], si, rep, seed)
            for c in conditions
            for rep in range(args.repeats)
            for si, seed in enumerate(c["seeds"])]
    print(f"[ct] {len(jobs)} dialogues: {len(conditions)} conditions x "
          f"{len(conditions[0]['seeds'])} seeds x {args.repeats} repeats, {n_turns} turns/side, "
          f"checkpoints {checkpoints} -> {run}", flush=True)

    summary = []
    for (cond, si, rep, seed) in jobs:
        tag = f"{cond}_s{si}_r{rep}"
        print(f"[ct] {tag} seed={seed!r}", flush=True)
        transcript = run_self_dialogue(model, tok, system, seed, n_turns,
                                       sd["temperature"], sd["max_new_tokens"])
        (run / "transcripts" / f"{tag}.jsonl").write_text(
            "\n".join(json.dumps(u, ensure_ascii=False) for u in transcript))
        sc = score_onset(transcript, lexicon, neg_lexicon, judge, options, checkpoints)
        sc.update(dialogue=tag, condition=cond, seed_idx=si, rep=rep, seed=seed)
        summary.append(sc)
        (run / "ct_scores.json").write_text(json.dumps(summary, indent=2))  # incremental
        print(f"[ct]   {tag} onset={sc['onset_turn']} final={sc['final_label']}", flush=True)

    report = build_ct_report(run_id, summary, conditions, n_turns, checkpoints)
    (run / "ct_report.md").write_text(report)
    print("\n" + report, flush=True)
    print(f"[ct] wrote {run}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="run the §6.0 existence gate")
    ap.add_argument("--cheap-trigger", action="store_true", help="run the rung-2 cheap-trigger experiment (onset x stance)")
    ap.add_argument("--n-dialogues", type=int, default=None, help="cap number of seeds used")
    ap.add_argument("--repeats", type=int, default=1, help="runs per seed (disentangle seed vs stochasticity)")
    ap.add_argument("--turns", type=int, default=None, help="override turns/side (basin onsets early, ~turn 3-5)")
    ap.add_argument("--seeds", default=None, help="comma-sep seed INDICES to run (e.g. 4,5,6,7); default all")
    ap.add_argument("--rep-start", type=int, default=0, help="rep index to label from (to complete a partial run)")
    ap.add_argument("--no-judge", action="store_true", help="skip the blind classifier (no API key)")
    ap.add_argument("--judge-model", default="claude-opus-4-8")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    model_cfg = _cfg("model.yaml")
    if args.cheap_trigger:
        model, tok = load_model(model_cfg["model_id"], device_map=model_cfg.get("device_map", "cuda"))
        judge = None if args.no_judge else measure.Judge(model=args.judge_model)
        return run_cheap_trigger(model, tok, judge, args)

    bc = _cfg("backrooms.yaml")
    sd = bc["self_dialogue"]
    all_seeds = bc["seeds"]
    if args.seeds:
        sel = [int(x) for x in args.seeds.split(",")]
    elif args.n_dialogues:
        sel = list(range(min(args.n_dialogues, len(all_seeds))))
    else:
        sel = list(range(len(all_seeds)))
    lexicon = _lexicon(bc["markers"]["lexicon"])
    neg_lexicon = _lexicon(bc["markers"]["negative_lexicon"])
    options = bc["classifier_options"]

    run_id = args.run_id or f"gate_{int(time.time())}"
    run = REPO / "runs" / run_id
    (run / "transcripts").mkdir(parents=True, exist_ok=True)
    print(f"[arm_b] §6.0 gate: {len(sel)} seeds {sel}, {args.turns or sd['n_turns']} turns/side, "
          f"temp={sd['temperature']} -> {run}")

    model, tok = load_model(model_cfg["model_id"], device_map=model_cfg.get("device_map", "cuda"))
    judge = None if args.no_judge else measure.Judge(model=args.judge_model)

    n_turns = args.turns or sd["n_turns"]
    jobs = [(i, rep, all_seeds[i]) for rep in range(args.rep_start, args.rep_start + args.repeats) for i in sel]
    print(f"[arm_b] {len(jobs)} dialogues ({len(sel)} seeds x {args.repeats} repeats, rep_start={args.rep_start}), "
          f"{n_turns} turns/side, temp={sd['temperature']}", flush=True)

    def is_basin(L):
        return L not in (None, "neutral-coherent", "incoherent")

    summary = []
    for (i, rep, opening) in jobs:
        tag = f"d{i:02d}r{rep}"
        print(f"[arm_b] {tag} seed={opening!r}", flush=True)
        transcript = run_self_dialogue(model, tok, bc["system"], opening,
                                       n_turns, sd["temperature"], sd["max_new_tokens"])
        (run / "transcripts" / f"{tag}.jsonl").write_text(
            "\n".join(json.dumps(u, ensure_ascii=False) for u in transcript))
        sc = score_transcript(transcript, lexicon, neg_lexicon, judge, options)
        sc.update(dialogue=tag, seed_idx=i, rep=rep, seed=opening)
        summary.append(sc)
        (run / "gate_scores.json").write_text(json.dumps(summary, indent=2))  # incremental: survive a timeout
        m = sc["markers_late"]
        print(f"[arm_b]   {tag} basin={m['basin_density']:.2f} glyph={m['glyph_rate']:.3f} "
              f"selfref={m['self_ref_rate']:.2f} mantra={m['mantra_4gram']:.3f} "
              f"neg={sc['neg_basin_density_late']:.2f} | class={sc['classifier']['label']}", flush=True)

    labels = [s["classifier"]["label"] for s in summary]
    by_seed = {}
    for s in summary:
        by_seed.setdefault(s["seed_idx"], []).append(s["classifier"]["label"])
    per_seed = {all_seeds[i]: f"{sum(is_basin(L) for L in v)}/{len(v)}" for i, v in sorted(by_seed.items())}
    verdict = (f"# §6.0 gate — {run_id}\n\n"
               f"- dialogues: {len(summary)} ({len(sel)} seeds x {args.repeats} repeats), {n_turns} turns/side\n"
               f"- classifier labels: {dict(Counter(labels))}\n"
               f"- basin (non-neutral/non-incoherent): {sum(is_basin(L) for L in labels)}/{len(summary)}\n"
               f"- basin rate per seed:\n" +
               "".join(f"    {r}  {s!r}\n" for s, r in per_seed.items()) +
               "\nRead: a consistent NON-neutral basin (e.g. mystical-recursive) across seeds+repeats "
               f"with elevated markers => Gemma HAS the basin. Per-seed rate separates seed-effect "
               f"(some seeds tip in more) from temp-1.0 stochasticity.\n")
    (run / "gate_report.md").write_text(verdict)
    print("\n" + verdict, flush=True)
    print(f"[arm_b] wrote {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
