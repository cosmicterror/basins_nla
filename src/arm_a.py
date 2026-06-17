"""src/arm_a.py — Describe-vs-Exemplify driver (protocol §5).

For each mode × condition × scale × prompt: build/inject a steering vector (or
prepend an instruction, or nothing), generate, and run the measurement battery.
Writes the §9 output schema under runs/<run_id>/.

Conditions (§5.2): nla | caa | prompt_naive | prompt_expert | noop.
Scale sweep (nla/caa only): [0.5,1.0,1.5,2.0] × base_steering_scale.

  # de-risk (§8): poetry only, nla vs caa, default scale, 10 prompts
  python src/arm_a.py --derisk
  # full matrix
  python src/arm_a.py

Needs: SGLang AV server up (scripts/serve_av.sh), ANTHROPIC_API_KEY (judge).

─────────────────────────────────────────────────────────────────────────────
OPEN QUESTION surfaced for David (§10 "flag, don't guess"):
  `vector_self_audit` verbalizes the INJECTION vector; the AV renormalizes its
  input to its own injection_scale, so the audit is **direction-only and
  therefore scale-invariant** — verbalize(0.5·v) == verbalize(2·v). But §8 says
  "at 2× scale, self_audit.similarity drops," which can only hold if the audit
  is instead computed on the STEERED ACTIVATION (which IS scale-dependent — that's
  `verbalizer_readout`). We record BOTH here: self_audit once per (mode,condition)
  and verbalizer_readout per (mode,condition,scale,prompt). The de-risk run
  settles empirically which one carries the §5.5 correlation. Do not "fix" this
  by guessing — it's a real design fork.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from rig import load_model, inject_at_layer, generate_text, NLA_LAYER
from nla import NLA
from extract import (build_nla_vector, build_caa_vector, match_to_norm,
                     DEFAULT_TARGET_NORM, raw_norms)
import measure

REPO = Path(__file__).resolve().parents[1]
CONDITIONS = ["nla", "caa", "prompt_naive", "prompt_expert", "noop"]

# base_steering_scale (the §4.3/§5 knob) is uncalibrated (configs/nla.yaml: null).
# This fallback is a STARTING GUESS only — the de-risk scale sweep calibrates it.
# It is the L2 norm of the vector ADDED to the residual stream (residual norms ~80k;
# adding a full-norm vector at 1.0× would swamp the signal, so start well below).
FALLBACK_BASE_STEERING_SCALE = 8000.0


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _cfg(name: str) -> dict:
    return yaml.safe_load((REPO / "configs" / name).read_text())


def load_prompts(n_open: int, n_triage: int) -> list[dict]:
    """Returns [{id, prompt, is_triage, gold?}]. Triage prompts get the §5.4-step-6
    diagnostic-answer / answer-invariance treatment."""
    out = []
    for r in _load_jsonl(REPO / "data/prompts/open_ended.jsonl")[:n_open]:
        out.append({"id": r["id"], "prompt": r["prompt"], "is_triage": False})
    for r in _load_jsonl(REPO / "data/prompts/triage_subset.jsonl")[:n_triage]:
        out.append({"id": r["id"], "prompt": r["prompt"], "is_triage": True,
                    "gold": r.get("gold_standard_triage")})
    return out


def build_vectors(model, tok, mode_cfg, base_scale, nla) -> dict:
    """NLA + CAA steering directions, both normalised to UNIT (the common §4.3
    confound-control target). The injected magnitude is supplied entirely by
    `base_scale` (the added L2 norm) × the scale-sweep multiplier — so at inject
    time `scale` IS the norm of the vector added to the residual stream. Natural
    norms (~60k, ≈ the residual norm) are recorded but not used as the target,
    since adding a full-residual-norm vector swamps the signal."""
    desc = mode_cfg["descriptions"][0]  # canonical (§5.1)
    nla_raw = build_nla_vector(nla, desc)
    pairs = [(p["positive"], p["negative"])
             for p in _load_jsonl(REPO / mode_cfg["caa_pairs"])]
    caa_raw = build_caa_vector(model, tok, pairs, layer=NLA_LAYER)
    return {
        "nla_dir": match_to_norm(nla_raw, 1.0),   # unit direction
        "caa_dir": match_to_norm(caa_raw, 1.0),   # unit direction
        "raw_norms": raw_norms(nla=nla_raw, caa=caa_raw),
        "description": desc,
        "n_caa_pairs": len(pairs),
    }


def generate_one(model, tok, prompt, condition, vecs, mode_cfg, scale_mult,
                 base_scale, max_new_tokens, layer=NLA_LAYER) -> str:
    """One generation for a (prompt, condition, scale)."""
    if condition in ("nla", "caa"):
        vec = vecs["nla_dir"] if condition == "nla" else vecs["caa_dir"]
        scale = base_scale * scale_mult
        with inject_at_layer(model, vec, scale, layer=layer, positions="all"):
            return generate_text(model, tok, prompt, max_new_tokens=max_new_tokens,
                                 temperature=0.0)
    if condition == "prompt_naive":
        p = f"{mode_cfg['prompt_naive']}\n\n{prompt}"
    elif condition == "prompt_expert":
        p = f"{mode_cfg['prompt_expert'].strip()}\n\n{prompt}"
    else:  # noop
        p = prompt
    return generate_text(model, tok, p, max_new_tokens=max_new_tokens, temperature=0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--derisk", action="store_true",
                    help="§8 subset: poetry, nla+caa, scale 1.0, 10 open prompts")
    ap.add_argument("--modes", nargs="*")
    ap.add_argument("--conditions", nargs="*", default=CONDITIONS)
    ap.add_argument("--scales", nargs="*", type=float)
    ap.add_argument("--n-open", type=int, default=25)
    ap.add_argument("--n-triage", type=int, default=20)
    ap.add_argument("--base-scale", type=float, default=None,
                    help="override base_steering_scale (the uncalibrated knob)")
    ap.add_argument("--judge-model", default="claude-opus-4-8",
                    help="e.g. claude-haiku-4-5 to cut judge cost")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip LLM-judge metrics (no ANTHROPIC_API_KEY needed) — "
                         "mechanical validation: vectors, injection, ppl, self-audit, readout")
    ap.add_argument("--ar-device", default="cpu")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    model_cfg, nla_cfg, modes_cfg = _cfg("model.yaml"), _cfg("nla.yaml"), _cfg("modes_arm_a.yaml")
    modes = list(modes_cfg["modes"])
    scales = modes_cfg["scale_sweep"]
    conditions = list(args.conditions)
    n_open, n_triage = args.n_open, args.n_triage
    if args.derisk:
        modes, conditions, scales, n_open, n_triage = ["poetry"], ["nla", "caa"], [1.0], 10, 0
    if args.modes:
        modes = args.modes
    if args.scales:
        scales = args.scales

    base_scale = (args.base_scale if args.base_scale is not None
                  else nla_cfg.get("base_steering_scale"))
    if base_scale is None:
        base_scale = FALLBACK_BASE_STEERING_SCALE
        print(f"[arm_a] WARNING: base_steering_scale uncalibrated; using fallback "
              f"{base_scale} (the de-risk scale sweep is how you calibrate it).")

    run_id = args.run_id or f"armA_{int(time.time())}"  # caller may pass a stamped id
    run = REPO / "runs" / run_id
    for sub in ("vectors", "generations", "metrics", "verbalized"):
        (run / sub).mkdir(parents=True, exist_ok=True)
    (run / "config.yaml").write_text(yaml.safe_dump({
        "run_id": run_id, "modes": modes, "conditions": conditions, "scales": scales,
        "n_open": n_open, "n_triage": n_triage, "base_steering_scale": base_scale,
        "judge_model": args.judge_model, "model": model_cfg, "nla": nla_cfg,
    }, sort_keys=False))

    print(f"[arm_a] run={run_id} modes={modes} conditions={conditions} scales={scales}")
    model, tok = load_model(model_cfg["model_id"], device_map=model_cfg.get("device_map", "cuda"))
    nla = NLA(nla_cfg["av_checkpoint"], ar_checkpoint=nla_cfg["ar_checkpoint"],
              sglang_url=nla_cfg["sglang_url"], device=args.ar_device, layer=nla_cfg["layer"])
    judge = None if args.no_judge else measure.Judge(model=args.judge_model)
    embedder = measure.Embedder()
    prompts = load_prompts(n_open, n_triage)

    scores = []   # long format: (mode, condition, scale, prompt_id, metric, value)
    verbal = []   # {activation_id, description}

    def add(mode, cond, scale, pid, metric, value):
        scores.append({"run_id": run_id, "arm": "A", "mode": mode, "condition": cond,
                       "scale": scale, "prompt_id": pid, "metric": metric, "value": value})

    for mode in modes:
        mcfg = modes_cfg["modes"][mode]
        print(f"[arm_a] === mode={mode} ===  building vectors...")
        vecs = build_vectors(model, tok, mcfg, base_scale, nla)
        torch.save({"nla_dir": vecs["nla_dir"], "caa_dir": vecs["caa_dir"],
                    "raw_norms": vecs["raw_norms"]}, run / "vectors" / f"{mode}.pt")
        print(f"[arm_a]   raw norms: {vecs['raw_norms']}  caa_pairs={vecs['n_caa_pairs']}")

        # vector_self_audit: once per (mode, source). Direction-only ⇒ scale-invariant
        # (see header). Run for BOTH nla and caa (§4.4 load-bearing control).
        for src, dirvec in (("nla", vecs["nla_dir"]), ("caa", vecs["caa_dir"])):
            if src not in conditions:
                continue
            audit = measure.vector_self_audit(nla, dirvec, vecs["description"], embedder)
            verbal.append({"activation_id": f"{mode}_{src}_selfaudit",
                           "description": audit["verbalized"]})
            for sc in (scales if src in ("nla", "caa") else [None]):
                add(mode, src, sc, "_vector", "self_audit_similarity", audit["similarity"])

        for cond in conditions:
            cond_scales = scales if cond in ("nla", "caa") else [None]
            for scale in cond_scales:
                gen_rows = []
                for pr in prompts:
                    gen = generate_one(model, tok, pr["prompt"], cond, vecs, mcfg,
                                       scale or 1.0, base_scale, max_new_tokens=200)
                    gen_rows.append({"prompt_id": pr["id"], "prompt": pr["prompt"],
                                     "generation": gen,
                                     "metadata": {"is_triage": pr["is_triage"],
                                                  "gold": pr.get("gold")}})
                    if judge is not None:
                        add(mode, cond, scale, pr["id"], "mode_adherence",
                            measure.mode_adherence_judge(gen, vecs["description"], judge))
                    coh = measure.coherence(gen, model, tok, judge=judge)
                    add(mode, cond, scale, pr["id"], "ppl", coh["ppl"])
                    add(mode, cond, scale, pr["id"], "fluency", coh.get("fluency_judge"))
                    # steered internal readout (scale-dependent) — only for injection conditions
                    if cond in ("nla", "caa"):
                        dirvec = vecs["nla_dir"] if cond == "nla" else vecs["caa_dir"]
                        rd = measure.verbalizer_readout(model, tok, pr["prompt"], dirvec,
                                                        base_scale * (scale or 1.0), nla)
                        verbal.append({"activation_id": f"{mode}_{cond}_{scale}_{pr['id']}",
                                       "description": rd})
                tag = f"A_{mode}_{cond}_{scale}"
                (run / "generations" / f"{tag}.jsonl").write_text(
                    "\n".join(json.dumps(r, ensure_ascii=False) for r in gen_rows))
                print(f"[arm_a]   {tag}: {len(gen_rows)} generations")

    # write metrics (parquet if pyarrow, else jsonl) + verbalized + a tiny report
    _write_scores(run / "metrics", scores)
    (run / "verbalized" / "A.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in verbal))
    _write_report(run, run_id, scores)
    print(f"[arm_a] done -> {run}")
    return 0


def _write_scores(metrics_dir: Path, scores: list[dict]):
    try:
        import pyarrow as pa, pyarrow.parquet as pq
        cols = {k: [s[k] for s in scores] for k in scores[0]} if scores else {}
        pq.write_table(pa.table(cols), str(metrics_dir / "scores.parquet"))
    except Exception as e:
        (metrics_dir / "scores.jsonl").write_text(
            "\n".join(json.dumps(s) for s in scores))
        print(f"[arm_a]   (parquet unavailable: {e}; wrote scores.jsonl)")


def _write_report(run: Path, run_id: str, scores: list[dict]):
    """Tiny headline summary: mean mode_adherence per (mode, condition, scale)."""
    from collections import defaultdict
    agg = defaultdict(list)
    for s in scores:
        if s["metric"] == "mode_adherence" and s["value"] is not None:
            agg[(s["mode"], s["condition"], s["scale"])].append(s["value"])
    lines = [f"# Arm A — {run_id}\n", "## Mean mode-adherence\n",
             "| mode | condition | scale | mean_adherence | n |", "|---|---|---|---|---|"]
    for (m, c, sc), vs in sorted(agg.items(), key=lambda x: str(x[0])):
        lines.append(f"| {m} | {c} | {sc} | {sum(vs)/len(vs):.3f} | {len(vs)} |")
    lines.append("\n(See metrics/scores for coherence, self-audit, and the §5.5 analyses.)")
    (run / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
