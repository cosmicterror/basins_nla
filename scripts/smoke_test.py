#!/usr/bin/env python3
"""Smoke-test the Gemma-3-12B AV injection pipeline on a vast.ai box.

Validates the riskiest, most failure-prone part of the stack BEFORE committing
to a long / expensive run: SGLang up + the gemma3_mm `input_embeds` patch
actually working + injection producing ENGLISH rather than the CJK soup that
means injection silently failed (the marker char's own embedding got verbalised).

Run order on the box:
    bash scripts/setup_remote.sh
    bash scripts/serve_av.sh fa3        # or: triton (on A100/Ampere)
    python scripts/smoke_test.py --sglang-url http://localhost:30000

Exit 0 = PASS, 1 = FAIL. Cheap by design — run it on a cheap instance first.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Our adapter lives in ../src (scripts/ is a sibling of src/).
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from nla import NLA  # noqa: E402


def _cjk_count(s: str) -> int:
    # Covers CJK symbols/enclosed (the U+321C '㈜' marker), kana, ideographs,
    # compatibility forms, and half/full-width forms.
    def is_cjk(c: str) -> bool:
        o = ord(c)
        return (0x3000 <= o <= 0x9FFF) or (0xF900 <= o <= 0xFAFF) or (0xFF00 <= o <= 0xFFEF)
    return sum(1 for c in s if is_cjk(c))


def cjk_ratio(s: str) -> float:
    nonspace = [c for c in s if not c.isspace()]
    return _cjk_count(s) / max(1, len(nonspace))


def wait_for_server(nla: NLA, timeout: float = 900.0) -> bool:
    print(f"[smoke] waiting up to {timeout:.0f}s for SGLang to come up ...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        if nla.health_check():
            print(f"[smoke] server healthy after {time.time() - t0:.0f}s")
            return True
        time.sleep(5)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-ckpt", default="kitft/nla-gemma3-12b-L32-av")
    ap.add_argument("--ar-ckpt", default="kitft/nla-gemma3-12b-L32-ar")
    ap.add_argument("--sglang-url", default="http://localhost:30000")
    ap.add_argument("--n", type=int, default=4, help="random vectors to verbalise")
    ap.add_argument("--real", action="store_true",
                    help="also verbalise a REAL Gemma L32 activation (loads base "
                         "model ~24GB — serve_av.sh must use a lower --mem-frac)")
    ap.add_argument("--ar", action="store_true",
                    help="also reconstruct+score the real activation (closes the loop)")
    ap.add_argument("--cjk-thresh", type=float, default=0.15,
                    help="max allowed CJK ratio per decode (above = injection failed)")
    args = ap.parse_args()

    print(f"[smoke] AV client -> {args.sglang_url}")
    nla = NLA(args.av_ckpt,
              ar_checkpoint=(args.ar_ckpt if args.ar else None),
              sglang_url=args.sglang_url, device="cpu")
    d = nla.av.cfg.d_model
    print(f"[smoke] d_model={d}  inj_scale={nla.av.cfg.injection_scale}  "
          f"inj_char={nla.av.cfg.injection_char!r}")

    if not wait_for_server(nla):
        print("[smoke] FAIL: server never became healthy. Check /workspace/sglang.log "
              "(OOM on head_dim=256 => wrong attention backend; 401 => HF token).")
        return 1

    ratios: list[float] = []

    print(f"\n[smoke] verbalising {args.n} random vectors (greedy):")
    rng = np.random.default_rng(0)
    for i in range(args.n):
        v = rng.standard_normal(d).astype(np.float32)   # direction; client rescales to inj_scale
        txt = nla.verbalize(v)
        r = cjk_ratio(txt)
        ratios.append(r)
        print(f"  [{i}] cjk={r:.2f}  {txt[:160]!r}")

    if args.real:
        print("\n[smoke] --real: loading base google/gemma-3-12b-it for a true L32 activation ...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # NB: gemma-3-12b-it is a multimodal wrapper; if AutoModelForCausalLM mis-maps,
        # switch to Gemma3ForConditionalGeneration and read .hidden_states the same way.
        tok = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")
        m = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-12b-it", torch_dtype=torch.bfloat16, device_map="cuda")
        ids = tok("The patient presented with acute chest pain and shortness of breath.",
                  return_tensors="pt").to(m.device)
        hs = m(**ids, output_hidden_states=True).hidden_states[32][0, -1]  # [d] last token, layer 32
        gold = hs.float().cpu().numpy()
        txt = nla.verbalize(gold)
        r = cjk_ratio(txt)
        ratios.append(r)
        print(f"  [real] cjk={r:.2f}  {txt[:240]!r}")
        if args.ar:
            mse, cos = nla.score(txt, gold)
            print(f"  [real] AR round-trip: cos={cos:.3f}  mse={mse:.3f}  "
                  f"(cos>0.5 ~ ok, >0.7 good)")

    worst = max(ratios)
    ok = worst <= args.cjk_thresh
    print(f"\n[smoke] worst CJK ratio = {worst:.2f} (thresh {args.cjk_thresh})  ->  "
          f"{'PASS ✓' if ok else 'FAIL ✗'}")
    if not ok:
        print("[smoke] High CJK == injection failed. Checklist (docs/inference.md):\n"
              "  1. gemma3_mm.py patch applied? (grep the installed sglang)\n"
              "  2. backend fa3 (Hopper) / triton (Ampere), NOT flashinfer\n"
              "  3. injection_scale loaded from sidecar (Gemma=80000)?\n"
              "  4. --disable-radix-cache set?  5. only input_embeds sent (no input_ids)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
